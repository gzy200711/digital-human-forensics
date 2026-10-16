#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
AGENTS_DIR="${HOME}/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"

API_LABEL="com.dhfp.api"
WEB_LABEL="com.dhfp.web"
KEEP_AWAKE_LABEL="com.dhfp.keepawake"
TUNNEL_LABEL="com.dhfp.tunnel"

VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
LOCAL_ENV_TEMPLATE="${PROJECT_DIR}/ops/local.env.example"
LAUNCHD_TEMPLATE_DIR="${PROJECT_DIR}/ops/launchd"
NPM="$(command -v npm 2>/dev/null || true)"
NODE="$(command -v node 2>/dev/null || true)"
CLOUDFLARED="$(command -v cloudflared 2>/dev/null || true)"
LAUNCHD_PATH="${PATH}"
ACCESS_CREDENTIALS_FILE="${PROJECT_DIR}/access-credentials.txt"

info() {
  printf '[dhfp] %s\n' "$*"
}

die() {
  printf '[dhfp] error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

escape_sed() {
  printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'
}

dotenv_value() {
  local key="$1"
  local file="${PROJECT_DIR}/.env"
  if [[ ! -f "${file}" ]]; then
    return
  fi
  awk -v key="${key}" -F= '
    $1 == key {
      sub(/^[^=]*=/, "")
      print
      exit
    }
  ' "${file}"
}

set_dotenv_value() {
  local key="$1"
  local value="$2"
  local file="${PROJECT_DIR}/.env"
  local escaped_key
  escaped_key="$(escape_sed "${key}")"
  if grep -q "^${escaped_key}=" "${file}"; then
    sed -i '' "s|^${escaped_key}=.*|${key}=$(escape_sed "${value}")|" "${file}"
  else
    printf '%s=%s\n' "${key}" "${value}" >>"${file}"
  fi
}

render_plist() {
  local template="$1"
  local output="$2"
  local access_user
  local access_password
  access_user="$(dotenv_value PUBLIC_ACCESS_USER)"
  access_password="$(dotenv_value PUBLIC_ACCESS_PASSWORD)"
  sed \
    -e "s|__PROJECT_DIR__|$(escape_sed "${PROJECT_DIR}")|g" \
    -e "s|__VENV_PYTHON__|$(escape_sed "${VENV_PYTHON}")|g" \
    -e "s|__NPM__|$(escape_sed "${NPM}")|g" \
    -e "s|__NODE__|$(escape_sed "${NODE}")|g" \
    -e "s|__CLOUDFLARED__|$(escape_sed "${CLOUDFLARED}")|g" \
    -e "s|__PUBLIC_ACCESS_USER__|$(escape_sed "${access_user}")|g" \
    -e "s|__PUBLIC_ACCESS_PASSWORD__|$(escape_sed "${access_password}")|g" \
    -e "s|__PATH__|$(escape_sed "${LAUNCHD_PATH}")|g" \
    -e "s|__LOG_DIR__|$(escape_sed "${LOG_DIR}")|g" \
    "${template}" >"${output}"
}

is_loaded() {
  launchctl print "${DOMAIN}/$1" >/dev/null 2>&1
}

bootout_agent() {
  local label="$1"
  if is_loaded "${label}"; then
    launchctl bootout "${DOMAIN}/${label}" >/dev/null 2>&1 || true
  fi
}

bootstrap_agent() {
  local label="$1"
  local plist="$2"
  local attempt
  local last_error=""

  if is_loaded "${label}"; then
    launchctl bootout "${DOMAIN}/${label}" >/dev/null 2>&1 || true
    for ((attempt = 1; attempt <= 40; attempt++)); do
      if ! is_loaded "${label}"; then
        break
      fi
      sleep 0.25
    done
  fi

  for ((attempt = 1; attempt <= 5; attempt++)); do
    if last_error="$(launchctl bootstrap "${DOMAIN}" "${plist}" 2>&1)"; then
      launchctl enable "${DOMAIN}/${label}" >/dev/null 2>&1 || true
      return 0
    fi
    if is_loaded "${label}"; then
      launchctl enable "${DOMAIN}/${label}" >/dev/null 2>&1 || true
      return 0
    fi
    sleep 1
  done

  die "failed to load ${label}: ${last_error}"
}

build_runtime() {
  require_command npm
  require_command ffmpeg
  require_command ffprobe
  NPM="$(command -v npm)"
  NODE="$(command -v node)"
  FFMPEG="$(command -v ffmpeg)"
  FFPROBE="$(command -v ffprobe)"
  CLOUDFLARED="$(command -v cloudflared 2>/dev/null || true)"
  LAUNCHD_PATH="$(dirname "${NODE}"):$(dirname "${NPM}"):$(dirname "${FFMPEG}"):$(dirname "${FFPROBE}"):${HOME}/bin:${HOME}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

  if [[ ! -x "${VENV_PYTHON}" ]]; then
    require_command uv
    info "creating Python environment"
    (cd "${PROJECT_DIR}" && uv sync --extra onnx)
  fi

  if [[ ! -d "${PROJECT_DIR}/apps/web/node_modules" ]]; then
    info "installing web dependencies"
    npm --prefix "${PROJECT_DIR}/apps/web" ci
  fi

  info "building web frontend"
  API_BASE_URL=http://127.0.0.1:8000 \
    NODE_ENV=production \
    npm --prefix "${PROJECT_DIR}/apps/web" run build

  local standalone="${PROJECT_DIR}/apps/web/.next/standalone"
  [[ -f "${standalone}/server.js" ]] || die "standalone server was not generated"
  rm -rf "${standalone}/.next/static" "${standalone}/public"
  cp -R "${PROJECT_DIR}/apps/web/.next/static" "${standalone}/.next/static"
  if [[ -d "${PROJECT_DIR}/apps/web/public" ]]; then
    cp -R "${PROJECT_DIR}/apps/web/public" "${standalone}/public"
  fi
}

ensure_local_env() {
  require_command openssl
  local secret
  local access_user
  local access_password
  secret="$(openssl rand -hex 32)"
  access_user="$(dotenv_value PUBLIC_ACCESS_USER)"
  access_password="$(dotenv_value PUBLIC_ACCESS_PASSWORD)"

  if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
    sed \
      -e "s|__SECRET_KEY__|${secret}|g" \
      -e "s|__PUBLIC_ACCESS_PASSWORD__|$(openssl rand -hex 24)|g" \
      "${LOCAL_ENV_TEMPLATE}" >"${PROJECT_DIR}/.env"
    info "created .env for local runtime"
  fi

  access_user="$(dotenv_value PUBLIC_ACCESS_USER)"
  access_password="$(dotenv_value PUBLIC_ACCESS_PASSWORD)"
  if [[ -z "${access_user}" || "${access_user}" == "__PUBLIC_ACCESS_USER__" ]]; then
    access_user="dhfp"
    set_dotenv_value PUBLIC_ACCESS_USER "${access_user}"
  fi
  if [[ -z "${access_password}" || "${access_password}" == "__PUBLIC_ACCESS_PASSWORD__" ]]; then
    access_password="$(openssl rand -hex 24)"
    set_dotenv_value PUBLIC_ACCESS_PASSWORD "${access_password}"
  fi

  chmod 600 "${PROJECT_DIR}/.env"
  printf 'username=%s\npassword=%s\n' "${access_user}" "${access_password}" \
    >"${ACCESS_CREDENTIALS_FILE}"
  chmod 600 "${ACCESS_CREDENTIALS_FILE}"
}

render_agents() {
  mkdir -p "${AGENTS_DIR}" "${LOG_DIR}"
  render_plist "${LAUNCHD_TEMPLATE_DIR}/${API_LABEL}.plist.template" "${AGENTS_DIR}/${API_LABEL}.plist"
  render_plist "${LAUNCHD_TEMPLATE_DIR}/${WEB_LABEL}.plist.template" "${AGENTS_DIR}/${WEB_LABEL}.plist"
  render_plist \
    "${LAUNCHD_TEMPLATE_DIR}/${KEEP_AWAKE_LABEL}.plist.template" \
    "${AGENTS_DIR}/${KEEP_AWAKE_LABEL}.plist"
  if [[ -n "${CLOUDFLARED}" ]]; then
    render_plist \
      "${LAUNCHD_TEMPLATE_DIR}/${TUNNEL_LABEL}.plist.template" \
      "${AGENTS_DIR}/${TUNNEL_LABEL}.plist"
  fi
}

wait_for_url() {
  local url="$1"
  local label="$2"
  local attempts="${3:-60}"
  local index
  for ((index = 1; index <= attempts; index++)); do
    if curl --noproxy '*' -fsS -m 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  printf '[dhfp] warning: %s did not become ready: %s\n' "${label}" "${url}" >&2
  return 1
}

lan_ip() {
  local candidate
  candidate="$(ipconfig getifaddr en0 2>/dev/null || true)"
  if [[ -z "${candidate}" ]]; then
    candidate="$(ipconfig getifaddr en1 2>/dev/null || true)"
  fi
  printf '%s' "${candidate}"
}

local_hostname() {
  local name
  name="$(scutil --get LocalHostName 2>/dev/null || true)"
  if [[ -z "${name}" ]]; then
    name="$(hostname -s 2>/dev/null || true)"
  fi
  printf '%s' "${name}"
}

public_url() {
  local log_file
  local url=""
  for log_file in \
    "${LOG_DIR}/cloudflared.err.log" \
    "${LOG_DIR}/cloudflared.log"; do
    if [[ -f "${log_file}" ]]; then
      url="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "${log_file}" | tail -n 1 || true)"
    fi
    if [[ -n "${url}" ]]; then
      printf '%s' "${url}"
      return 0
    fi
  done
  return 0
}

print_access() {
  local address
  local hostname
  local public
  address="$(lan_ip)"
  hostname="$(local_hostname)"
  public="$(public_url)"
  printf '\nPlatform is available at:\n'
  printf '  Local:  http://localhost:3000\n'
  if [[ -n "${address}" ]]; then
    printf '  Windows / LAN: http://%s:3000\n' "${address}"
  fi
  if [[ -n "${hostname}" ]]; then
    printf '  Bonjour:       http://%s.local:3000\n' "${hostname}"
  fi
  if [[ -n "${public}" ]]; then
    printf '  Public:        %s\n' "${public}"
    printf '  Public auth:   %s\n' "${ACCESS_CREDENTIALS_FILE}"
  fi
  printf '  API docs (Mac only): http://127.0.0.1:8000/docs\n'
}

wait_for_public_url() {
  local public
  local status_code
  local index
  for ((index = 1; index <= 90; index++)); do
    public="$(public_url)"
    if [[ -n "${public}" ]]; then
      status_code="$(
        curl --noproxy '*' -sS -m 5 -o /dev/null -w '%{http_code}' "${public}" || true
      )"
      if [[ "${status_code}" == "200" || "${status_code}" == "401" ]]; then
        return 0
      fi
    fi
    sleep 1
  done
  printf '[dhfp] warning: public tunnel did not become ready\n' >&2
  return 1
}

wait_until_ready() {
  local address
  wait_for_url "http://127.0.0.1:8000/healthz" "API" 60
  wait_for_url "http://127.0.0.1:3000" "web" 90
  address="$(lan_ip)"
  if [[ -n "${address}" ]]; then
    wait_for_url "http://${address}:3000" "web LAN" 15
  fi
  if is_loaded "${TUNNEL_LABEL}"; then
    wait_for_public_url || true
  fi
  print_access
}

install_all() {
  local keep_awake="${1:-true}"
  ensure_local_env
  build_runtime
  render_agents

  info "installing launch agents"
  bootstrap_agent "${API_LABEL}" "${AGENTS_DIR}/${API_LABEL}.plist"
  bootstrap_agent "${WEB_LABEL}" "${AGENTS_DIR}/${WEB_LABEL}.plist"
  if [[ -n "${CLOUDFLARED}" ]]; then
    bootstrap_agent "${TUNNEL_LABEL}" "${AGENTS_DIR}/${TUNNEL_LABEL}.plist"
  else
    printf '[dhfp] warning: cloudflared not installed; public access disabled\n' >&2
  fi
  if [[ "${keep_awake}" == "true" ]]; then
    bootstrap_agent "${KEEP_AWAKE_LABEL}" "${AGENTS_DIR}/${KEEP_AWAKE_LABEL}.plist"
  else
    bootout_agent "${KEEP_AWAKE_LABEL}"
    rm -f "${AGENTS_DIR}/${KEEP_AWAKE_LABEL}.plist"
  fi

  wait_until_ready
}

uninstall_all() {
  info "removing launch agents"
  for label in "${API_LABEL}" "${WEB_LABEL}" "${KEEP_AWAKE_LABEL}" "${TUNNEL_LABEL}"; do
    bootout_agent "${label}"
    rm -f "${AGENTS_DIR}/${label}.plist"
  done
  info "runtime data and .env were kept"
}

start_all() {
  local label
  for label in "${API_LABEL}" "${WEB_LABEL}" "${KEEP_AWAKE_LABEL}" "${TUNNEL_LABEL}"; do
    if [[ -f "${AGENTS_DIR}/${label}.plist" ]]; then
      if ! is_loaded "${label}"; then
        launchctl bootstrap "${DOMAIN}" "${AGENTS_DIR}/${label}.plist"
      fi
      if [[ "${label}" != "${TUNNEL_LABEL}" ]]; then
        launchctl kickstart "${DOMAIN}/${label}" >/dev/null 2>&1 || true
      fi
    fi
  done
  wait_until_ready
}

stop_all() {
  local label
  for label in "${API_LABEL}" "${WEB_LABEL}" "${KEEP_AWAKE_LABEL}" "${TUNNEL_LABEL}"; do
    bootout_agent "${label}"
  done
  info "platform stopped"
}

restart_all() {
  local label
  ensure_local_env
  render_agents
  for label in "${API_LABEL}" "${WEB_LABEL}"; do
    if [[ -f "${AGENTS_DIR}/${label}.plist" ]]; then
      bootstrap_agent "${label}" "${AGENTS_DIR}/${label}.plist"
    fi
  done
  if [[ -f "${AGENTS_DIR}/${TUNNEL_LABEL}.plist" ]] && ! is_loaded "${TUNNEL_LABEL}"; then
    bootstrap_agent "${TUNNEL_LABEL}" "${AGENTS_DIR}/${TUNNEL_LABEL}.plist"
  fi
  wait_until_ready
}

set_keep_awake() {
  local enabled="$1"
  if [[ "${enabled}" == "on" ]]; then
    if [[ ! -f "${AGENTS_DIR}/${KEEP_AWAKE_LABEL}.plist" ]]; then
      render_plist \
        "${LAUNCHD_TEMPLATE_DIR}/${KEEP_AWAKE_LABEL}.plist.template" \
        "${AGENTS_DIR}/${KEEP_AWAKE_LABEL}.plist"
    fi
    bootstrap_agent "${KEEP_AWAKE_LABEL}" "${AGENTS_DIR}/${KEEP_AWAKE_LABEL}.plist"
    info "keep-awake enabled while connected to AC power"
  else
    bootout_agent "${KEEP_AWAKE_LABEL}"
    rm -f "${AGENTS_DIR}/${KEEP_AWAKE_LABEL}.plist"
    info "keep-awake disabled"
  fi
}

show_status() {
  local label
  for label in "${API_LABEL}" "${WEB_LABEL}" "${KEEP_AWAKE_LABEL}" "${TUNNEL_LABEL}"; do
    if is_loaded "${label}"; then
      printf '%-24s running\n' "${label}"
    else
      printf '%-24s stopped\n' "${label}"
    fi
  done

  if curl -fsS -m 3 "http://127.0.0.1:8000/healthz" >/dev/null 2>&1; then
    printf '%-24s ok\n' "api health"
  else
    printf '%-24s unavailable\n' "api health"
  fi
  if curl -fsS -m 3 "http://127.0.0.1:3000" >/dev/null 2>&1; then
    printf '%-24s ok\n' "web health"
    print_access
  else
    printf '%-24s unavailable\n' "web health"
  fi
}

show_logs() {
  local service="${1:-api}"
  case "${service}" in
    api) tail -n 200 -f "${LOG_DIR}/api.log" "${LOG_DIR}/api.err.log" ;;
    web) tail -n 200 -f "${LOG_DIR}/web.log" "${LOG_DIR}/web.err.log" ;;
    keepawake) tail -n 200 -f "${LOG_DIR}/keepawake.log" "${LOG_DIR}/keepawake.err.log" ;;
    tunnel) tail -n 200 -f "${LOG_DIR}/cloudflared.log" "${LOG_DIR}/cloudflared.err.log" ;;
    *) die "usage: $0 logs [api|web|keepawake|tunnel]" ;;
  esac
}

usage() {
  cat <<'EOF'
Usage: scripts/platform-service.sh <command>

Commands:
  install [--no-keep-awake]  Build and install auto-start services.
  uninstall                  Remove auto-start services but keep data.
  start                      Start the installed services.
  stop                       Stop the services until the next start/login.
  restart                    Rebuild launch agent processes.
  status                     Show service state and access URLs.
  url                        Print the current Windows / LAN access URL.
  public-url                 Print the current public HTTPS URL.
  logs [api|web|keepawake|tunnel]
                             Follow service logs.
  keep-awake <on|off>        Keep macOS awake on AC power while serving.
EOF
}

command_name="${1:-}"
case "${command_name}" in
  install)
    keep_awake="true"
    if [[ "${2:-}" == "--no-keep-awake" ]]; then
      keep_awake="false"
    fi
    install_all "${keep_awake}"
    ;;
  uninstall)
    uninstall_all
    ;;
  start)
    start_all
    ;;
  stop)
    stop_all
    ;;
  restart)
    restart_all
    ;;
  status)
    show_status
    ;;
  url)
    print_access
    ;;
  public-url)
    public_url
    printf '\n'
    ;;
  logs)
    show_logs "${2:-api}"
    ;;
  keep-awake)
    set_keep_awake "${2:-}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
