#!/bin/zsh
# 打印当前 Cloudflare 快速隧道（quick tunnel）的公网网址。
# 每次 tunnel 服务重启都会分配新网址，以 cloudflared 日志中最后一次签发的为准。

LOG_FILE="$(dirname "$0")/../logs/cloudflared.err.log"

URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_FILE" | grep -v 'api\.trycloudflare\.com' | tail -1)"

if [[ -z "$URL" ]]; then
  echo "未在日志中找到 trycloudflare 网址：$LOG_FILE" >&2
  exit 1
fi

echo "$URL"
echo "用户名：dhfp（密码见 access-credentials.txt）" >&2
