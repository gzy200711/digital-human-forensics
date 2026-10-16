from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any

from apps.api.detector_plugins.base import PluginContext, ProvenanceResult, SignalResult
from apps.api.detector_plugins.watermark import WatermarkOutcome, WatermarkVerifier

AI_METADATA_EXPLICIT_KEYS = {
    "aigc",
    "ai-generated",
    "ai_generated",
    "synthid",
}
AI_METADATA_SOURCE_TYPE_KEYS = {
    "digital-source-type",
    "digital_source_type",
    "digitalsourcetype",
}
AI_METADATA_GENERATOR_KEYS = {
    "generated-by",
    "generated_by",
    "generator",
    "software",
    "writing_application",
}
AI_METADATA_CONTEXT_KEYS = {
    "comment",
    "description",
    "encoder",
    "handler_name",
}
AI_METADATA_FALSE_VALUES = {
    "0",
    "false",
    "n",
    "no",
    "none",
    "not_ai",
    "not-ai",
    "not detected",
    "disabled",
    "off",
    "unknown",
}
NEGATED_GENERATIVE_PATTERN = re.compile(
    r"(?:"
    r"\b(?:not|no|never|without)\b.{0,16}\b(?:ai|aigc|synthetic|generated|generative)\b|"
    r"未(?:使用|由|经).{0,12}(?:ai|人工智能|生成)"
    r")",
    re.IGNORECASE,
)
AI_GENERATIVE_VALUE_PATTERN = re.compile(
    r"(?:"
    r"\baigc\b|"
    r"\bai[-_ ](?:generated|generation|synthesi[sz]ed)\b|"
    r"\bgenerated\s+by\s+(?:an?\s+)?ai\b|"
    r"\bgenerative\s+ai\b|"
    r"\bsynthetic\s+(?:media|video|image|audio|voice|speech)\b|"
    r"\btext[-_ ]to[-_ ](?:video|image|audio|speech)\b|"
    r"人工智能生成|生成式人工智能|文生(?:视频|图|音频)|"
    r"ai\s*生成|由\s*ai\s*(?:生成|制作)"
    r")",
    re.IGNORECASE,
)
AI_PRODUCT_PATTERN = re.compile(
    r"(?:"
    r"\brunway(?:ml)?\b|"
    r"\bsora\b|"
    r"\bveo(?:\s*\d+)?\b|"
    r"\bheygen\b|"
    r"\bsynthesia\b|"
    r"\bkling\b|"
    r"\bchatcut\b|"
    r"\bpika\b|"
    r"\bluma\b|"
    r"\bmidjourney\b|"
    r"\bstable\s+diffusion\b|"
    r"\bcomfyui\b"
    r")",
    re.IGNORECASE,
)
CREATION_VERB_PATTERN = re.compile(
    r"(?:"
    r"\b(?:made|created|generated|rendered|produced|synthesi[sz]ed)\s+(?:by|with)\b|"
    r"由.{0,24}(?:生成|制作)|使用.{0,24}(?:生成|制作)"
    r")",
    re.IGNORECASE,
)

REMOTION_MENTION_PATTERN = re.compile(r"remotion", re.IGNORECASE)
REMOTION_VERSION_PATTERN = re.compile(
    r"remotion[^0-9]{0,24}(\d+\.\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
LAVF_ENCODER_PATTERN = re.compile(r"\blavf\d+(?:\.\d+)*", re.IGNORECASE)
RENDER_CHAIN_TAG_KEYS = {
    "comment",
    "description",
    "title",
    "handler_name",
    "encoder",
    "writing_application",
    "software",
}
GENERATIVE_SOURCE_TYPE_PATTERN = re.compile(
    r"algorithmicmedia|trainedalgorithmicmedia|compositesynthetic|synthetic",
    re.IGNORECASE,
)


def _is_generative_source_type(value: str) -> bool:
    return bool(GENERATIVE_SOURCE_TYPE_PATTERN.search(value))


def _metadata_value_is_generative(key: str, value: str) -> bool:
    normalized_value = " ".join(str(value).strip().split())
    lowered = normalized_value.lower()
    if not lowered or lowered in AI_METADATA_FALSE_VALUES:
        return False
    if lowered.startswith(("not ", "not_", "not-")) or NEGATED_GENERATIVE_PATTERN.search(
        normalized_value
    ):
        return False

    if key in AI_METADATA_SOURCE_TYPE_KEYS:
        return _is_generative_source_type(normalized_value)
    if key in AI_METADATA_EXPLICIT_KEYS:
        return True
    if AI_GENERATIVE_VALUE_PATTERN.search(normalized_value):
        return True

    product_match = AI_PRODUCT_PATTERN.search(normalized_value)
    ai_token_match = re.search(r"(?<![A-Za-z0-9])ai(?![A-Za-z0-9])", normalized_value, re.I)
    if key in AI_METADATA_GENERATOR_KEYS:
        return bool(product_match or ai_token_match)
    if key in AI_METADATA_CONTEXT_KEYS:
        return bool(product_match and CREATION_VERB_PATTERN.search(normalized_value))
    return False


class ProvenanceAnalyzer:
    model_name = "ffprobe-c2patool"
    model_version = "system"
    license = "LGPL-2.1 / Apache-2.0"

    def __init__(self, watermark_verifier: WatermarkVerifier | None = None) -> None:
        self._watermark_verifier = watermark_verifier

    def analyze(self, context: PluginContext) -> ProvenanceResult:
        checks: list[dict[str, Any]] = []
        indicators: list[str] = []

        metadata_hits = self._metadata_hits(context.probe.tags, context.probe.streams)
        indicators.extend(metadata_hits)
        checks.append(
            {
                "name": "容器与流元数据",
                "status": "阳性" if metadata_hits else "阴性",
                "positive": bool(metadata_hits),
                "details": {"hits": metadata_hits, "format": context.probe.format_name},
                "explanation": (
                    "元数据中存在显式 AI 生成标识。"
                    if metadata_hits
                    else "未发现显式 AI 标识；该结果不能证明内容真实。"
                ),
            }
        )

        render_chain_hits = self._render_chain_hits(context.probe.tags, context.probe.streams)
        checks.append(
            {
                "name": "渲染链路指纹",
                "status": "阳性" if render_chain_hits else "阴性",
                "positive": bool(render_chain_hits),
                "details": {"hits": render_chain_hits},
                "explanation": (
                    "容器/流标签命中 Remotion + FFmpeg Lavf 渲染链路指纹，"
                    "作为弱阳性参与融合，不足以单独判定。"
                    if render_chain_hits
                    else "未命中 Remotion / Lavf 渲染链路指纹；阴性结果不证明内容真实。"
                ),
            }
        )

        c2pa_check = self._check_c2pa(context)
        checks.append(c2pa_check)
        c2pa_details = c2pa_check["details"]
        if c2pa_details.get("ai_evidence_eligible"):
            indicators.extend(c2pa_details.get("ai_digital_source_types", []))

        watermark = self._watermark(context)
        checks.append(watermark.to_check())
        if watermark.positive:
            indicators.append(f"{watermark.provider_label}水印阳性")

        deduplicated = list(dict.fromkeys(indicators))
        return ProvenanceResult(
            explicit_ai_indicators=deduplicated,
            checks=checks,
            render_chain_hits=render_chain_hits,
        )

    def _watermark(self, context: PluginContext) -> WatermarkOutcome:
        verifier = self._watermark_verifier
        if verifier is None:
            from apps.api.config import get_settings

            settings = context.settings if context.settings is not None else get_settings()
            verifier = WatermarkVerifier.from_settings(settings)
        return verifier.verify(context.media_path)

    @staticmethod
    def _metadata_hits(tags: dict[str, str], streams: list[Any]) -> list[str]:
        merged_tags = dict(tags)
        for stream in streams:
            merged_tags.update(stream.tags or {})
        hits: list[str] = []
        for key, value in merged_tags.items():
            normalized_key = key.lower().replace(" ", "_")
            if _metadata_value_is_generative(normalized_key, str(value)):
                hits.append(f"{key}={value}")
        return hits

    @staticmethod
    def _render_chain_hits(
        tags: dict[str, str],
        streams: list[Any],
    ) -> list[dict[str, Any]]:
        """提取渲染链路指纹：Remotion 注释 + FFmpeg Lavf muxer 版本。

        指纹可被重编码剥离或伪造，因此只作为弱阳性信号返回，
        不会写入 explicit_ai_indicators（该字段触发硬规则判定）。
        """
        merged: list[tuple[str, str, str]] = [
            (key, str(value), "format") for key, value in tags.items()
        ]
        for stream in streams:
            for key, value in (stream.tags or {}).items():
                merged.append((str(key), str(value), f"stream#{stream.index}"))

        hits: list[dict[str, Any]] = []
        lavf_evidence: list[str] = []
        remotion_evidence: list[tuple[str, str | None]] = []
        for key, value, scope in merged:
            normalized_key = key.lower().replace(" ", "_")
            if normalized_key in RENDER_CHAIN_TAG_KEYS or normalized_key == "encoder":
                if LAVF_ENCODER_PATTERN.search(value):
                    lavf_evidence.append(f"{scope}:{key}={value.strip()}")
                if REMOTION_MENTION_PATTERN.search(value):
                    version_match = REMOTION_VERSION_PATTERN.search(value)
                    remotion_evidence.append(
                        (
                            f"{scope}:{key}={value.strip()}",
                            version_match.group(1) if version_match else None,
                        )
                    )

        if not remotion_evidence:
            return []

        for evidence, version in remotion_evidence:
            hits.append(
                {
                    "rule": "remotion_comment",
                    "level": "weak",
                    "remotion_version": version,
                    "evidence": evidence,
                }
            )
        if lavf_evidence:
            version = next(
                (item[1] for item in remotion_evidence if item[1] is not None),
                None,
            )
            hits.append(
                {
                    "rule": "lavf_plus_remotion",
                    "level": "fingerprint",
                    "remotion_version": version,
                    "evidence": " | ".join(
                        [*lavf_evidence[:2], *[item[0] for item in remotion_evidence[:2]]]
                    ),
                }
            )
        return hits

    @staticmethod
    def _check_c2pa(context: PluginContext) -> dict[str, Any]:
        executable = shutil.which("c2patool")
        if not executable:
            return {
                "name": "C2PA / Content Credentials",
                "status": "不可用",
                "positive": False,
                "details": {"tool_available": False},
                "explanation": "环境未安装 c2patool，本项未执行。",
            }

        commands = [
            [executable, str(context.media_path), "--detailed"],
            [executable, str(context.media_path), "-d"],
        ]
        last_error = ""
        for command in commands:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return {
                    "name": "C2PA / Content Credentials",
                    "status": "不可用",
                    "positive": False,
                    "details": {
                        "tool_available": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    "explanation": "c2patool 执行失败或超时；本项未获得有效结果。",
                }
            if completed.returncode == 0:
                try:
                    payload = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    payload = {"raw": completed.stdout.strip()[:4000]}
                summary = ProvenanceAnalyzer._c2pa_summary(payload)
                ai_evidence_eligible = ProvenanceAnalyzer._c2pa_ai_evidence_eligible(summary)
                return {
                    "name": "C2PA / Content Credentials",
                    "status": "阳性",
                    "positive": True,
                    "details": {
                        "tool_available": True,
                        "manifest": payload,
                        "summary": summary,
                        "ai_digital_source_types": summary["ai_digital_source_types"],
                        "ai_evidence_eligible": ai_evidence_eligible,
                    },
                    "explanation": ProvenanceAnalyzer._c2pa_explanation(summary),
                }
            last_error = completed.stderr.strip() or completed.stdout.strip()

        return {
            "name": "C2PA / Content Credentials",
            "status": "阴性",
            "positive": False,
            "details": {"tool_available": True, "message": last_error[:1000], "summary": {}},
            "explanation": "c2patool 未发现可验证清单；清单也可能被剥离，不能证明内容真实。",
        }

    @staticmethod
    def _c2pa_ai_evidence_eligible(summary: dict[str, Any]) -> bool:
        state = str(summary.get("validation_state") or "").strip().lower()
        failures = summary.get("failures") or []
        return state in {"valid", "trusted"} and not failures

    @staticmethod
    def _c2pa_summary(payload: Any) -> dict[str, Any]:
        """从 c2patool --detailed 输出中提取签发者、验证状态和生成来源类型。"""
        summary: dict[str, Any] = {
            "active_manifest": None,
            "validation_state": None,
            "claim_generator": None,
            "issuer": None,
            "signing_time": None,
            "failures": [],
            "digital_source_types": [],
            "ai_digital_source_types": [],
        }
        if not isinstance(payload, dict):
            return summary

        active = payload.get("active_manifest")
        summary["active_manifest"] = active
        summary["validation_state"] = payload.get("validation_state")

        manifests = payload.get("manifests")
        manifest = manifests.get(active, {}) if isinstance(manifests, dict) and active else {}
        claim = manifest.get("claim") if isinstance(manifest, dict) else None
        if isinstance(claim, dict):
            generators = claim.get("claim_generator_info")
            if isinstance(generators, list):
                names = [
                    " ".join(
                        str(part)
                        for part in (item.get("name"), item.get("version"))
                        if part
                    )
                    for item in generators
                    if isinstance(item, dict)
                ]
                summary["claim_generator"] = " / ".join(name for name in names if name) or None
            if not summary["claim_generator"]:
                summary["claim_generator"] = claim.get("claim_generator")
            signature = claim.get("signature")
            if isinstance(signature, dict):
                summary["issuer"] = signature.get("issuer") or signature.get("common_name")
                summary["signing_time"] = signature.get("time")

        validation = payload.get("validation_results")
        active_validation = (
            validation.get("activeManifest") if isinstance(validation, dict) else None
        )
        if isinstance(active_validation, dict):
            failures = active_validation.get("failure")
            if isinstance(failures, list):
                summary["failures"] = [
                    str(item.get("code"))
                    for item in failures
                    if isinstance(item, dict) and item.get("code")
                ]

        store = manifest.get("assertion_store") if isinstance(manifest, dict) else None
        if isinstance(store, dict):
            # 断言标签随规范版本变化：c2pa.actions.v2 / c2pa.actions / c2pa.actions.v1
            action_assertions = [
                value
                for key, value in store.items()
                if str(key).startswith("c2pa.actions") and isinstance(value, dict)
            ]
            for actions in action_assertions:
                for action in actions.get("actions", []):
                    if not isinstance(action, dict):
                        continue
                    source_type = action.get("digitalSourceType")
                    if not source_type:
                        continue
                    text = str(source_type)
                    summary["digital_source_types"].append(text)
                    if _is_generative_source_type(text):
                        summary["ai_digital_source_types"].append(
                            f"C2PA digitalSourceType={text.rsplit('/', maxsplit=1)[-1]}"
                        )
        summary["digital_source_types"] = list(dict.fromkeys(summary["digital_source_types"]))
        summary["ai_digital_source_types"] = list(
            dict.fromkeys(summary["ai_digital_source_types"])
        )
        return summary

    @staticmethod
    def _c2pa_explanation(summary: dict[str, Any]) -> str:
        state = summary.get("validation_state")
        issuer = summary.get("issuer")
        issuer_note = f"签发者：{issuer}。" if issuer else ""
        failures = summary.get("failures") or []
        failure_note = (
            f"验证告警：{', '.join(str(item) for item in failures[:3])}。" if failures else ""
        )
        if state == "Invalid":
            return (
                f"发现 Content Credentials 清单，但签名或内容完整性验证未通过。"
                f"{issuer_note}{failure_note}阳性只说明存在声明，不能直接采信其内容。"
            )
        if failures:
            return (
                f"发现 Content Credentials 清单，验证状态 {state}，但存在未通过项。"
                f"{issuer_note}{failure_note}需要结合信任列表与操作记录核验。"
            )
        return (
            f"发现 Content Credentials 清单，验证状态 {state or '未知'}。"
            f"{issuer_note}需要结合签发者与操作记录核验。"
        )


class ProvenanceSignalPlugin:
    signal_id = "provenance"
    display_name = "溯源与元数据"
    model_name = "ffprobe-c2patool"
    model_version = "system"
    license = "LGPL-2.1 / Apache-2.0"

    def analyze(self, context: PluginContext) -> SignalResult:
        result = ProvenanceAnalyzer().analyze(context)
        positive = bool(result.explicit_ai_indicators)
        return SignalResult(
            signal_id=self.signal_id,
            display_name=self.display_name,
            score=0.98 if positive else None,
            available=True,
            direction="指向生成" if positive else "中性",
            raw_value={
                "explicit_ai_indicators": result.explicit_ai_indicators,
                "check_count": len(result.checks),
            },
            explanation=(
                "命中显式生成标识，作为强阳性证据。"
                if positive
                else "未命中生成标识；元数据阴性不作为真实性证据。"
            ),
            model_name=self.model_name,
            model_version=self.model_version,
            license=self.license,
        )


class RenderChainSignalPlugin:
    """把容器层渲染链路指纹转成弱阳性信号，参与融合但不触发硬规则。"""

    signal_id = "render_chain"
    display_name = "渲染链路指纹"
    model_name = "container-render-chain-rule"
    model_version = "2026-09"
    license = "internal"

    def analyze(self, context: PluginContext) -> SignalResult:
        return build_render_chain_signal(ProvenanceAnalyzer().analyze(context))


def build_render_chain_signal(provenance: ProvenanceResult) -> SignalResult:
    hits = provenance.render_chain_hits
    if not hits:
        return SignalResult(
            signal_id=RenderChainSignalPlugin.signal_id,
            display_name=RenderChainSignalPlugin.display_name,
            score=None,
            available=False,
            direction="中性",
            raw_value={"hits": [], "rule_version": RenderChainSignalPlugin.model_version},
            explanation=(
                "未命中 Remotion / FFmpeg Lavf 渲染链路指纹。"
                "阴性结果不构成真实性证据，许多生成器同样会剥离容器标签。"
            ),
            model_name=RenderChainSignalPlugin.model_name,
            model_version=RenderChainSignalPlugin.model_version,
            license=RenderChainSignalPlugin.license,
        )

    fingerprint = next((hit for hit in hits if hit.get("level") == "fingerprint"), None)
    selected = fingerprint or hits[0]
    score = 0.68 if fingerprint else 0.58
    version = selected.get("remotion_version")
    version_note = f"（Remotion {version}）" if version else ""
    return SignalResult(
        signal_id=RenderChainSignalPlugin.signal_id,
        display_name=RenderChainSignalPlugin.display_name,
        score=score,
        available=True,
        direction="指向生成" if score >= 0.5 else "中性",
        raw_value={
            "rule": selected.get("rule"),
            "level": selected.get("level"),
            "remotion_version": version,
            "evidence": selected.get("evidence"),
            "hits": hits,
            "rule_version": RenderChainSignalPlugin.model_version,
        },
        explanation=(
            "容器或流标签同时命中 Remotion 与 FFmpeg Lavf muxer 版本指纹"
            f"{version_note}，说明素材很可能来自 Remotion 渲染链路。"
            "该指纹可被重编码剥离或人为写入，因此以弱阳性参与融合，不能单独形成判定。"
        ),
        model_name=RenderChainSignalPlugin.model_name,
        model_version=RenderChainSignalPlugin.model_version,
        license=RenderChainSignalPlugin.license,
    )
