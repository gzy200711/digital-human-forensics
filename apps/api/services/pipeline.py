from __future__ import annotations

import asyncio
import ipaddress
import mimetypes
import socket
import tempfile
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import cv2
import httpx

from apps.api.config import Settings, get_settings
from apps.api.db import get_session_factory
from apps.api.detector_plugins.audio import AudioConsistencyPlugin, AudioSpoofPlugin
from apps.api.detector_plugins.audio_onnx import OnnxAudioSpoofPlugin
from apps.api.detector_plugins.base import PluginContext, SignalResult
from apps.api.detector_plugins.digital_human import DigitalHumanPlugin
from apps.api.detector_plugins.face_quality import FaceQualityAnalyzer
from apps.api.detector_plugins.provenance import (
    ProvenanceAnalyzer,
    build_render_chain_signal,
)
from apps.api.detector_plugins.temporal import TemporalConsistencyPlugin
from apps.api.detector_plugins.visual_artifact import VisualArtifactPlugin
from apps.api.detector_plugins.visual_onnx import OnnxVisualArtifactPlugin
from apps.api.repository import JobRepository
from apps.api.services.fusion import load_fusion_engine
from apps.api.services.media import MediaProbeError, probe_media
from apps.api.services.reporting import build_report, render_static_report
from apps.api.services.storage import (
    ObjectStore,
    build_object_store,
    derived_key,
    raw_key,
    safe_filename,
)
from dhfp.domain import ArtifactLinks
from dhfp.review.http_provider import (
    HttpEvidenceReviewProvider,
    UnavailableEvidenceProvider,
)
from dhfp.review.verifier import EvidenceReviewEngine


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DetectionPipeline:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        storage: ObjectStore | None = None,
        review_engine: EvidenceReviewEngine | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.storage = storage or build_object_store(self.settings)
        self.face_analyzer = FaceQualityAnalyzer(
            pass_threshold=self.settings.quality_pass_threshold,
            marginal_threshold=self.settings.quality_marginal_threshold,
        )
        self.provenance_analyzer = ProvenanceAnalyzer()
        onnx_path = (
            Path(self.settings.visual_onnx_path).expanduser()
            if self.settings.visual_onnx_path
            else None
        )
        if onnx_path:
            self.visual_plugin = OnnxVisualArtifactPlugin(
                onnx_path,
                input_size=self.settings.onnx_input_size,
                source=self.settings.visual_onnx_source,
                license_name=self.settings.visual_onnx_license,
            )
        else:
            self.visual_plugin = VisualArtifactPlugin()
        self.temporal_plugin = TemporalConsistencyPlugin()
        self.digital_human_plugin = DigitalHumanPlugin(sample_fps=self.settings.sample_fps)
        self.audio_consistency_plugin = AudioConsistencyPlugin()
        self.audio_spoof_plugin = self._build_audio_spoof_plugin()
        self.fusion_engine = load_fusion_engine(
            self.settings.fusion_config_path,
            self.settings.calibration_path,
        )
        self.review_engine = review_engine or self._build_review_engine()

    async def run(self, job_id: str) -> None:
        try:
            await self._run_pipeline(job_id)
        except PipelineError as exc:
            await self._mark_failed(job_id, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001
            await self._mark_failed(job_id, "INTERNAL_PIPELINE_ERROR", str(exc))
            # 保留异常供 Celery / 进程日志观测，避免把系统性故障静默吞掉。
            raise

    async def _run_pipeline(self, job_id: str) -> None:
        job = await self._with_repository(lambda repository: repository.get_job(job_id))
        if job is None:
            raise PipelineError("JOB_NOT_FOUND", f"任务不存在: {job_id}")
        if job.status == "completed":
            return

        await self._set_stage(
            job_id,
            status="processing",
            stage="读取媒体",
            progress=3,
            started_at=datetime.now(UTC),
        )
        if job.source_url and not await asyncio.to_thread(
            self.storage.exists,
            job.object_key,
        ):
            object_key, filename, mime_type, size_bytes = await download_url_to_store(
                job.source_url,
                job_id=job_id,
                settings=self.settings,
                storage=self.storage,
            )
            await self._with_repository(
                lambda repository: repository.update_job(
                    job_id,
                    object_key=object_key,
                    original_filename=filename,
                    mime_type=mime_type,
                    size_bytes=size_bytes,
                )
            )
            job = await self._with_repository(lambda repository: repository.get_job(job_id))
            if job is None:
                raise PipelineError("JOB_NOT_FOUND", f"任务不存在: {job_id}")

        try:
            media_path = self.storage.local_path(job.object_key)
        except Exception as exc:
            raise PipelineError("MEDIA_READ_FAILED", f"无法读取原始媒体: {exc}") from exc

        await self._set_stage(job_id, status="processing", stage="解析容器与元数据", progress=10)
        try:
            probe = await asyncio.to_thread(probe_media, media_path)
        except MediaProbeError as exc:
            raise PipelineError("MEDIA_PROBE_FAILED", str(exc)) from exc

        if probe.video_stream is None and probe.audio_stream is None:
            raise PipelineError("NO_MEDIA_STREAM", "文件不含可检测的视频或音频轨")
        if (
            probe.duration_seconds is not None
            and probe.duration_seconds > self.settings.max_media_duration_seconds
        ):
            raise PipelineError(
                "MEDIA_TOO_LONG",
                f"媒体时长超过 {self.settings.max_media_duration_seconds} 秒限制",
            )

        context = PluginContext(
            job_id=job_id,
            original_filename=job.original_filename,
            source_object_key=job.object_key,
            media_path=media_path,
            probe=probe,
            settings=self.settings,
        )
        provenance = await asyncio.to_thread(self.provenance_analyzer.analyze, context)
        provenance_signal = await asyncio.to_thread(lambda: self._provenance_signal(provenance))
        render_chain_signal = build_render_chain_signal(provenance)

        await self._set_stage(job_id, status="processing", stage="人脸检测与质量门控", progress=23)
        face_quality = await asyncio.to_thread(
            self.face_analyzer.analyze,
            probe,
            sample_fps=self.settings.sample_fps,
            max_frames=self.settings.max_sampled_frames,
        )
        context.face_analysis = face_quality

        signals: list[SignalResult] = [provenance_signal, render_chain_signal]
        visual_result = None
        if face_quality.status == "fail":
            signals.extend(self._unavailable_signal_set("可鉴定性门控失败，该分支未运行。"))
        else:
            await self._set_stage(
                job_id, status="processing", stage="视觉伪影与时序分析", progress=42
            )
            visual_result = await asyncio.to_thread(self.visual_plugin.analyze, context)
            temporal_signal = await asyncio.to_thread(self.temporal_plugin.analyze, context)
            digital_human_signal = await asyncio.to_thread(
                self.digital_human_plugin.analyze,
                context,
            )
            signals.extend([visual_result.signal, temporal_signal, digital_human_signal])
            if visual_result.beauty_filter_score >= 0.35:
                signals.append(
                    SignalResult(
                        signal_id="beauty_filter",
                        display_name="美颜 / 滤镜干扰",
                        score=visual_result.beauty_filter_score,
                        available=True,
                        direction="中性",
                        raw_value=visual_result.beauty_filter_raw,
                        explanation=(
                            "检测到人脸纹理可能被磨皮或滤镜平滑，视觉伪影分支已按干扰程度降权。"
                        ),
                        model_name="skin-texture-confounder",
                        model_version="bootstrap-1",
                        license="internal",
                    )
                )
            await self._set_stage(job_id, status="processing", stage="音视频一致性", progress=63)
            lip_sync_signal = await asyncio.to_thread(
                self.audio_consistency_plugin.analyze,
                context,
            )
            signals.append(lip_sync_signal)
            if not probe.has_audio:
                signals.append(self._unavailable_audio_signal())
            else:
                signals.append(await asyncio.to_thread(self.audio_spoof_plugin.analyze, context))

        if face_quality.status == "fail" and probe.has_audio:
            signals.append(await asyncio.to_thread(self.audio_spoof_plugin.analyze, context))

        await self._set_stage(
            job_id, status="processing", stage="融合、校准与生成报告", progress=82
        )
        heatmap_key = derived_key(job_id, "heatmap.png")
        keyframe_key = derived_key(job_id, "keyframe.jpg")
        if visual_result and visual_result.heatmap_overlay is not None:
            success, encoded = cv2.imencode(".png", visual_result.heatmap_overlay)
            if success:
                await asyncio.to_thread(
                    self.storage.put_bytes,
                    heatmap_key,
                    encoded.tobytes(),
                    content_type="image/png",
                )
        if face_quality.samples:
            representative = max(
                face_quality.samples,
                key=lambda sample: sample.bbox[2] * sample.bbox[3],
            )
            success, encoded = cv2.imencode(".jpg", representative.frame)
            if success:
                await asyncio.to_thread(
                    self.storage.put_bytes,
                    keyframe_key,
                    encoded.tobytes(),
                    content_type="image/jpeg",
                )

        artifacts = ArtifactLinks(
            media_url=self.storage.asset_url(job.object_key),
            heatmap_url=(
                self.storage.asset_url(heatmap_key)
                if visual_result and visual_result.heatmap_overlay is not None
                else None
            ),
            keyframe_url=(self.storage.asset_url(keyframe_key) if face_quality.samples else None),
        )
        decision = self.fusion_engine.decide(
            signals,
            provenance_positive=bool(provenance.explicit_ai_indicators),
            assessability_status=face_quality.status,
            beauty_filter_score=(
                visual_result.beauty_filter_score if visual_result is not None else 0.0
            ),
        )
        report = await asyncio.to_thread(
            build_report,
            job_id=job_id,
            context=context,
            face_quality=face_quality,
            provenance=provenance,
            signals=signals,
            decision=decision,
            duration_seconds=probe.duration_seconds,
            artifacts=artifacts,
            review_engine=self.review_engine,
            fusion_version=self.fusion_engine.config.version,
            calibration_version=self.fusion_engine.calibration.version,
        )
        json_key = derived_key(job_id, "report.json")
        html_key = derived_key(job_id, "report.html")
        report.artifacts.json_report_url = self.storage.asset_url(json_key)
        report.artifacts.html_report_url = self.storage.asset_url(html_key)
        report_payload = report.model_dump(mode="json")
        await asyncio.gather(
            asyncio.to_thread(
                self.storage.put_bytes,
                json_key,
                report.model_dump_json(indent=2).encode("utf-8"),
                content_type="application/json",
            ),
            asyncio.to_thread(
                self.storage.put_bytes,
                html_key,
                render_static_report(report).encode("utf-8"),
                content_type="text/html",
            ),
        )
        await self._with_repository(
            lambda repository: repository.save_report(job_id, report_payload)
        )

    async def _set_stage(
        self,
        job_id: str,
        *,
        status: str,
        stage: str,
        progress: int,
        started_at: datetime | None = None,
    ) -> None:
        values: dict[str, object] = {
            "status": status,
            "stage": stage,
            "progress": progress,
        }
        if started_at is not None:
            values["started_at"] = started_at
        await self._with_repository(
            lambda repository: repository.update_job(
                job_id,
                **values,
            )
        )

    def _build_review_engine(self) -> EvidenceReviewEngine:
        provider = None
        mode = self.settings.review_provider_mode.strip().lower()
        if mode not in {"rules", "http", "openai-compatible"}:
            provider = UnavailableEvidenceProvider(
                f"不支持的 REVIEW_PROVIDER_MODE: {self.settings.review_provider_mode}"
            )
        elif mode in {"http", "openai-compatible"}:
            endpoint = self.settings.review_provider_endpoint
            model = self.settings.review_provider_model
            if not endpoint or not model:
                provider = UnavailableEvidenceProvider(
                    "启用外部复核 provider 时必须配置 endpoint 和 model"
                )
            else:
                provider = HttpEvidenceReviewProvider(
                    endpoint=endpoint,
                    model=model,
                    api_key=self.settings.review_provider_api_key,
                    timeout_seconds=self.settings.review_provider_timeout_seconds,
                    prompt_version=self.settings.review_prompt_version,
                    license_name=self.settings.review_provider_license,
                )
        return EvidenceReviewEngine(
            provider=provider,
            gray_zone_min=self.fusion_engine.gray_zone_min,
            gray_zone_max=self.fusion_engine.gray_zone_max,
        )

    def _build_audio_spoof_plugin(self):
        configured = self.settings.audio_spoof_onnx_path
        if not configured:
            return AudioSpoofPlugin()
        return OnnxAudioSpoofPlugin(
            Path(configured).expanduser(),
            input_mode=self.settings.audio_spoof_input_mode,
            sample_rate=self.settings.audio_spoof_sample_rate,
            segment_seconds=self.settings.audio_spoof_segment_seconds,
            max_segments=self.settings.audio_spoof_max_segments,
            aggregation=self.settings.audio_spoof_aggregation,
            source=self.settings.audio_spoof_source,
            license_name=self.settings.audio_spoof_license,
        )

    async def _mark_failed(self, job_id: str, code: str, message: str) -> None:
        try:
            await self._with_repository(
                lambda repository: repository.mark_failed(job_id, code, message)
            )
        except Exception:
            return

    async def _with_repository(self, callback):
        session_factory = get_session_factory()
        async with session_factory() as session:
            repository = JobRepository(session)
            result = await callback(repository)
            await session.commit()
            return result

    @staticmethod
    def _provenance_signal(provenance):
        positive = bool(provenance.explicit_ai_indicators)
        return SignalResult(
            signal_id="provenance",
            display_name="溯源与元数据",
            score=0.98 if positive else None,
            available=True,
            direction="指向生成" if positive else "中性",
            raw_value={
                "explicit_ai_indicators": provenance.explicit_ai_indicators,
                "check_count": len(provenance.checks),
            },
            explanation=(
                "命中显式生成标识，作为强阳性证据。"
                if positive
                else "未命中生成标识；元数据阴性不作为真实性证据。"
            ),
            model_name="ffprobe-c2patool",
            model_version="system",
            license="LGPL-2.1 / Apache-2.0",
        )

    @staticmethod
    def _unavailable_signal_set(reason: str) -> list[SignalResult]:
        return [
            SignalResult(
                signal_id="visual_artifact",
                display_name="帧级视觉伪影",
                score=None,
                available=False,
                direction="中性",
                explanation=reason,
                model_name="wavelet-srm-heuristic",
                model_version="bootstrap-1",
                license="internal",
            ),
            SignalResult(
                signal_id="temporal_consistency",
                display_name="时序一致性",
                score=None,
                available=False,
                direction="中性",
                explanation=reason,
                model_name="track-statistics-heuristic",
                model_version="bootstrap-1",
                license="internal",
            ),
            SignalResult(
                signal_id="digital_human",
                display_name="数字人专属特征",
                score=None,
                available=False,
                direction="中性",
                explanation=reason,
                model_name="digital-human-cv-heuristic",
                model_version="2026-09",
                license="internal",
            ),
            SignalResult(
                signal_id="lip_sync",
                display_name="唇音同步",
                score=None,
                available=False,
                direction="中性",
                explanation=reason,
                model_name="envelope-correlation-baseline",
                model_version="bootstrap-1",
                license="internal",
            ),
        ]

    def _unavailable_audio_signal(self) -> SignalResult:
        return SignalResult(
            signal_id="audio_spoof",
            display_name="音频反欺骗",
            score=None,
            available=False,
            direction="中性",
            explanation="输入不含音轨，未执行音频反欺骗。",
            model_name=self.audio_spoof_plugin.model_name,
            model_version=self.audio_spoof_plugin.model_version,
            license=self.audio_spoof_plugin.license,
        )


async def download_url_to_store(
    source_url: str,
    *,
    job_id: str,
    settings: Settings,
    storage: ObjectStore,
) -> tuple[str, str, str | None, int]:
    current_url = source_url
    timeout = httpx.Timeout(settings.url_download_timeout_seconds)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        for _ in range(6):
            await asyncio.to_thread(
                validate_remote_url,
                current_url,
                allow_private=settings.allow_private_urls,
            )
            async with client.stream("GET", current_url) as response:
                _validate_connected_peer(response, allow_private=settings.allow_private_urls)
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise PipelineError("URL_REDIRECT_INVALID", "远端重定向缺少 Location")
                    current_url = str(httpx.URL(current_url).join(location))
                    continue
                if response.status_code >= 400:
                    raise PipelineError(
                        "URL_DOWNLOAD_FAILED",
                        f"远端返回 HTTP {response.status_code}",
                    )
                content_type = response.headers.get("content-type", "").split(";")[0].strip()
                if content_type and not (
                    content_type.startswith("video/")
                    or content_type.startswith("audio/")
                    or content_type == "application/octet-stream"
                ):
                    raise PipelineError(
                        "URL_MEDIA_TYPE_UNSUPPORTED",
                        f"远端内容类型不受支持: {content_type}",
                    )
                filename = _filename_from_response(response, current_url, content_type)
                key = raw_key(job_id, filename)
                with tempfile.NamedTemporaryFile(delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                try:
                    size = 0
                    with temporary_path.open("wb") as output:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            size += len(chunk)
                            if size > settings.max_upload_bytes:
                                raise PipelineError(
                                    "MEDIA_TOO_LARGE",
                                    f"远端媒体超过 {settings.max_upload_bytes} 字节限制",
                                )
                            output.write(chunk)
                    with temporary_path.open("rb") as handle:
                        await asyncio.to_thread(
                            storage.put_stream,
                            key,
                            handle,
                            content_type=content_type or None,
                        )
                finally:
                    temporary_path.unlink(missing_ok=True)
                return key, filename, content_type or None, size
    raise PipelineError("URL_REDIRECT_LIMIT", "远端重定向次数过多")


def validate_remote_url(url: str, *, allow_private: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PipelineError("URL_SCHEME_UNSUPPORTED", "只支持 http/https URL")
    if allow_private:
        return
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise PipelineError("URL_DNS_FAILED", "远端域名解析失败") from exc
    for address in addresses:
        if _is_disallowed_ip(address[4][0]):
            raise PipelineError("URL_PRIVATE_ADDRESS_BLOCKED", "URL 指向内网或保留地址")


def _validate_connected_peer(response: httpx.Response, *, allow_private: bool) -> None:
    if allow_private:
        return
    network_stream = response.extensions.get("network_stream")
    if network_stream is None:
        return
    try:
        server_address = network_stream.get_extra_info("server_addr")
    except (OSError, AttributeError):
        return
    if not server_address:
        return
    peer_host = str(server_address[0]).split("%", maxsplit=1)[0]
    if _is_disallowed_ip(peer_host):
        raise PipelineError(
            "URL_PRIVATE_ADDRESS_BLOCKED",
            "URL 实际连接到了内网或保留地址",
        )


def _is_disallowed_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.ip_network("100.64.0.0/10"):
        return True
    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )


def _filename_from_response(
    response: httpx.Response,
    url: str,
    content_type: str,
) -> str:
    content_disposition = response.headers.get("content-disposition", "")
    if content_disposition:
        message = Message()
        message["content-disposition"] = content_disposition
        candidate = message.get_filename()
        if candidate:
            return safe_filename(candidate, fallback="remote-media.bin")
    parsed = urlparse(url)
    candidate = Path(url2pathname(parsed.path)).name
    if candidate:
        return safe_filename(candidate, fallback="remote-media.bin")
    suffix = mimetypes.guess_extension(content_type) or ".bin"
    return f"remote-media{suffix}"
