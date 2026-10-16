from __future__ import annotations

from pathlib import Path

from apps.api.detector_plugins.base import PluginContext
from apps.api.detector_plugins.provenance import (
    ProvenanceAnalyzer,
    build_render_chain_signal,
)
from apps.api.services.media import MediaProbe, MediaStream


def _probe(
    tmp_path: Path,
    format_tags: dict[str, str],
    *,
    stream_tags: dict[str, str] | None = None,
) -> MediaProbe:
    video = MediaStream(
        index=0,
        codec_type="video",
        codec_name="h264",
        width=1080,
        height=1920,
        frame_rate=30.0,
        sample_rate=None,
        channels=None,
        duration_seconds=10.0,
        tags=stream_tags or {},
    )
    return MediaProbe(
        path=tmp_path / "sample.mp4",
        format_name="mov,mp4,m4a",
        format_long_name="QuickTime / MOV",
        duration_seconds=10.0,
        size_bytes=4096,
        bit_rate=800000,
        tags=format_tags,
        streams=[video],
    )


def _context(probe: MediaProbe) -> PluginContext:
    return PluginContext(
        job_id="render-chain",
        original_filename="sample.mp4",
        source_object_key="samples/sample.mp4",
        media_path=probe.path,
        probe=probe,
        settings=None,
    )


def test_lavf_plus_remotion_is_weak_positive_not_hard_rule(tmp_path: Path) -> None:
    probe = _probe(
        tmp_path,
        {"comment": "Made with Remotion 4.0.190", "encoder": "Lavf60.16.100"},
    )
    result = ProvenanceAnalyzer().analyze(_context(probe))

    assert result.explicit_ai_indicators == []
    assert any(hit["level"] == "fingerprint" for hit in result.render_chain_hits)

    signal = build_render_chain_signal(result)
    assert signal.available
    assert signal.score is not None
    assert 0.5 < signal.score < 0.75
    assert signal.raw_value["remotion_version"] == "4.0.190"


def test_plain_lavf_encoder_is_not_a_fingerprint(tmp_path: Path) -> None:
    probe = _probe(tmp_path, {"encoder": "Lavf60.16.100"})
    result = ProvenanceAnalyzer().analyze(_context(probe))

    assert result.render_chain_hits == []
    signal = build_render_chain_signal(result)
    assert not signal.available
    assert signal.score is None


def test_remotion_comment_without_lavf_stays_weak(tmp_path: Path) -> None:
    probe = _probe(tmp_path, {"comment": "rendered by Remotion"})
    result = ProvenanceAnalyzer().analyze(_context(probe))

    assert [hit["level"] for hit in result.render_chain_hits] == ["weak"]
    signal = build_render_chain_signal(result)
    assert signal.available
    assert signal.score == 0.58
