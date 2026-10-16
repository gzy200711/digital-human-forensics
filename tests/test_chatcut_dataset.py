from __future__ import annotations

from pathlib import Path

from apps.api.services.media import MediaProbe, MediaStream
from dhfp.evaluation.schema import AuthorizationStatus, DatasetLabel, ForgeryType
from scripts.build_chatcut_dataset import (
    build_record,
    parse_variant_tags,
    variant_subset,
)


def test_parse_variant_tags_reads_directory_conventions() -> None:
    tags = parse_variant_tags(
        Path("avatar=a1,voice=v2/ratio=9x16/bg=on/clip-01.mp4"),
    )

    assert tags == {"avatar": "a1", "voice": "v2", "ratio": "9x16", "background": "on"}


def test_manifest_entry_overrides_directory_tags() -> None:
    tags = parse_variant_tags(Path("raw/clip.mp4"), {"avatar": "a9", "voice": "v9"})

    assert tags == {"avatar": "a9", "voice": "v9"}


def test_variant_subset_is_stable() -> None:
    assert variant_subset({"voice": "v2", "avatar": "a1"}) == "chatcut/avatar=a1/voice=v2"
    assert variant_subset({}, prefix="chatcut") == "chatcut/default"


def test_build_record_from_probe(tmp_path: Path) -> None:
    media = tmp_path / "clip-01.mp4"
    media.write_bytes(b"placeholder")
    probe = MediaProbe(
        path=media,
        format_name="mp4",
        format_long_name="MP4",
        duration_seconds=12.5,
        size_bytes=11,
        bit_rate=2_000_000,
        tags={},
        streams=[
            MediaStream(
                index=0,
                codec_type="video",
                codec_name="h264",
                width=1080,
                height=1920,
                frame_rate=30.0,
                sample_rate=None,
                channels=None,
                duration_seconds=12.5,
                tags={},
            ),
            MediaStream(
                index=1,
                codec_type="audio",
                codec_name="aac",
                width=None,
                height=None,
                frame_rate=None,
                sample_rate=44100,
                channels=1,
                duration_seconds=12.5,
                tags={},
            ),
        ],
    )

    record = build_record(
        media,
        probe,
        sample_id="chatcut-01",
        label=DatasetLabel.GENERATED,
        generator="chatcut",
        generator_version="2026-09",
        forgery_type=ForgeryType.COMBINED,
        tags={"avatar": "a1", "voice": "v2"},
        subset_prefix="chatcut",
    )

    assert record.width == 1080
    assert record.height == 1920
    assert record.frame_rate == 30.0
    assert record.has_audio
    assert record.authorization_status == AuthorizationStatus.SELF_OWNED
    assert record.subset == "chatcut/avatar=a1/voice=v2"
    assert record.forgery_type == ForgeryType.COMBINED
