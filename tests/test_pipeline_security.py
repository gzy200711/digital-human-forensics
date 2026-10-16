from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from apps.api.services.pipeline import (
    PipelineError,
    _filename_from_response,
    _is_disallowed_ip,
    _validate_connected_peer,
)
from apps.api.services.storage import safe_filename


@pytest.mark.parametrize(
    ("value", "blocked"),
    [
        ("127.0.0.1", True),
        ("::ffff:10.0.0.1", True),
        ("100.64.0.1", True),
        ("169.254.169.254", True),
        ("0.0.0.0", True),
        ("8.8.8.8", False),
        ("2606:4700:4700::1111", False),
    ],
)
def test_private_and_special_address_detection(value: str, blocked: bool) -> None:
    assert _is_disallowed_ip(value) is blocked


def test_filename_from_response_is_sanitized() -> None:
    response = httpx.Response(
        200,
        headers={
            "content-disposition": 'attachment; filename="../../evil:video.mp4"'
        },
        request=httpx.Request("GET", "https://example.com/media"),
    )

    filename = _filename_from_response(response, "https://example.com/media", "video/mp4")

    assert filename == "evil_video.mp4"


def test_connected_private_peer_is_rejected() -> None:
    stream = SimpleNamespace(get_extra_info=lambda _name: ("127.0.0.1", 443))
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://example.com"),
        extensions={"network_stream": stream},
    )

    with pytest.raises(PipelineError, match="实际连接"):
        _validate_connected_peer(response, allow_private=False)


def test_safe_filename_preserves_unicode_and_removes_paths() -> None:
    assert safe_filename("../../视频 文件.mp4") == "视频_文件.mp4"
