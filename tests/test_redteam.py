from __future__ import annotations

import json
import subprocess

from dhfp.redteam.transforms import (
    SUPPORTED_PERTURBATIONS,
    build_redteam_plan,
    run_redteam_suite,
    write_redteam_manifest,
)


def test_redteam_plan_covers_supported_perturbations(tmp_path, synthetic_video) -> None:
    jobs = build_redteam_plan(synthetic_video, tmp_path / "redteam")

    assert [job.family for job in jobs] == list(SUPPORTED_PERTURBATIONS)
    assert len({job.case_id for job in jobs}) == len(jobs)
    subtitle_job = next(job for job in jobs if job.family == "subtitle_overlay")
    assert "subtitles=" in " ".join(subtitle_job.command)
    assert subtitle_job.subtitle_text


def test_redteam_suite_runs_low_resolution_and_writes_manifest(
    tmp_path,
    synthetic_video,
) -> None:
    output = tmp_path / "redteam"
    results = run_redteam_suite(
        synthetic_video,
        output,
        families=["low_resolution"],
    )

    assert len(results) == 1
    assert results[0].success, results[0].error
    manifest = write_redteam_manifest(results, output / "manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["success_count"] == 1

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            results[0].output_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["width"] == 320
