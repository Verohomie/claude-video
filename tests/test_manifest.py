"""frames.json — the frame index written alongside the extracted images.

Without it, the mapping from frame_NNNN.jpg to its timestamp exists only in the
stdout report. That report is mostly transcript and gets piped or truncated in
practice, which discards the mapping while keeping the JPEGs — leaving a
directory of images with no way to say when any of them happened. Fine for a
summary; useless the moment a frame is cited as evidence, which is what frames
are for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import watch  # noqa: E402


def _write(tmp_path, **overrides):
    kwargs = dict(
        source="https://example.com/v",
        info={"title": "A Tutorial", "uploader": "Someone"},
        meta={"duration_seconds": 120.0, "width": 1920, "height": 1080},
        frames=[
            {"path": str(tmp_path / "frames" / "frame_0001.jpg"),
             "timestamp_seconds": 0.0, "reason": "first-frame"},
            {"path": str(tmp_path / "frames" / "frame_0002.jpg"),
             "timestamp_seconds": 75.0, "reason": "scene-change"},
        ],
        detail="balanced",
        quality="1080",
        resolution=1536,
        resolution_source="screen-recording",
        screen_evidence={"reason": "6/6 frames", "flat_tiles": 7},
        effective_start=0.0,
        effective_end=120.0,
        focused=False,
        transcript_source="captions",
        transcript_segments=[{}, {}, {}],
    )
    kwargs.update(overrides)
    path = watch.write_manifest(tmp_path, **kwargs)
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_manifest_lands_beside_the_frames(tmp_path):
    path, _ = _write(tmp_path)
    assert path == tmp_path / "frames.json"


def test_every_frame_carries_a_timestamp_and_reason(tmp_path):
    _, data = _write(tmp_path)
    assert [f["file"] for f in data["frames"]] == ["frame_0001.jpg", "frame_0002.jpg"]
    second = data["frames"][1]
    assert second["timestamp_seconds"] == 75.0
    assert second["timestamp"] == "01:15"      # same rendering the report uses
    assert second["reason"] == "scene-change"


def test_manifest_records_how_the_frames_were_made(tmp_path):
    """Enough to reproduce or to explain the run later, without the stdout."""
    _, data = _write(tmp_path)
    assert data["frame_width"] == 1536
    assert data["frame_width_source"] == "screen-recording"
    assert data["screen_recording"]["flat_tiles"] == 7
    assert data["download_quality_ceiling"] == "1080"
    assert data["detail"] == "balanced"
    assert data["transcript"] == {"source": "captions", "segments": 3}


def test_local_file_has_no_download_ceiling(tmp_path):
    _, data = _write(tmp_path, quality=None)
    assert data["download_quality_ceiling"] is None


def test_focused_range_is_recorded(tmp_path):
    _, data = _write(tmp_path, focused=True, effective_start=30.0, effective_end=45.0)
    assert data["range"] == {"focused": True, "start_seconds": 30.0, "end_seconds": 45.0}


def test_no_frames_still_writes_a_manifest(tmp_path):
    """transcript detail extracts nothing; an empty index beats a missing file."""
    _, data = _write(tmp_path, frames=[], detail="transcript")
    assert data["frames"] == []


def test_unwritable_dir_does_not_fail_the_run(tmp_path):
    """Every frame and the whole report are already produced by this point, so a
    manifest that cannot be written must degrade rather than raise."""
    missing = tmp_path / "does" / "not" / "exist"
    result = watch.write_manifest(
        missing,
        source="s", info={}, meta={}, frames=[], detail="balanced", quality=None,
        resolution=512, resolution_source="default", screen_evidence={},
        effective_start=0.0, effective_end=0.0, focused=False,
        transcript_source=None, transcript_segments=[],
    )
    assert result is None
