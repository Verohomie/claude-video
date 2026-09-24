"""A dropped duplicate is not a gap: survivors record the stretch they stand for.

Dedup removes frames that look the same as the one before, which is correct and
saves a lot of tokens. The hazard is what the result then looks like: a 40-minute
screen recording of one slide collapses to a handful of frames whose last
timestamp is early, and that reads as "only the first few minutes were examined"
when the whole range was in fact sampled.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import frames

WATCH = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "watch.py"


def _touch(dirpath: Path, times: list[float]) -> list[dict]:
    dirpath.mkdir(parents=True, exist_ok=True)
    out = []
    for i, t in enumerate(times):
        p = dirpath / f"frame_{i:04d}.jpg"
        p.write_bytes(b"x")
        out.append({"index": i, "timestamp_seconds": t, "path": str(p), "reason": "uniform"})
    return out


FLAT0 = bytes([0, 0, 0, 0])
FLAT255 = bytes([255, 255, 255, 255])


# --- the unit: _dedupe_by_deltas records what it absorbed --------------------


def test_survivor_records_the_stretch_it_absorbed(tmp_path: Path):
    cands = _touch(tmp_path / "f", [0.0, 10.0, 20.0, 30.0])
    thumbs = [FLAT0, FLAT0, FLAT0, FLAT0]  # all identical
    survivors, dropped = frames._dedupe_by_deltas(cands, thumbs, threshold=2.0)
    assert len(survivors) == 1
    assert dropped == 3
    assert survivors[0]["represents_until"] == 30.0


def test_distinct_frames_carry_no_marker(tmp_path: Path):
    """The marker must mean something, so it is absent when nothing was dropped."""
    cands = _touch(tmp_path / "f", [0.0, 10.0])
    survivors, dropped = frames._dedupe_by_deltas(cands, [FLAT0, FLAT255], threshold=2.0)
    assert dropped == 0
    assert all("represents_until" not in s for s in survivors)


def test_each_survivor_records_its_own_run(tmp_path: Path):
    """Two runs of sameness separated by a real change."""
    cands = _touch(tmp_path / "f", [0.0, 5.0, 10.0, 20.0, 25.0])
    thumbs = [FLAT0, FLAT0, FLAT0, FLAT255, FLAT255]
    survivors, dropped = frames._dedupe_by_deltas(cands, thumbs, threshold=2.0)
    assert [s["timestamp_seconds"] for s in survivors] == [0.0, 20.0]
    assert survivors[0]["represents_until"] == 10.0
    assert survivors[1]["represents_until"] == 25.0


def test_the_marker_never_precedes_its_own_frame(tmp_path: Path):
    cands = _touch(tmp_path / "f", [0.0, 1.0, 2.0, 3.0])
    survivors, _ = frames._dedupe_by_deltas(cands, [FLAT0] * 4, threshold=2.0)
    for s in survivors:
        if "represents_until" in s:
            assert s["represents_until"] >= s["timestamp_seconds"]


# --- end to end: the report says it in words --------------------------------


def _run(clip: Path, out_dir: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    return subprocess.run(
        [sys.executable, str(WATCH), str(clip), "--no-whisper",
         "--out-dir", str(out_dir), *args],
        capture_output=True, text=True, env=env,
    )


def test_report_marks_the_unchanged_stretch(clustered_cut_clip: Path, tmp_path: Path):
    """9 cuts then 34s of nothing: the survivor must say it stands for the tail."""
    proc = _run(clustered_cut_clip, tmp_path / "w")
    assert proc.returncode == 0, proc.stderr
    assert "unchanged, reason=" in proc.stdout, proc.stdout


def test_report_states_the_sampled_range(clustered_cut_clip: Path, tmp_path: Path):
    proc = _run(clustered_cut_clip, tmp_path / "w")
    assert proc.returncode == 0, proc.stderr
    assert "**Frame coverage:** sampled across" in proc.stdout


def test_coverage_reaches_the_end_despite_an_early_last_frame(
    clustered_cut_clip: Path, tmp_path: Path
):
    """The precise failure: last frame early, coverage nonetheless to the end."""
    work = tmp_path / "w"
    proc = _run(clustered_cut_clip, work)
    assert proc.returncode == 0, proc.stderr

    manifest = json.loads((work / "frames.json").read_text(encoding="utf-8"))
    frame_list = manifest["frames"]
    last_ts = frame_list[-1]["timestamp_seconds"]
    covered_to = max(
        fr.get("represents_until_seconds", fr["timestamp_seconds"]) for fr in frame_list
    )
    end = manifest["range"]["end_seconds"]

    assert last_ts < end * 0.5, "expected the naive last timestamp to look early"
    assert covered_to > end * 0.8, "coverage should reach the tail"


def test_manifest_omits_the_field_when_nothing_was_dropped(cut_clip: Path, tmp_path: Path):
    work = tmp_path / "w"
    proc = _run(cut_clip, work, "--no-dedup")
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((work / "frames.json").read_text(encoding="utf-8"))
    assert all("represents_until_seconds" not in fr for fr in manifest["frames"])
