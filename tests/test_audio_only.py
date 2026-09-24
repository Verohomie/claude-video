"""A source with no video stream finishes transcript-only instead of crashing."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import frames

WATCH = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "watch.py"


def _build_audio_clip(path: Path, duration: float = 4.0) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-t", str(duration), "-i", "sine=frequency=440:sample_rate=44100",
         "-c:a", "libmp3lame", str(path)],
        check=True, capture_output=True,
    )


def _build_cover_art_clip(path: Path, duration: float = 4.0) -> None:
    """An MP3 with embedded artwork: ffprobe reports a video stream of one frame."""
    art = path.parent / "art.png"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-t", "0.1", "-i", "color=c=red:s=64x64", "-frames:v", "1", str(art)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-t", str(duration), "-i", "sine=frequency=440:sample_rate=44100",
         "-i", str(art), "-map", "0:a", "-map", "1:v", "-c:a", "libmp3lame",
         "-c:v", "copy", "-disposition:v:0", "attached_pic", str(path)],
        check=True, capture_output=True,
    )


@pytest.fixture(scope="session")
def audio_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "tone.mp3"
    _build_audio_clip(path)
    return path


@pytest.fixture(scope="session")
def cover_art_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "tagged.mp3"
    _build_cover_art_clip(path)
    return path


# --- metadata ---------------------------------------------------------------


def test_metadata_reports_no_video_for_audio(audio_clip: Path):
    meta = frames.get_metadata(str(audio_clip))
    assert meta["has_video"] is False
    assert meta["has_audio"] is True
    assert meta["duration_seconds"] > 0


def test_metadata_reports_video_for_a_real_clip(cut_clip: Path):
    meta = frames.get_metadata(str(cut_clip))
    assert meta["has_video"] is True


def test_cover_art_is_not_mistaken_for_video(cover_art_clip: Path):
    """An embedded JPEG is a video stream to ffprobe, but nothing to extract."""
    meta = frames.get_metadata(str(cover_art_clip))
    assert meta["has_video"] is False


def test_the_extractors_really_do_fail_on_audio(audio_clip: Path, tmp_path: Path):
    """Counterfactual: this is what the guard in watch.py is preventing."""
    with pytest.raises(SystemExit):
        frames.extract_scene_or_uniform(
            str(audio_clip), tmp_path / "f", fps=2.0, target_frames=12, max_frames=100,
        )


# --- end to end -------------------------------------------------------------


def _run(clip: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    return subprocess.run(
        [sys.executable, str(WATCH), str(clip), "--no-whisper", *args],
        capture_output=True, text=True, env=env,
    )


def test_audio_only_source_succeeds_at_default_detail(audio_clip: Path):
    proc = _run(audio_clip)
    assert proc.returncode == 0, proc.stderr
    assert "no video stream" in proc.stderr
    assert "**Frames:** 0 selected" in proc.stdout


def test_audio_only_source_succeeds_with_an_explicit_engine(audio_clip: Path):
    """--detail efficient routes to the keyframe engine, which also needs video."""
    proc = _run(audio_clip, "--detail", "efficient")
    assert proc.returncode == 0, proc.stderr


def test_audio_only_source_does_not_report_a_bogus_resolution(audio_clip: Path):
    proc = _run(audio_clip)
    assert "**Resolution:**" not in proc.stdout
