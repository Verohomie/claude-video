"""Shared pytest fixtures: ffmpeg-synthesized clips and scripts/ on sys.path."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Make the bundled scripts importable (mirrors watch.py's sys.path insert).
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# 14 visually distinct fills → 14 abrupt cuts → x264 emits a keyframe per cut.
COLORS = [
    "red", "green", "blue", "white", "black", "yellow", "cyan",
    "magenta", "gray", "orange", "purple", "brown", "navy", "olive",
]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr}")


def build_cut_clip(
    path: Path,
    n: int = 14,
    seg: float = 0.4,
    size: str = "320x240",
    fps: int = 10,
) -> None:
    """Concatenate ``n`` solid-color segments into one clip with ``n`` cuts.

    Each color change is a hard scene cut, so the scene selector finds ~n-1
    changes. x264's own scenecut detection is unreliable on flat fills, so we
    force a keyframe at every ``seg`` boundary — giving ~n real keyframes for
    the keyframe engine to find.
    """
    inputs: list[str] = []
    for i in range(n):
        color = COLORS[i % len(COLORS)]
        inputs += ["-f", "lavfi", "-t", str(seg), "-i", f"color=c={color}:s={size}:r={fps}"]
    streams = "".join(f"[{i}:v]" for i in range(n))
    filt = f"{streams}concat=n={n}:v=1:a=0[out]"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", filt, "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-force_key_frames", f"expr:gte(t,n_forced*{seg})",
        str(path),
    ])


def build_static_clip(
    path: Path,
    duration: float = 3.0,
    size: str = "320x240",
    fps: int = 10,
) -> None:
    """One solid color: 1 keyframe, no scene changes → triggers both fallbacks."""
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-t", str(duration), "-i", f"color=c=blue:s={size}:r={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "600",
        str(path),
    ])


def build_screen_clip(path: Path, duration: float = 4.0, fps: int = 10) -> None:
    """A flat background with UI-like boxes: high luma flatness, like a screencast.

    1280x720 because detect_screen_recording ignores anything narrower — below
    that there is no detail for a higher frame width to preserve.
    """
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-t", str(duration), "-i", f"color=c=white:s=1280x720:r={fps}",
        "-vf", (
            "drawbox=x=40:y=40:w=380:h=560:color=black:t=fill,"
            "drawbox=x=520:y=80:w=300:h=14:color=gray:t=fill,"
            "drawbox=x=520:y=140:w=220:h=14:color=gray:t=fill"
        ),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ])


def build_photographic_clip(path: Path, duration: float = 4.0, fps: int = 10) -> None:
    """Broad luma histogram with motion — stands in for camera footage."""
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-t", str(duration), "-i", f"testsrc2=s=1280x720:r={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ])


def build_pip_clip(path: Path, duration: float = 4.0, fps: int = 10) -> None:
    """Screen capture with a busy webcam inset in the corner.

    The layout that defeats whole-frame flatness: averaging a large flat screen
    against a small detailed inset lands near the boundary and reads as camera
    footage. Per-tile scoring has to see through it.
    """
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-t", str(duration), "-i", f"color=c=white:s=1280x720:r={fps}",
        "-f", "lavfi", "-t", str(duration), "-i", f"testsrc2=s=340x200:r={fps}",
        "-filter_complex",
        "[0:v]drawbox=x=40:y=40:w=380:h=560:color=black:t=fill,"
        "drawbox=x=520:y=80:w=300:h=14:color=gray:t=fill[bg];"
        "[bg][1:v]overlay=x=0:y=520[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ])


def build_dark_clip(path: Path, duration: float = 4.0, fps: int = 10) -> None:
    """Near-black with faint specks — flat everywhere but nothing to resolve.

    Passes the flatness test on every tile, so only the detail guard rejects it.
    A night-sky timelapse is the real-world case.
    """
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-t", str(duration), "-i", f"color=c=black:s=1280x720:r={fps}",
        "-vf", "drawbox=x=300:y=200:w=2:h=2:color=white:t=fill,"
               "drawbox=x=900:y=500:w=2:h=2:color=white:t=fill",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ])


@pytest.fixture(scope="session")
def pip_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "pip.mp4"
    build_pip_clip(path)
    return path


@pytest.fixture(scope="session")
def dark_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "dark.mp4"
    build_dark_clip(path)
    return path


@pytest.fixture(scope="session")
def screen_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "screen.mp4"
    build_screen_clip(path)
    return path


@pytest.fixture(scope="session")
def photographic_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "photo.mp4"
    build_photographic_clip(path)
    return path


@pytest.fixture(scope="session")
def cut_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "cuts.mp4"
    build_cut_clip(path)
    return path


@pytest.fixture(scope="session")
def static_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "static.mp4"
    build_static_clip(path)
    return path
