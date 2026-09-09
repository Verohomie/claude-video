"""Screen-recording detection and the frame width it selects.

Why this exists: a 1920-wide screen capture rendered at the old 512px default is
unreadable — menu text lands about three pixels tall — and downloading a better
copy does not help, because the loss happens at frame extraction. So the
pipeline measures whether the source is a screen recording and raises the width
when it is.

The discriminator is luma FLATNESS, not motion. Motion is the intuitive choice
and it does not work: once YouTube has re-encoded and denoised an upload,
adjacent frames are near-identical for a talking head as well as a screencast,
so a motion test calls everything a screen recording.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import frames  # noqa: E402


def test_flatness_of_uniform_frame_is_total():
    assert frames._frame_flatness(bytes([128]) * 1000) == 1.0


def test_flatness_of_uniform_spread_is_low():
    """A frame using the whole luma range sits near the floor: two adjacent
    buckets out of 32 hold about 1/16th of the pixels."""
    spread = bytes(range(256)) * 8
    assert frames._frame_flatness(spread) < 0.1


def test_flatness_survives_a_bucket_boundary():
    """A flat region straddling a bucket edge must still score as one region —
    that is what the sliding two-bucket window buys."""
    shift = 8 - frames.SCREEN_FLAT_BUCKETS.bit_length() + 1
    edge = 1 << shift  # first value of the second bucket
    straddling = bytes([edge - 1, edge] * 500)
    assert frames._frame_flatness(straddling) > 0.95


def test_empty_frame_is_not_flat():
    assert frames._frame_flatness(b"") == 0.0


def test_screen_recording_detected(screen_clip):
    meta = frames.get_metadata(str(screen_clip))
    detected, evidence = frames.detect_screen_recording(str(screen_clip), meta)
    assert detected, evidence
    assert evidence["flatness"] >= frames.SCREEN_FLATNESS_THRESHOLD
    assert evidence["samples"] > 0


def test_photographic_source_not_detected(photographic_clip):
    meta = frames.get_metadata(str(photographic_clip))
    detected, evidence = frames.detect_screen_recording(str(photographic_clip), meta)
    assert not detected, evidence
    assert evidence["flatness"] < frames.SCREEN_FLATNESS_THRESHOLD


def test_narrow_source_skips_the_probe(static_clip):
    """320x240 has no detail to preserve — _scale_filter never upscales — so the
    probe is skipped entirely rather than spending ffmpeg calls on it."""
    meta = frames.get_metadata(str(static_clip))
    detected, evidence = frames.detect_screen_recording(str(static_clip), meta)
    assert not detected
    assert evidence["samples"] == 0
    assert str(frames.SCREEN_MIN_WIDTH) in evidence["reason"]


def test_zero_duration_fails_open():
    detected, evidence = frames.detect_screen_recording(
        "nonexistent.mp4", {"width": 1920, "height": 1080, "duration_seconds": 0}
    )
    assert not detected
    assert evidence["samples"] == 0


def test_undecodable_source_fails_open():
    """A probe that cannot read the file must not raise — it returns "not a
    screen recording" and the caller keeps the ordinary default."""
    detected, evidence = frames.detect_screen_recording(
        "nonexistent.mp4", {"width": 1920, "height": 1080, "duration_seconds": 60}
    )
    assert not detected
    assert evidence["reason"] == "frame probe returned nothing"


def test_screen_resolution_is_within_read_limit():
    assert frames.SCREEN_RESOLUTION <= frames.MAX_READ_DIMENSION
    assert frames.SCREEN_RESOLUTION > frames.DEFAULT_RESOLUTION
