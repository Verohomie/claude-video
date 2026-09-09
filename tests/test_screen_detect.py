"""Screen-recording detection and the frame width it selects.

Why this exists: a 1920-wide screen capture rendered at the old 512px default is
unreadable — menu text lands about three pixels tall — and downloading a better
copy does not help, because the loss happens at frame extraction. So the
pipeline measures whether the source is a screen recording and raises the width
when it is.

Two axes, both required per frame, majority vote across samples:

- FLATNESS PER TILE, not per frame. Whole-frame flatness fails on the standard
  picture-in-picture tutorial layout: a large flat screen averaged against a
  small busy webcam inset lands near the boundary and reads as camera footage.
- DETAIL as a guard. A near-black frame is flat on every tile but has nothing to
  resolve, so a wider frame would spend tokens for no gain.

Motion is deliberately not used, though it is the intuitive choice. Once a
platform has re-encoded and denoised an upload, adjacent frames are
near-identical for a talking head as well as a screencast, so a motion test
reports everything as a screen recording.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import frames  # noqa: E402


def _solid(value: int = 128) -> bytes:
    return bytes([value]) * (frames.SCREEN_PROBE_WIDTH * frames.SCREEN_PROBE_HEIGHT)


# ---- flatness -------------------------------------------------------------

def test_flatness_of_uniform_frame_is_total():
    assert frames._frame_flatness(bytes([128]) * 1000) == 1.0


def test_flatness_of_uniform_spread_is_low():
    """A frame using the whole luma range sits near the floor: two adjacent
    buckets out of 32 hold about 1/16th of the pixels."""
    assert frames._frame_flatness(bytes(range(256)) * 8) < 0.1


def test_flatness_survives_a_bucket_boundary():
    """A flat region straddling a bucket edge must still score as one region —
    that is what the sliding two-bucket window buys."""
    shift = 8 - frames.SCREEN_FLAT_BUCKETS.bit_length() + 1
    edge = 1 << shift
    assert frames._frame_flatness(bytes([edge - 1, edge] * 500)) > 0.95


def test_empty_frame_is_not_flat():
    assert frames._frame_flatness(b"") == 0.0


# ---- detail ---------------------------------------------------------------

def test_detail_of_a_solid_frame_is_zero():
    assert frames._frame_detail(_solid()) == 0.0


def test_detail_of_an_alternating_frame_is_high():
    frame = bytes([0, 255] * (frames.SCREEN_PROBE_WIDTH * frames.SCREEN_PROBE_HEIGHT // 2))
    assert frames._frame_detail(frame) > frames.SCREEN_DETAIL_FLOOR


# ---- tiles ----------------------------------------------------------------

def test_solid_frame_has_every_tile_flat():
    assert frames._flat_tile_count(_solid()) == frames.SCREEN_TILE_GRID ** 2


def test_noisy_frame_has_no_flat_tiles():
    frame = bytes(range(256)) * (
        frames.SCREEN_PROBE_WIDTH * frames.SCREEN_PROBE_HEIGHT // 256
    )
    assert frames._flat_tile_count(frame) == 0


def test_a_flat_frame_with_no_detail_is_not_screenlike():
    """The whole point of the detail guard: flat on all 9 tiles, still rejected."""
    verdict, detail, tiles = frames._frame_is_screenlike(_solid())
    assert tiles == frames.SCREEN_TILE_GRID ** 2
    assert detail < frames.SCREEN_DETAIL_FLOOR
    assert not verdict


# ---- end to end -----------------------------------------------------------

def test_screen_recording_detected(screen_clip):
    meta = frames.get_metadata(str(screen_clip))
    detected, evidence = frames.detect_screen_recording(str(screen_clip), meta)
    assert detected, evidence
    assert evidence["flat_tiles"] >= frames.SCREEN_MIN_FLAT_TILES
    assert evidence["screenlike"] == evidence["samples"]


def test_picture_in_picture_screencast_detected(pip_clip):
    """The regression this tiling exists for. Whole-frame flatness reads the
    n8n masterclass at 0.41-0.49 — under the old 0.45 threshold — because the
    webcam inset drags the average down. Per tile it is unambiguous."""
    meta = frames.get_metadata(str(pip_clip))
    detected, evidence = frames.detect_screen_recording(str(pip_clip), meta)
    assert detected, evidence


def test_photographic_source_not_detected(photographic_clip):
    meta = frames.get_metadata(str(photographic_clip))
    detected, evidence = frames.detect_screen_recording(str(photographic_clip), meta)
    assert not detected, evidence
    assert evidence["flat_tiles"] < frames.SCREEN_MIN_FLAT_TILES


def test_dark_source_not_detected(dark_clip):
    """Flat on every tile, so flatness alone would call it a screen recording —
    a night-sky timelapse measured 9/9 tiles flat. The detail guard is the only
    thing that rejects it, and raising the width would resolve nothing anyway."""
    meta = frames.get_metadata(str(dark_clip))
    detected, evidence = frames.detect_screen_recording(str(dark_clip), meta)
    assert not detected, evidence
    assert evidence["flat_tiles"] >= frames.SCREEN_MIN_FLAT_TILES  # flat...
    assert evidence["detail"] < frames.SCREEN_DETAIL_FLOOR         # ...but empty


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


def test_tile_threshold_sits_between_the_measured_groups():
    """Measured medians: screen recordings 4-7 flat tiles, camera footage 0-1."""
    assert 1 < frames.SCREEN_MIN_FLAT_TILES < 4
