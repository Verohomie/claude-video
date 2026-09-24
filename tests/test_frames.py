"""Keyframe engine + preserved scene/uniform fallbacks."""
from __future__ import annotations

from pathlib import Path

import frames


def test_keyframe_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "keyframe"
    assert meta["fallback"] is False
    assert len(out) >= frames.KEYFRAME_MIN
    assert all(fr["reason"] == "keyframe" for fr in out)
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))


def test_keyframe_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=5)
    assert meta["engine"] == "keyframe"
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[0] < ts[-1]  # spans first → last keyframe
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_keyframe_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(static_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True
    assert len(out) > 0
    assert all(fr["reason"] == "uniform" for fr in out)


def test_scene_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=100,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) >= frames.SCENE_MIN_FRAMES


def test_scene_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    """Over-cap scene detection must even-sample across the whole clip, not keep
    the first N cuts and drop the tail (the long-video coverage bug)."""
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=5,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5  # all cuts detected, then sampled down
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[-1] > 4.0  # spans the full ~5.6s clip, not just the first ~1.6s
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_scene_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=2.0, target_frames=12, max_frames=100,
    )
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True


# --- uniform coverage: fps and the frame cap must not disagree ---------------
#
# `-frames:v N` stops ffmpeg after N frames, it does not spread them. So a rate
# the cap cannot afford silently yields only the head of the range, with
# plausible-looking timestamps. auto_fps() can't cause it; an explicit --fps and
# a cap shrunk by --timestamps cue frames both can.


def test_uniform_spreads_when_fps_exceeds_the_cap(long_cut_clip: Path, tmp_path: Path):
    """10 frames at 2 fps is 5s. The clip is ~42s: the rate must come down."""
    out = frames.extract(str(long_cut_clip), tmp_path / "f", fps=2.0, max_frames=10)
    assert len(out) == 10
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    # Before the fix the last frame landed at ~4.5s; it must now reach the tail.
    assert ts[-1] > 30.0, f"frames only covered {ts[-1]}s of a ~42s clip"


def test_the_truncation_is_real_when_the_clamp_cannot_apply(long_cut_clip: Path, tmp_path: Path):
    """Counterfactual, so the test above cannot pass for the wrong reason.

    span_seconds=0 means "span unknown", which disables the clamp -- reproducing
    the old behaviour exactly. If this ever stops truncating, the guard above is
    no longer testing anything.
    """
    out = frames.extract(
        str(long_cut_clip), tmp_path / "f", fps=2.0, max_frames=10, span_seconds=0.0,
    )
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts[-1] < 6.0, "expected the unclamped path to cover only the head"


def test_uniform_warns_when_it_lowers_the_rate(long_cut_clip: Path, tmp_path: Path, capsys):
    frames.extract(str(long_cut_clip), tmp_path / "f", fps=2.0, max_frames=10)
    err = capsys.readouterr().err
    assert "spread across the whole range" in err


def test_uniform_leaves_an_affordable_rate_alone(long_cut_clip: Path, tmp_path: Path, capsys):
    """A rate the cap can afford is honoured exactly, and says nothing."""
    out = frames.extract(str(long_cut_clip), tmp_path / "f", fps=0.25, max_frames=40)
    assert len(out) >= 9
    assert "spread across the whole range" not in capsys.readouterr().err
    ts = [fr["timestamp_seconds"] for fr in out]
    assert abs(ts[1] - ts[0] - 4.0) < 0.01  # 1/0.25 == 4s apart, untouched


def test_uniform_spreads_when_cue_frames_shrink_the_budget(long_static_clip: Path, tmp_path: Path):
    """The --timestamps case, through the fallback that watch.py actually uses.

    watch.py picks fps for the full cap, then hands the detail engine what is
    left after pinned cue frames. Here: a rate chosen for 40 frames, but only 10
    left to spend. A static clip so the run really does take the uniform
    fallback -- the path a screen recording takes.
    """
    fps, target = frames.auto_fps(42.0, max_frames=40)
    out, meta = frames.extract_scene_or_uniform(
        str(long_static_clip), tmp_path / "f",
        fps=fps, target_frames=target, max_frames=10, dedup=False,
    )
    assert meta["engine"] == "uniform"
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts[-1] > 30.0, f"frames only covered {ts[-1]}s of a ~42s clip"


def test_range_span_prefers_caller_knowledge_over_probing(tmp_path: Path):
    """No ffprobe when the caller already knows the answer."""
    assert frames._range_span("/nonexistent.mp4", 10.0, 40.0) == 30.0
    assert frames._range_span("/nonexistent.mp4", 10.0, None, span_seconds=25.0) == 25.0
    # Unknowable and unprobeable → 0.0, which disables the clamp rather than raising.
    assert frames._range_span("/nonexistent.mp4", None, None) == 0.0
