"""Scene detection is gated on coverage, not just on how many cuts it found.

The risk runs both ways. Too loose and a clustered detection is accepted, showing
ten seconds of an hour under engine "scene". Too tight and the scene engine --
the only one that puts frames where something actually changed -- is quietly
traded for uniform sampling on ordinary videos. Both directions are asserted.
"""
from __future__ import annotations

from pathlib import Path

import frames


# --- _worst_gap --------------------------------------------------------------


def test_worst_gap_measures_from_the_range_edges():
    """A candidate at 0 and one at 10 in a 0-100 range leaves 90s uncovered."""
    assert frames._worst_gap([0.0, 10.0], 0.0, 100.0) == 90.0


def test_worst_gap_counts_a_late_start():
    assert frames._worst_gap([50.0], 0.0, 60.0) == 50.0


def test_worst_gap_of_even_candidates_is_one_interval():
    assert frames._worst_gap([25.0, 50.0, 75.0], 0.0, 100.0) == 25.0


def test_worst_gap_handles_an_empty_or_degenerate_range():
    assert frames._worst_gap([], 0.0, 0.0) == 0.0
    assert frames._worst_gap([1.0], 5.0, 5.0) == 0.0


# --- _covers_range: the accept direction -------------------------------------


def test_evenly_spread_cuts_are_accepted():
    times = [float(i) * 10 for i in range(10)]  # 0..90 across a 100s range
    assert frames._covers_range(times, 0.0, 100.0) is True


def test_mildly_uneven_cuts_are_still_accepted():
    """Real videos are never metronomic; the gate must tolerate ordinary variation."""
    times = [0.0, 8.0, 12.0, 31.0, 38.0, 55.0, 61.0, 79.0, 96.0]
    assert frames._covers_range(times, 0.0, 100.0) is True


def test_a_single_candidate_is_not_judged():
    """Too little information to call it clustered; SCENE_MIN_FRAMES handles this."""
    assert frames._covers_range([50.0], 0.0, 100.0) is True


def test_empty_and_degenerate_inputs_are_accepted():
    assert frames._covers_range([], 0.0, 100.0) is True
    assert frames._covers_range([1.0, 2.0], 5.0, 5.0) is True


# --- _covers_range: the reject direction -------------------------------------


def test_cuts_bunched_at_the_head_are_rejected():
    """The case from the wild: cuts inside a title sequence, then nothing."""
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    assert frames._covers_range(times, 0.0, 3600.0) is False


def test_a_hole_in_the_middle_is_rejected():
    """First and last are covered, so a span check passes and this must not."""
    times = [0.0, 1.0, 2.0, 3.0, 597.0, 598.0, 599.0, 600.0]
    assert frames._covers_range(times, 0.0, 600.0) is False


def test_the_gate_is_a_multiple_of_the_implied_spacing():
    """Pins the shape of the rule, so a future tweak is a deliberate one."""
    # 4 candidates over 100s imply one every 20s; MAX_GAP_RATIO=4 allows 80s.
    assert frames.MAX_GAP_RATIO == 4.0
    just_inside = [0.0, 1.0, 2.0, 79.0]
    just_outside = [0.0, 1.0, 2.0, 99.0]
    assert frames._covers_range(just_inside, 0.0, 100.0) is True
    assert frames._covers_range(just_outside, 0.0, 100.0) is False


# --- integration: the engine still prefers scene on a normal clip ------------


def test_a_normally_cut_clip_still_uses_the_scene_engine(cut_clip: Path, tmp_path: Path):
    """The regression that would matter most: losing the scene engine everywhere."""
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=100,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert "fallback_reason" not in meta


def test_a_clip_with_a_cold_open_falls_back_on_coverage(
    clustered_cut_clip: Path, tmp_path: Path, capsys
):
    """The real shape, end to end: 9 cuts in the first ~3s, then 34s of nothing.

    Before the gate this reported engine "scene" with every frame inside the first
    three seconds, and the first/last-frame guarantee meant a span check passed.
    """
    out, meta = frames.extract_scene_or_uniform(
        str(clustered_cut_clip), tmp_path / "f",
        fps=1.0, target_frames=40, max_frames=40, dedup=False,
    )
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True
    assert meta["fallback_reason"] == "coverage"
    assert meta["fallback_candidate_count"] >= frames.SCENE_MIN_FRAMES  # it DID clear the floor

    # And the frames it produced instead actually reach the tail.
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts[-1] > 25.0, f"uniform frames only reached {ts[-1]}s"

    assert "no frame" in capsys.readouterr().err


def test_a_static_clip_falls_back_and_says_it_was_sparse(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=4.0, target_frames=12, max_frames=100,
    )
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True
    assert meta["fallback_reason"] == "sparse"


# --- the report's arithmetic has to hold -------------------------------------


def test_uniform_fallback_never_reports_more_selected_than_candidates(
    static_clip: Path, tmp_path: Path
):
    """"12 selected from 3 candidates" was printable, which teaches the reader to
    stop trusting that line."""
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=4.0, target_frames=12, max_frames=100,
    )
    assert meta["selected_count"] <= meta["candidate_count"]
    assert meta["candidate_count"] == meta["selected_count"] + meta["deduped_count"]
    # The rejected detector's count is still reported, just not as "candidates".
    assert "fallback_candidate_count" in meta


def test_keyframe_fallback_arithmetic_also_holds(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(static_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "uniform"
    assert meta["selected_count"] <= meta["candidate_count"]
    assert meta["candidate_count"] == meta["selected_count"] + meta["deduped_count"]
    assert meta["fallback_from"] == "keyframe"


def test_scene_engine_arithmetic_holds(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=5,
    )
    assert meta["engine"] == "scene"
    assert meta["selected_count"] == 5
    assert meta["selected_count"] <= meta["candidate_count"]
