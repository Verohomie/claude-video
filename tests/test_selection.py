"""Thinning candidates down to a cap: spacing by time, not by list position."""
from __future__ import annotations

from pathlib import Path

import frames


# --- _even_time_indices ------------------------------------------------------


def test_evenly_spread_candidates_are_unchanged_by_the_switch():
    """When candidates are already even in time, time- and index-spacing agree."""
    times = [float(i) for i in range(11)]  # 0..10, one per second
    assert frames._even_time_indices(times, 6) == frames._even_indices(len(times), 6)


def test_a_cluster_does_not_keep_its_share_of_the_budget():
    """9 candidates in the first 2s, 3 spread over the next 58s.

    Index spacing keeps mostly the cluster; time spacing has to reach into the
    sparse tail, because that is where most of the runtime is.
    """
    times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 20.0, 40.0, 60.0]

    by_index = frames._even_indices(len(times), 4)
    by_time = frames._even_time_indices(times, 4)

    picked_by_index = [times[i] for i in by_index]
    picked_by_time = [times[i] for i in by_time]

    # The old behaviour spends 2 of 4 slots inside the first 1.6 seconds.
    assert sum(1 for t in picked_by_index if t <= 1.6) >= 2
    # Time spacing keeps only the pinned first frame down there.
    assert sum(1 for t in picked_by_time if t <= 1.6) == 1
    assert picked_by_time == [0.0, 20.0, 40.0, 60.0]


def test_first_and_last_are_always_kept():
    times = [0.0, 1.0, 2.0, 50.0, 99.0]
    for n in (2, 3, 4):
        idx = frames._even_time_indices(times, n)
        assert idx[0] == 0
        assert idx[-1] == len(times) - 1
        assert len(idx) == n


def test_no_candidate_is_selected_twice():
    """Two targets landing nearest the same candidate must not collapse the count."""
    times = [0.0, 10.0, 10.1, 10.2, 100.0]
    idx = frames._even_time_indices(times, 4)
    assert len(idx) == len(set(idx)) == 4


def test_indices_come_back_sorted():
    times = [0.0, 5.0, 30.0, 31.0, 90.0, 120.0]
    idx = frames._even_time_indices(times, 4)
    assert idx == sorted(idx)


def test_degenerate_cases():
    assert frames._even_time_indices([4.0, 4.0, 4.0], 2) == frames._even_indices(3, 2)
    assert frames._even_time_indices([1.0, 2.0], 5) == [0, 1]   # n >= count
    assert frames._even_time_indices([1.0, 2.0, 3.0], 1) == [0]
    assert frames._even_time_indices([], 3) == []


# --- _even_sample: the integration point every capped engine shares ----------


def _candidates(dirpath: Path, times: list[float]) -> list[dict]:
    dirpath.mkdir(parents=True, exist_ok=True)
    out = []
    for i, t in enumerate(times):
        p = dirpath / f"frame_{i:04d}.jpg"
        p.write_bytes(b"x")
        out.append({"index": i, "timestamp_seconds": t, "path": str(p), "reason": "scene-change"})
    return out


def test_even_sample_spaces_a_clustered_list_by_time(tmp_path: Path):
    times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 20.0, 40.0, 60.0]
    cands = _candidates(tmp_path / "f", times)
    selected = frames._even_sample(cands, 4)
    assert [c["timestamp_seconds"] for c in selected] == [0.0, 20.0, 40.0, 60.0]
    # Dropped JPEGs are deleted and survivors reindexed, as before.
    assert len(list((tmp_path / "f").glob("frame_*.jpg"))) == 4
    assert [c["index"] for c in selected] == [0, 1, 2, 3]


def test_even_sample_falls_back_when_timestamps_are_not_monotonic(tmp_path: Path):
    """Nothing in the engines produces this, but a nearest-target search over a
    scrambled timeline would be meaningless, so it must degrade rather than lie."""
    cands = _candidates(tmp_path / "f", [0.0, 50.0, 10.0, 60.0, 20.0])
    selected = frames._even_sample(cands, 3)
    assert len(selected) == 3
    assert [c["index"] for c in selected] == [0, 1, 2]


def test_even_sample_under_the_cap_keeps_everything(tmp_path: Path):
    cands = _candidates(tmp_path / "f", [0.0, 10.0, 20.0])
    assert len(frames._even_sample(cands, 10)) == 3
    assert len(list((tmp_path / "f").glob("frame_*.jpg"))) == 3


def test_scene_engine_capped_run_spreads_across_the_clip(cut_clip: Path, tmp_path: Path):
    """End to end: 14 real cuts thinned to 5 must still reach the last shot."""
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=5,
    )
    assert meta["engine"] == "scene"
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert len(ts) == 5
    assert ts[-1] > 4.0
