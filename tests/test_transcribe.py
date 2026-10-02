"""WebVTT parsing: timestamp shapes, entity handling, rolling-cue dedupe, rendering."""
from __future__ import annotations

from pathlib import Path

import transcribe


def _vtt(tmp_path: Path, body: str, name: str = "subs.vtt") -> str:
    path = tmp_path / name
    path.write_text("WEBVTT\n\n" + body, encoding="utf-8")
    return str(path)


# --- timestamp shapes --------------------------------------------------------


def test_parses_hhmmss(tmp_path: Path):
    segs = transcribe.parse_vtt(_vtt(tmp_path, "00:00:01.500 --> 00:00:03.000\nhello\n"))
    assert segs == [{"start": 1.5, "end": 3.0, "text": "hello"}]


def test_parses_mmss_without_an_hours_field(tmp_path: Path):
    """WebVTT makes hours optional; requiring them parsed the file to zero cues."""
    segs = transcribe.parse_vtt(_vtt(tmp_path, "01:02.500 --> 01:04.000\nhello\n"))
    assert segs == [{"start": 62.5, "end": 64.0, "text": "hello"}]


def test_parses_more_than_two_hour_digits(tmp_path: Path):
    segs = transcribe.parse_vtt(_vtt(tmp_path, "100:00:00.000 --> 100:00:02.000\nlate\n"))
    assert segs[0]["start"] == 360000.0


def test_parses_srt_style_comma_decimals(tmp_path: Path):
    segs = transcribe.parse_vtt(_vtt(tmp_path, "00:00:01,250 --> 00:00:02,000\nhi\n"))
    assert segs[0]["start"] == 1.25


# --- cue text ---------------------------------------------------------------


def test_unescapes_html_entities(tmp_path: Path):
    """">>" speaker markers arrive as "&gt;&gt;" on real YouTube tracks."""
    segs = transcribe.parse_vtt(
        _vtt(tmp_path, "00:00:01.000 --> 00:00:02.000\n&gt;&gt; Bell &amp; Co. isn&#39;t here\n")
    )
    assert segs[0]["text"] == ">> Bell & Co. isn't here"


def test_strips_tags_before_unescaping(tmp_path: Path):
    """Order matters: an escaped tag must survive as literal text, not be eaten.

    Unescaping first would turn "&lt;b&gt;" into "<b>", which the tag stripper
    would then delete -- losing text the caption author actually wrote.
    """
    segs = transcribe.parse_vtt(
        _vtt(tmp_path, "00:00:01.000 --> 00:00:02.000\n<c.colorE5E5E5>use &lt;b&gt; here</c>\n")
    )
    assert segs[0]["text"] == "use <b> here"


def test_skips_cues_that_are_only_markup(tmp_path: Path):
    segs = transcribe.parse_vtt(
        _vtt(tmp_path, "00:00:01.000 --> 00:00:02.000\n<c></c>\n\n"
                       "00:00:03.000 --> 00:00:04.000\nreal\n")
    )
    assert [s["text"] for s in segs] == ["real"]


# --- rolling-duplicate dedupe ------------------------------------------------


def test_collapses_identical_consecutive_cues(tmp_path: Path):
    segs = transcribe.parse_vtt(
        _vtt(tmp_path, "00:00:01.000 --> 00:00:02.000\nsame\n\n"
                       "00:00:02.000 --> 00:00:05.000\nsame\n")
    )
    assert len(segs) == 1
    assert segs[0]["end"] == 5.0  # time range merged


def test_collapses_a_growing_rolling_cue(tmp_path: Path):
    segs = transcribe.parse_vtt(
        _vtt(tmp_path, "00:00:01.000 --> 00:00:02.000\nthe quick\n\n"
                       "00:00:02.000 --> 00:00:03.000\nthe quick brown fox\n")
    )
    assert [s["text"] for s in segs] == ["the quick brown fox"]
    assert segs[0]["start"] == 1.0 and segs[0]["end"] == 3.0


def test_keeps_genuinely_different_cues(tmp_path: Path):
    segs = transcribe.parse_vtt(
        _vtt(tmp_path, "00:00:01.000 --> 00:00:02.000\nfirst\n\n"
                       "00:00:03.000 --> 00:00:04.000\nsecond\n")
    )
    assert [s["text"] for s in segs] == ["first", "second"]


# --- rendering --------------------------------------------------------------


def test_format_transcript_uses_mmss_under_an_hour():
    assert transcribe.format_transcript([{"start": 65.0, "end": 66.0, "text": "x"}]) == "[01:05] x"


def test_format_transcript_rolls_over_to_hours():
    """1h10m05s was rendering as [70:05], which matches no frame timestamp."""
    out = transcribe.format_transcript([{"start": 4205.0, "end": 4206.0, "text": "x"}])
    assert out == "[1:10:05] x"


def test_format_transcript_matches_frames_format_time():
    """The two halves of the report must agree on how a time is written."""
    import frames

    for seconds in (0, 59, 60, 3599, 3600, 4205, 7322):
        seg = [{"start": float(seconds), "end": float(seconds), "text": "x"}]
        rendered = transcribe.format_transcript(seg).split(" ")[0].strip("[]")
        assert rendered == frames.format_time(float(seconds))


# --- range filtering --------------------------------------------------------


def test_filter_range_keeps_overlapping_segments():
    segs = [
        {"start": 0.0, "end": 5.0, "text": "a"},
        {"start": 10.0, "end": 15.0, "text": "b"},
        {"start": 20.0, "end": 25.0, "text": "c"},
    ]
    kept = transcribe.filter_range(segs, 12.0, 22.0)
    assert [s["text"] for s in kept] == ["b", "c"]


def test_filter_range_without_bounds_is_a_noop():
    segs = [{"start": 0.0, "end": 5.0, "text": "a"}]
    assert transcribe.filter_range(segs, None, None) is segs


# --- YouTube's scrolling automatic captions -----------------------------------

# Verbatim shape of a YouTube auto-caption file: each cue opens with a line
# holding one space, carries the previous line over untimed, and adds the new
# words with per-word timing tags; ~10ms cues in between repeat the carried line.
ROLLING = (
    "WEBVTT\nKind: captions\nLanguage: en\n\n"
    "00:00:00.000 --> 00:00:02.630 align:start position:0%\n \n"
    "AI<00:00:00.400><c> can</c><00:00:00.560><c> now</c><00:00:00.760><c> watch</c>\n\n"
    "00:00:02.630 --> 00:00:02.640 align:start position:0%\n"
    "AI can now watch\n \n\n"
    "00:00:02.640 --> 00:00:04.390 align:start position:0%\n"
    "AI can now watch\n"
    "thank<00:00:02.960><c> you</c><00:00:03.120><c> thank</c><00:00:03.280><c> you</c>\n\n"
    "00:00:04.390 --> 00:00:04.400 align:start position:0%\n"
    "thank you thank you\n \n\n"
    "00:00:04.400 --> 00:00:06.230 align:start position:0%\n"
    "thank you thank you\n"
    "for<00:00:04.640><c> watching</c>\n\n"
)


def _raw_vtt(tmp_path, text: str) -> str:
    path = tmp_path / "subs.vtt"
    path.write_bytes(text.encode("utf-8"))
    return str(path)


def test_rolling_captions_print_each_word_once(tmp_path):
    segs = transcribe.parse_vtt(_raw_vtt(tmp_path, ROLLING))
    words = " ".join(s["text"] for s in segs).split()
    assert words == "AI can now watch thank you thank you for watching".split()


def test_rolling_captions_keep_the_first_line(tmp_path):
    """The opening cue starts with a one-space line; it must not end the cue."""
    segs = transcribe.parse_vtt(_raw_vtt(tmp_path, ROLLING))
    assert segs[0]["start"] == 0.0
    assert segs[0]["text"] == "AI can now watch"


def test_real_repetition_in_speech_survives(tmp_path):
    """'thank you thank you' was said twice; only the carried copy goes."""
    text = " ".join(s["text"] for s in transcribe.parse_vtt(_raw_vtt(tmp_path, ROLLING)))
    assert text.count("thank you") == 2


def test_human_captions_without_word_timings_are_untouched(tmp_path):
    manual = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\nFirst line of a caption\nsecond line of it\n\n"
        "00:00:03.000 --> 00:00:05.000\nsecond line of it\nthen something new\n\n"
    )
    segs = transcribe.parse_vtt(_raw_vtt(tmp_path, manual))
    assert [s["text"] for s in segs] == [
        "First line of a caption second line of it",
        "second line of it then something new",
    ]


def test_a_timestamp_line_ends_a_cue_even_without_a_blank_line(tmp_path):
    squashed = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\none\n   \n"
        "00:00:02.000 --> 00:00:03.000\ntwo\n"
    )
    segs = transcribe.parse_vtt(_raw_vtt(tmp_path, squashed))
    assert [s["text"] for s in segs] == ["one", "two"]
