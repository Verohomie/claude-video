"""Whisper invents fluent dialogue from music or silence; say so when it looks like it.

The thresholds are Whisper's own signals, not guesses about content. The risk
runs both ways: a false flag on a real transcript teaches the reader to ignore
the warning, so the accept direction is asserted as carefully as the reject one.
"""
from __future__ import annotations

import whisper


def _seg(text: str, start: float = 0.0, **fields) -> dict:
    return {"start": start, "end": start + 1.0, "text": text, **fields}


# --- nothing to flag ---------------------------------------------------------


def test_empty_transcript_is_not_suspect():
    assert whisper.assess_speech([])["suspect"] is False


def test_a_normal_transcript_is_not_flagged():
    segs = [
        _seg("welcome back to the channel", 0.0, no_speech_prob=0.01, avg_logprob=-0.2),
        _seg("today we are looking at the new release", 3.0, no_speech_prob=0.02, avg_logprob=-0.3),
        _seg("the first thing to notice is the layout", 7.0, no_speech_prob=0.05, avg_logprob=-0.25),
    ]
    assert whisper.assess_speech(segs)["suspect"] is False


def test_segments_without_confidence_fields_are_not_accused():
    """Caption transcripts and local backends omit these; absence must not vote."""
    segs = [_seg("one"), _seg("two", 2.0), _seg("three", 4.0)]
    assert whisper.assess_speech(segs)["suspect"] is False


def test_an_occasional_repeated_line_is_fine():
    """Real speech repeats. Only a phrase dominating the transcript is a signal."""
    segs = [
        _seg("right", 0.0, no_speech_prob=0.01),
        _seg("so the next step is the config file", 1.0, no_speech_prob=0.01),
        _seg("right", 5.0, no_speech_prob=0.02),
        _seg("and that writes it to disk", 6.0, no_speech_prob=0.01),
        _seg("now we run the tests", 9.0, no_speech_prob=0.01),
        _seg("right", 12.0, no_speech_prob=0.02),
    ]
    assert whisper.assess_speech(segs)["suspect"] is False


# --- the signals -------------------------------------------------------------


def test_mostly_non_speech_is_flagged():
    """A music bed: the model itself says most of this is not speech."""
    segs = [
        _seg("thanks for watching", 0.0, no_speech_prob=0.95),
        _seg("please subscribe", 3.0, no_speech_prob=0.91),
        _seg("see you next time", 6.0, no_speech_prob=0.88),
        _seg("hello there", 9.0, no_speech_prob=0.20),
    ]
    result = whisper.assess_speech(segs)
    assert result["suspect"] is True
    assert any("not-speech" in r for r in result["reasons"])


def test_a_looping_phrase_is_flagged():
    """The runaway signature on non-speech audio."""
    segs = [_seg("thank you", float(i) * 3) for i in range(6)] + [_seg("okay", 18.0)]
    result = whisper.assess_speech(segs)
    assert result["suspect"] is True
    assert any("transcript" in r and "%" in r for r in result["reasons"])


def test_low_token_confidence_is_flagged():
    segs = [
        _seg("indistinct muttering here", 0.0, avg_logprob=-1.6),
        _seg("more of the same", 3.0, avg_logprob=-1.4),
    ]
    result = whisper.assess_speech(segs)
    assert result["suspect"] is True
    assert any("confidence" in r for r in result["reasons"])


def test_reasons_accumulate():
    segs = [_seg("thank you", float(i) * 3, no_speech_prob=0.9, avg_logprob=-1.5) for i in range(5)]
    result = whisper.assess_speech(segs)
    assert result["suspect"] is True
    assert len(result["reasons"]) >= 2


def test_punctuation_does_not_defeat_the_repetition_check():
    segs = [_seg("Thank you.", 0.0), _seg("thank you!", 3.0), _seg("THANK YOU", 6.0),
            _seg("thank you,", 9.0)]
    result = whisper.assess_speech(segs)
    assert result["suspect"] is True


# --- the fields survive parsing ---------------------------------------------


def test_confidence_fields_are_kept_from_the_api_response():
    data = {"segments": [
        {"start": 0, "end": 2, "text": " hello", "no_speech_prob": 0.8, "avg_logprob": -1.2},
    ]}
    segs = whisper._segments_from_response(data)
    assert segs[0]["no_speech_prob"] == 0.8
    assert segs[0]["avg_logprob"] == -1.2
    assert segs[0]["text"] == "hello"


def test_missing_confidence_fields_are_simply_absent():
    data = {"segments": [{"start": 0, "end": 2, "text": " hi"}]}
    segs = whisper._segments_from_response(data)
    assert "no_speech_prob" not in segs[0]
    assert segs[0]["text"] == "hi"
