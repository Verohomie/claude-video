"""yt-dlp health: staleness/impersonation preflight and download-failure diagnosis.

yt-dlp is perishable software — sites change their defences every few weeks and
yt-dlp answers within days — so a copy more than a month old, or one built
without curl_cffi, predicts a refused download. Both facts are one command away
from fixed, which is why they belong in preflight and in the failure message
rather than being discovered halfway through a download that reports only that
no video file appeared.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download  # noqa: E402
import setup  # noqa: E402


# ---- version age ----------------------------------------------------------

def test_age_of_a_known_old_release():
    old = (datetime.now() - timedelta(days=67)).strftime("%Y.%m.%d")
    assert setup._ytdlp_age_days(old) == 67


def test_age_of_todays_release_is_zero():
    assert setup._ytdlp_age_days(datetime.now().strftime("%Y.%m.%d")) == 0


def test_age_of_unparseable_version_is_none():
    assert setup._ytdlp_age_days("not-a-version") is None
    assert setup._ytdlp_age_days(None) is None


# ---- warnings -------------------------------------------------------------

def _probe(version_age_days=0, impersonation=True, path="/usr/local/bin/yt-dlp"):
    version = (datetime.now() - timedelta(days=version_age_days)).strftime("%Y.%m.%d")
    return {
        "path": path,
        "version": version,
        "impersonation": impersonation,
        "age_days": version_age_days,
    }


def test_healthy_ytdlp_warns_about_nothing():
    assert setup.ytdlp_warnings(_probe()) == []


def test_stale_ytdlp_warns():
    warnings = setup.ytdlp_warnings(_probe(version_age_days=setup.YTDLP_STALE_DAYS + 1))
    assert any("days old" in w for w in warnings)
    assert any(setup.YTDLP_UPGRADE_COMMAND in w for w in warnings)


def test_ytdlp_without_impersonation_warns():
    warnings = setup.ytdlp_warnings(_probe(impersonation=False))
    assert any("impersonation" in w for w in warnings)


def test_unknown_impersonation_does_not_warn():
    """An older build without the flag reports None. Absence of evidence is not
    evidence of a broken install, so it must not produce a warning."""
    assert setup.ytdlp_warnings(_probe(impersonation=None)) == []


def test_missing_ytdlp_is_the_binary_checks_job():
    """A missing binary is already a hard preflight failure; this probe must not
    also emit a warning about it."""
    assert setup.ytdlp_warnings({"path": None}) == []


# ---- failure diagnosis ----------------------------------------------------

def test_403_is_diagnosed():
    msg = download.diagnose_failure("ERROR: unable to download video data: HTTP Error 403: Forbidden")
    assert msg is not None and "403" in msg


def test_bot_check_is_diagnosed():
    msg = download.diagnose_failure("ERROR: Sign in to confirm you're not a bot")
    assert msg is not None and "bot check" in msg.lower()


def test_only_images_available_is_the_bot_check():
    """YouTube's gated response offers preview thumbnails and no video."""
    msg = download.diagnose_failure("ERROR: Requested format is not available. only images are available")
    assert msg is not None and "bot check" in msg.lower()


def test_private_video_is_diagnosed_without_a_local_fix():
    msg = download.diagnose_failure("ERROR: Private video. Sign in if you've been granted access")
    assert msg is not None
    assert "yt-dlp" not in msg  # nothing local to upgrade


def test_rate_limit_is_diagnosed():
    msg = download.diagnose_failure("ERROR: HTTP Error 429: Too Many Requests")
    assert msg is not None and "429" in msg


def test_unrecognized_output_is_not_guessed_at():
    assert download.diagnose_failure("some unrelated noise") is None
    assert download.diagnose_failure("") is None


@pytest.fixture
def pinned_probe(monkeypatch):
    """Pin the yt-dlp probe used by the failure message.

    Without this the message path calls the real probe, which shells out to the
    installed yt-dlp AND writes its cache into the developer's own
    ~/.config/watch — so the test would be both machine-dependent and dirty.
    """
    monkeypatch.setattr(setup, "probe_ytdlp", lambda **kw: _probe(
        version_age_days=67, impersonation=False, path="/opt/homebrew/bin/yt-dlp"
    ))


def test_state_line_reports_version_age_and_impersonation(pinned_probe):
    line = download._ytdlp_state_line()
    assert "67 days old" in line
    assert "no browser impersonation" in line
    assert "/opt/homebrew/bin/yt-dlp" in line


def test_failure_message_names_cause_and_fix(tmp_path, pinned_probe):
    msg = download._download_failure_message(
        "https://example.com/v", tmp_path, 1,
        "ERROR: unable to download video data: HTTP Error 403: Forbidden",
    )
    assert "403" in msg
    assert "pipx install" in msg
    assert "67 days old" in msg  # says what YOUR install is, not just the generic advice


def test_failure_message_without_a_known_cause_still_helps(tmp_path, pinned_probe):
    msg = download._download_failure_message("https://example.com/v", tmp_path, 1, "noise")
    assert str(tmp_path) in msg
    assert "pipx install" not in msg  # don't prescribe an unrelated fix


def test_state_line_is_silent_when_the_probe_cannot_answer(monkeypatch):
    monkeypatch.setattr(setup, "probe_ytdlp", lambda **kw: {"path": None, "version": None})
    assert download._ytdlp_state_line() is None
