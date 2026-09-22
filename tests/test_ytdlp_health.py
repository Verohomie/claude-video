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

# Captured before conftest's autouse `offline` fixture replaces it, so the real
# implementation stays reachable for the tests that exercise it directly.
REAL_LIVENESS_PROBE = download.probe_video_liveness


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


def test_old_but_current_ytdlp_does_not_warn(monkeypatch):
    """Age alone is not a fault. On 2026-09-22 the newest yt-dlp in existence was
    34 days old, so warning on age sent users to reinstall the identical build."""
    probe = _probe(version_age_days=setup.YTDLP_STALE_DAYS + 4)
    monkeypatch.setattr(setup, "latest_ytdlp_release", lambda **kw: probe["version"])
    assert setup.ytdlp_warnings(probe) == []


def test_old_ytdlp_warns_when_a_newer_release_exists(monkeypatch):
    monkeypatch.setattr(setup, "latest_ytdlp_release", lambda **kw: "2099.12.31")
    warnings = setup.ytdlp_warnings(_probe(version_age_days=setup.YTDLP_STALE_DAYS + 4))
    assert any("2099.12.31 is available" in w for w in warnings)
    assert any(setup.YTDLP_UPGRADE_COMMAND in w for w in warnings)


def test_unreachable_pypi_admits_it_rather_than_asserting_staleness(monkeypatch):
    """Offline, the honest answer is 'unknown', not 'you are behind'."""
    monkeypatch.setattr(setup, "latest_ytdlp_release", lambda **kw: None)
    warnings = setup.ytdlp_warnings(_probe(version_age_days=setup.YTDLP_STALE_DAYS + 4))
    assert any("may already be current" in w for w in warnings)


def test_zero_padded_calver_compares_equal_to_pypis_unpadded_form(monkeypatch):
    """PyPI normalizes 2026.08.19 to 2026.8.19; a string compare would call an
    up-to-date install stale."""
    probe = _probe(version_age_days=setup.YTDLP_STALE_DAYS + 4)
    probe["version"] = "2026.08.19"
    monkeypatch.setattr(setup, "latest_ytdlp_release", lambda **kw: "2026.8.19")
    assert setup.ytdlp_warnings(probe) == []


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
    # Patch the name in download, not in setup: download.py does
    # `from setup import probe_ytdlp`, binding its own reference at import,
    # so patching setup.probe_ytdlp would leave the caller on the real one.
    monkeypatch.setattr(download, "probe_ytdlp", lambda **kw: _probe(
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
    monkeypatch.setattr(download, "probe_ytdlp", lambda **kw: {"path": None, "version": None})
    assert download._ytdlp_state_line() is None


# ---- refusals: measured cause, not assumed cause --------------------------

YT_URL = "https://youtu.be/tTnUcSj-QPA"


@pytest.fixture
def healthy_probe(monkeypatch):
    """A yt-dlp with nothing wrong with it — the case the old advice got wrong."""
    monkeypatch.setattr(download, "probe_ytdlp", lambda **kw: _probe(
        version_age_days=2, impersonation=True, path="/Users/x/.local/bin/yt-dlp"
    ))


def test_refusal_with_a_healthy_ytdlp_says_wait_not_reinstall(tmp_path, healthy_probe):
    msg = download._download_failure_message(
        YT_URL, tmp_path, 1, "ERROR: Sign in to confirm you're not a bot",
    )
    assert "NOT the out-of-date-tool case" in msg
    assert "Do not retry in a loop" in msg
    assert "pipx install" not in msg  # the fix that does nothing here


def test_refusal_with_a_faulty_ytdlp_still_prescribes_the_upgrade(tmp_path, pinned_probe):
    msg = download._download_failure_message(
        YT_URL, tmp_path, 1, "ERROR: Sign in to confirm you're not a bot",
    )
    assert "likely cause" in msg
    assert "pipx install" in msg
    assert "Do not retry in a loop" not in msg


def test_refusal_reports_that_the_video_itself_is_still_there(
    tmp_path, healthy_probe, monkeypatch
):
    monkeypatch.setattr(
        download, "probe_video_liveness",
        lambda url: ("live", '"Some Title" by Some Channel'),
    )
    msg = download._download_failure_message(
        YT_URL, tmp_path, 1, "ERROR: Sign in to confirm you're not a bot",
    )
    assert "Some Title" in msg
    assert "not a dead or private video" in msg


def test_a_dead_video_stops_at_that_and_gives_no_waiting_advice(
    tmp_path, healthy_probe, monkeypatch
):
    """Telling someone to wait for a video that no longer exists is worse than
    saying nothing."""
    monkeypatch.setattr(
        download, "probe_video_liveness",
        lambda url: ("gone", "oEmbed returned HTTP 404"),
    )
    msg = download._download_failure_message(
        YT_URL, tmp_path, 1, "ERROR: Sign in to confirm you're not a bot",
    )
    assert "private, removed, or the ID is wrong" in msg
    assert "Do not retry in a loop" not in msg
    assert "pipx install" not in msg


def test_a_broken_diagnosis_never_swallows_the_real_error(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "probe_ytdlp", lambda **kw: 1 / 0)
    msg = download._download_failure_message(
        YT_URL, tmp_path, 1, "ERROR: Sign in to confirm you're not a bot",
    )
    assert "bot check" in msg.lower()


def test_liveness_probe_only_applies_to_youtube():
    assert download._is_youtube("https://youtu.be/abc")
    assert download._is_youtube("https://www.youtube.com/watch?v=abc")
    assert not download._is_youtube("https://vimeo.com/123")
    assert not download._is_youtube("https://notyoutube.com/watch?v=abc")


@pytest.mark.parametrize(
    "code,expected",
    [(400, "gone"), (401, "gone"), (403, "gone"), (404, "gone"),
     (500, "unknown"), (429, "unknown")],
)
def test_liveness_probe_reads_oembed_status_codes(monkeypatch, code, expected):
    """400 is YouTube's answer for an ID it cannot resolve; 5xx/429 are the
    endpoint's own trouble and say nothing about the video."""
    def raise_http(*_a, **_kw):
        raise download.HTTPError(YT_URL, code, "nope", {}, None)

    monkeypatch.setattr(download, "urlopen", raise_http)
    state, _detail = REAL_LIVENESS_PROBE(YT_URL)
    assert state == expected


def test_liveness_probe_of_a_non_youtube_url_makes_no_request(monkeypatch):
    """Non-YouTube sources must not reach the endpoint at all."""
    monkeypatch.setattr(download, "urlopen", lambda *a, **kw: pytest.fail("no request"))
    assert REAL_LIVENESS_PROBE("https://vimeo.com/123") == ("unknown", None)
