"""Choosing WHICH yt-dlp to run.

The skill's installer recommends `pipx install yt-dlp`, which lands in
~/.local/bin. On a stock Mac, Homebrew's bin directory precedes ~/.local/bin on
PATH, so a bare `yt-dlp` resolves to Homebrew's build — which lags weeks behind
and is compiled without curl_cffi, leaving every browser-impersonation target
"unavailable". YouTube then refuses the video stream while metadata and captions
keep working, so the run fails minutes in with a confusing error.

That made the skill defeat its own advice: follow the setup instructions, and
the download path still picks the copy that cannot download. So the binary is
chosen by capability, not by PATH order.

These tests build real stub executables rather than mocking the probe, so the
version parsing and the "(unavailable)" detection are exercised too.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import setup  # noqa: E402


def _fake_ytdlp(directory: Path, version: str, *, impersonation: bool) -> Path:
    """A stub that answers --version and --list-impersonate-targets like the real
    thing — including the trap that a build with no curl_cffi still LISTS every
    target and exits 0, marking each one "(unavailable)"."""
    directory.mkdir(parents=True, exist_ok=True)
    if impersonation:
        targets = "Chrome-133      Macos-15     curl_cffi"
    else:
        targets = "Chrome          -            curl_cffi (unavailable)"
    path = directory / "yt-dlp"
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        f'  echo "{version}"\n'
        'elif [ "$1" = "--list-impersonate-targets" ]; then\n'
        '  echo "[info] Available impersonate targets"\n'
        '  echo "Client          OS           Source"\n'
        '  echo "--------------------------------------"\n'
        f'  echo "{targets}"\n'
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Point the probe cache at a temp dir so tests never touch ~/.config/watch."""
    cfg = tmp_path / "config"
    monkeypatch.setattr(setup, "CONFIG_DIR", cfg)
    monkeypatch.setattr(setup, "YTDLP_PROBE_CACHE", cfg / ".ytdlp-probe.json")
    monkeypatch.setattr(setup, "ytdlp_extra_dirs", lambda: [])
    monkeypatch.setenv("PATH", "")
    return tmp_path


def test_finds_a_copy_that_is_not_on_path(isolated, monkeypatch):
    """The pipx case: installed where the installer puts it, invisible to PATH."""
    local = isolated / "local"
    _fake_ytdlp(local, "2026.08.19", impersonation=True)
    monkeypatch.setattr(setup, "ytdlp_extra_dirs", lambda: [local])
    assert setup.ytdlp_candidates() == [str(local / "yt-dlp")]


def test_impersonation_beats_path_order(isolated, monkeypatch):
    """The actual defect. Homebrew's build is first on PATH and newer here, and
    must still lose to the one that can impersonate a browser."""
    brew = isolated / "brew"
    pipx = isolated / "pipx"
    _fake_ytdlp(brew, "2026.09.01", impersonation=False)
    _fake_ytdlp(pipx, "2026.08.19", impersonation=True)
    monkeypatch.setenv("PATH", str(brew))
    monkeypatch.setattr(setup, "ytdlp_extra_dirs", lambda: [pipx])

    probe = setup.probe_ytdlp(force=True)
    assert probe["path"] == str(pipx / "yt-dlp")
    assert probe["impersonation"] is True
    assert probe["shadowed"] == [str(brew / "yt-dlp")]
    assert setup.resolve_ytdlp() == str(pipx / "yt-dlp")


def test_newest_wins_when_both_can_impersonate(isolated, monkeypatch):
    old = isolated / "old"
    new = isolated / "new"
    _fake_ytdlp(old, "2026.01.02", impersonation=True)
    _fake_ytdlp(new, "2026.08.19", impersonation=True)
    monkeypatch.setenv("PATH", str(old))
    monkeypatch.setattr(setup, "ytdlp_extra_dirs", lambda: [new])
    assert setup.probe_ytdlp(force=True)["path"] == str(new / "yt-dlp")


def test_only_a_bad_copy_still_resolves_and_warns(isolated, monkeypatch):
    """With nothing better installed, run what there is — but say so. Refusing
    would be wrong: plenty of sources download fine from an old build."""
    brew = isolated / "brew"
    _fake_ytdlp(brew, "2020.01.01", impersonation=False)
    monkeypatch.setenv("PATH", str(brew))

    probe = setup.probe_ytdlp(force=True)
    assert probe["path"] == str(brew / "yt-dlp")
    warnings = setup.ytdlp_warnings(probe)
    assert any("days old" in w for w in warnings)
    assert any("impersonation" in w for w in warnings)


def test_no_copy_anywhere(isolated):
    probe = setup.probe_ytdlp(force=True)
    assert probe["path"] is None
    assert setup.ytdlp_warnings(probe) == []
    assert setup.resolve_ytdlp() == "yt-dlp"  # let the caller's own error fire


def test_check_binaries_accepts_an_off_path_copy(isolated, monkeypatch):
    """A pipx-only install must not be reported as "yt-dlp missing" — that would
    send the user to install a second copy of what is already there."""
    local = isolated / "local"
    _fake_ytdlp(local, "2026.08.19", impersonation=True)
    monkeypatch.setattr(setup, "ytdlp_extra_dirs", lambda: [local])
    monkeypatch.setattr(setup, "_which", lambda name: "/usr/bin/" + name)
    assert "yt-dlp" not in setup._check_binaries()


def test_warning_names_the_chosen_copy_when_others_exist(isolated, monkeypatch):
    """With several installed and all of them bad, "upgrade yt-dlp" is ambiguous
    unless the message says which one actually ran."""
    a, b = isolated / "a", isolated / "b"
    _fake_ytdlp(a, "2020.01.01", impersonation=False)
    _fake_ytdlp(b, "2020.02.02", impersonation=False)
    monkeypatch.setenv("PATH", os.pathsep.join([str(a), str(b)]))

    probe = setup.probe_ytdlp(force=True)
    assert probe["shadowed"]
    assert any(probe["path"] in w for w in setup.ytdlp_warnings(probe))


def test_version_key_orders_calver_numerically():
    """String ordering would put 2026.9.1 before 2026.08.19; it is newer."""
    assert setup._version_key("2026.9.1") > setup._version_key("2026.08.19")
    assert setup._version_key(None) < setup._version_key("2020.01.01")
    assert setup._version_key("garbage") < setup._version_key("2020.01.01")


def test_probe_cache_reprobes_when_the_candidate_set_changes(isolated, monkeypatch):
    """Installing a better copy must take effect immediately, not after the TTL."""
    brew = isolated / "brew"
    _fake_ytdlp(brew, "2020.01.01", impersonation=False)
    monkeypatch.setenv("PATH", str(brew))
    assert setup.probe_ytdlp()["path"] == str(brew / "yt-dlp")

    pipx = isolated / "pipx"
    _fake_ytdlp(pipx, "2026.08.19", impersonation=True)
    monkeypatch.setattr(setup, "ytdlp_extra_dirs", lambda: [pipx])
    assert setup.probe_ytdlp()["path"] == str(pipx / "yt-dlp")


# ---- the resolved path must actually reach yt-dlp's argv -------------------

def test_download_invokes_the_resolved_binary(isolated, monkeypatch, tmp_path):
    """Resolving correctly is worthless if the argv still says bare "yt-dlp" —
    that was the defect. Both entry points must use the absolute path."""
    import io

    import download

    pipx = isolated / "pipx"
    _fake_ytdlp(pipx, "2026.08.19", impersonation=True)
    chosen = str(pipx / "yt-dlp")
    monkeypatch.setattr(download, "resolve_ytdlp", lambda: chosen)

    calls: list[list[str]] = []

    class _Popen:
        def __init__(self, cmd, *a, **kw):
            calls.append(list(cmd))
            self.stdout = io.StringIO("")

        def wait(self):
            return 0

    def _run(cmd, *a, **kw):
        calls.append(list(cmd))

        class R:
            returncode = 0
            stdout = stderr = ""
        return R()

    monkeypatch.setattr(download.subprocess, "Popen", _Popen)
    monkeypatch.setattr(download.subprocess, "run", _run)

    download.fetch_captions("https://example.com/v", tmp_path / "a")
    with pytest.raises(SystemExit):  # no real file lands
        download.download_url("https://example.com/v", tmp_path / "b")

    assert len(calls) == 2
    for argv in calls:
        assert argv[0] == chosen, f"argv still uses {argv[0]!r} instead of the resolved binary"
