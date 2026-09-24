"""setup.py --json surfaces the resolved watch detail."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"


def _seed_ytdlp_probe(home: Path, *, version=None, impersonation=True, latest=None) -> None:
    """Pin the cached yt-dlp probe, and the release check, under a temp HOME.

    Preflight warns about a yt-dlp that is behind or built without
    impersonation, so without this every assertion about setup's output would
    depend on which copies the developer happens to have installed. The probe
    cache is keyed on the candidate binaries — which depend on PATH and HOME —
    so the seed is written by setup.py itself under the same environment rather
    than guessed at here.

    ``signatures`` is not optional: the cache is validated against it, and a
    seed that omits it is silently discarded in favour of a real probe of the
    developer's machine. That is not a hypothetical — it happened when the
    signature check was added, and every test in this file quietly started
    measuring this Mac while still passing. test_a_seeded_probe_reaches_setup
    is the guard against it recurring.

    ``latest`` pins what the staleness check believes the newest release is,
    which keeps the suite off the network. These run setup.py in a subprocess,
    so conftest's ``offline`` fixture cannot reach them — it patches the
    in-process module only. Defaults to the seeded version, i.e. "nothing newer
    exists", which is the no-warning case.

    PATH is deliberately NOT scrubbed here or in _run. Emptying it would make
    the ffmpeg/ffprobe lookups report those as missing and change what these
    tests measure; pinning the cache is what makes the yt-dlp answer
    deterministic, and it does so whatever PATH contains.
    """
    version = version or datetime.now().strftime("%Y.%m.%d")
    params = json.dumps({
        "version": version,
        "impersonation": impersonation,
        "latest": latest or version,
    })
    snippet = (
        "import json, sys, time\n"
        f"sys.path.insert(0, {str(SETUP.parent)!r})\n"
        "import setup\n"
        f"p = json.loads({params!r})\n"
        "cands = setup.ytdlp_candidates()\n"
        "probes = {c: {'path': c, 'version': p['version'],\n"
        "              'impersonation': p['impersonation'],\n"
        "              'checked_at': time.time()} for c in cands}\n"
        "sigs = sorted(setup._candidate_signature(c) for c in cands)\n"
        "setup.CONFIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "setup.YTDLP_PROBE_CACHE.write_text(json.dumps(\n"
        "    {'checked_at': time.time(), 'signatures': sigs, 'probes': probes}),\n"
        "    encoding='utf-8')\n"
        "setup.YTDLP_LATEST_CACHE.write_text(json.dumps(\n"
        "    {'checked_at': time.time(), 'version': p['latest']}), encoding='utf-8')\n"
    )
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    subprocess.run([sys.executable, "-c", snippet], check=True, env=env, capture_output=True)


def _run(args, *, home=None, extra_env=None, seed_probe=True):
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    env.pop("WATCH_QUALITY", None)
    # Don't let a real key in the developer's shell env leak into the test.
    env.pop("GROQ_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("SETUP_COMPLETE", None)
    if home is not None:
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)  # Windows
        if seed_probe:
            _seed_ytdlp_probe(Path(home))
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SETUP), *args],
        capture_output=True, text=True, env=env,
    )


def _write_env(home: Path, body: str) -> None:
    cfg = home / ".config" / "watch"
    cfg.mkdir(parents=True, exist_ok=True)
    f = cfg / ".env"
    f.write_text(body, encoding="utf-8")
    f.chmod(0o600)


def test_a_seeded_probe_reaches_setup(tmp_path):
    """The seed must actually be used — silence here means it is being ignored.

    Every other test in this file assumes the seeded yt-dlp is what setup sees.
    When that stopped being true the suite did not fail; it kept passing while
    measuring the developer's own install, and the next assertion about warning
    output would have failed only on somebody else's machine. So one test
    seeds a deliberately broken yt-dlp and demands the complaint.
    """
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\nSETUP_COMPLETE=true\n")
    probed = json.loads(_run(["--json"], home=tmp_path, seed_probe=False).stdout)
    if probed["ytdlp"]["path"] is None:
        pytest.skip("no yt-dlp installed, so there is no candidate to describe")

    _seed_ytdlp_probe(
        tmp_path, version="2020.01.01", impersonation=False, latest="2099.12.31",
    )
    chk = _run(["--check"], home=tmp_path, seed_probe=False)
    assert "2020.01.01" in chk.stderr, (
        "seeded probe was discarded; setup re-probed the real machine instead"
    )
    assert "impersonation" in chk.stderr


def test_json_reports_watch_detail(tmp_path):
    # Always run under a temp HOME: setup.py both reads the config dir and
    # writes its yt-dlp probe cache there, so a real HOME means the suite
    # mutates the developer's own ~/.config/watch.
    proc = _run(["--json"], home=tmp_path)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["watch_detail"] == "balanced"
    assert data["watch_quality"] == "1080"


def test_keyless_completed_setup_proceeds_silently(tmp_path):
    """A user who finished setup without a key must NOT be nagged forever."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\nSETUP_COMPLETE=true\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, f"keyless-complete should pass --check; got {chk.returncode}: {chk.stderr}"
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is True
    assert js["first_run"] is False
    assert js["setup_complete"] is True
    # status still encourages a key even though we can proceed
    assert js["status"] == "needs_key"


def test_keyless_first_run_is_encouraged(tmp_path):
    """Genuine first run with no key: --check reports exit 3 (encourage a key)."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 3, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is False
    assert js["first_run"] is True


def test_key_present_is_ready(tmp_path):
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["whisper_backend"] == "groq"
