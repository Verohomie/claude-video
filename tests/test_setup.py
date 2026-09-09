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

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"


def _seed_ytdlp_probe(home: Path, *, version=None, impersonation=True) -> None:
    """Pin the cached yt-dlp probe under a temp HOME.

    Preflight warns about a stale or impersonation-less yt-dlp, so without this
    every assertion about setup's output would depend on which yt-dlp copies the
    developer happens to have installed. The cache is keyed on the exact
    candidate set, which depends on PATH and on HOME, so the seed is written by
    setup.py itself under the same environment rather than guessed at here.
    """
    version = version or datetime.now().strftime("%Y.%m.%d")
    snippet = (
        "import json, sys, time\n"
        f"sys.path.insert(0, {str(SETUP.parent)!r})\n"
        "import setup\n"
        "cands = setup.ytdlp_candidates()\n"
        "probes = {p: {'path': p, 'version': %r, 'impersonation': %r,\n"
        "              'checked_at': time.time()} for p in cands}\n"
        "setup.CONFIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "setup.YTDLP_PROBE_CACHE.write_text(\n"
        "    json.dumps({'checked_at': time.time(), 'probes': probes}), encoding='utf-8')\n"
    ) % (version, impersonation)
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
