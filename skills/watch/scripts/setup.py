#!/usr/bin/env python3
"""Setup / preflight for /watch.

Modes:
  setup.py --check      Silent preflight. Exit 0 if ready, 2/3/4 on failure.
  setup.py --json       Machine-readable status for Claude to parse.
  setup.py              Installer. Auto-installs deps, scaffolds .env, marks SETUP_COMPLETE.

Design:
- Silent on success: --check exits 0 with no output when everything's ready so
  that /watch doesn't spam "setup is complete" on every turn.
- Idempotent: re-running the installer is safe — it never clobbers existing
  keys and only appends missing ones.
- SETUP_COMPLETE=true in ~/.config/watch/.env tells us the user has been
  through a successful installer run at least once.
- Never sudo. On macOS, auto-install via brew. Elsewhere, print exact commands.
- Never write an API key to disk automatically — only scaffold placeholders.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from config import get_config  # noqa: E402


REQUIRED_BINARIES = ["ffmpeg", "ffprobe", "yt-dlp"]
CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"

# yt-dlp is perishable: sites change their defences every few weeks and yt-dlp
# answers within days, so a copy older than this is a likely download failure
# rather than merely a stale one.
YTDLP_STALE_DAYS = 30
# Probing costs two yt-dlp launches (~0.5s), and --check runs on every /watch
# call, so the answer is cached. Keyed on the resolved binary path so swapping
# installs re-probes immediately instead of reporting the old one for a day.
YTDLP_PROBE_CACHE = CONFIG_DIR / ".ytdlp-probe.json"
YTDLP_PROBE_TTL_SECONDS = 24 * 60 * 60
YTDLP_UPGRADE_COMMAND = "pipx install --force 'yt-dlp[default,curl-cffi]'"
ENV_TEMPLATE = """# /watch API configuration
#
# Whisper transcription fallback — used only when yt-dlp cannot get captions
# (or when you point /watch at a local file with no subtitles).
#
# Groq is preferred: it runs whisper-large-v3 at a fraction of OpenAI's price
# and is faster in practice. OpenAI is the compatible fallback.
#
# Get a Groq key:  https://console.groq.com/keys
# Get an OpenAI key:  https://platform.openai.com/api-keys
#
# Leave both blank to disable Whisper — /watch will still work, but videos
# without native captions will come back frames-only.

GROQ_API_KEY=
OPENAI_API_KEY=

# Default watch behavior (the /watch first-run wizard sets this for you).
# Allowed values: transcript | efficient | balanced | token-burner
# Keep the value on its own line with no trailing comment.
# WATCH_DETAIL=balanced

# Download quality ceiling, as a max video height. Default 1080.
# Allowed values: 360 | 480 | 720 | 1080 | 1440 | 2160 | best
# Lower it on a slow connection; raise it for dense on-screen text.
# WATCH_QUALITY=1080
"""


def _which(name: str) -> str | None:
    return shutil.which(name)


def _check_binaries() -> list[str]:
    return [b for b in REQUIRED_BINARIES if not _which(b)]


def _ytdlp_run(args: list[str]) -> str | None:
    """Run yt-dlp and return its stdout, or None on any failure. Fail-open:
    a probe that cannot answer must never block or crash /watch."""
    try:
        result = subprocess.run(
            ["yt-dlp", *args],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _ytdlp_age_days(version: str | None) -> int | None:
    """Days since a yt-dlp CalVer release string (``YYYY.MM.DD``)."""
    if not version:
        return None
    head = version.strip().split()[0]
    for fmt in ("%Y.%m.%d", "%Y.%m.%d.%f"):
        try:
            released = datetime.strptime(head, fmt)
        except ValueError:
            continue
        return max(0, (datetime.now() - released).days)
    return None


def _probe_ytdlp_uncached(path: str) -> dict:
    version_out = _ytdlp_run(["--version"])
    version = version_out.strip().splitlines()[0].strip() if version_out else None

    # --list-impersonate-targets exits 0 and still lists every target when the
    # build has no curl_cffi — it just marks each one "(unavailable)" in the
    # Source column. So presence of rows proves nothing; a usable target is a
    # row NOT marked unavailable. Older builds lack the flag and return None.
    targets_out = _ytdlp_run(["--list-impersonate-targets"])
    impersonation: bool | None
    if targets_out is None:
        impersonation = None
    else:
        usable = [
            line for line in targets_out.splitlines()
            if line.strip()
            and not line.startswith("[")
            and not line.lower().startswith("client")
            and set(line.strip()) != {"-"}
            and "unavailable" not in line.lower()
        ]
        impersonation = bool(usable)

    return {
        "path": path,
        "version": version,
        "impersonation": impersonation,
        "checked_at": time.time(),
    }


def probe_ytdlp(*, force: bool = False) -> dict:
    """Installed yt-dlp version and whether it can impersonate a browser.

    Both facts predict a failed download and both are one command away from
    fixed, so they belong in preflight rather than being discovered halfway
    through a download. Cached for ``YTDLP_PROBE_TTL_SECONDS``; ``age_days`` is
    always recomputed from the cached version so it never goes stale itself.
    """
    path = _which("yt-dlp")
    if path is None:
        return {"path": None, "version": None, "impersonation": None, "age_days": None}

    cached: dict | None = None
    if not force:
        try:
            cached = json.loads(YTDLP_PROBE_CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
        if not isinstance(cached, dict):
            cached = None
        elif (
            cached.get("path") != path
            or (time.time() - float(cached.get("checked_at") or 0)) > YTDLP_PROBE_TTL_SECONDS
        ):
            cached = None

    probe = cached if cached is not None else _probe_ytdlp_uncached(path)
    if cached is None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            YTDLP_PROBE_CACHE.write_text(json.dumps(probe), encoding="utf-8")
        except OSError:
            pass

    probe = dict(probe)
    probe["age_days"] = _ytdlp_age_days(probe.get("version"))
    return probe


def ytdlp_warnings(probe: dict) -> list[str]:
    """Actionable one-liners for a yt-dlp that is likely to fail a download."""
    if probe.get("path") is None:
        return []
    warnings: list[str] = []
    age = probe.get("age_days")
    if isinstance(age, int) and age > YTDLP_STALE_DAYS:
        warnings.append(
            f"yt-dlp {probe.get('version')} is {age} days old. Sites change their "
            "defences every few weeks, so downloads are likely to be refused."
        )
    if probe.get("impersonation") is False:
        warnings.append(
            f"yt-dlp at {probe.get('path')} has no browser-impersonation targets "
            "(built without curl_cffi). YouTube's bot check refuses downloads "
            "from clients it cannot verify, while titles and captions keep working."
        )
    if warnings:
        warnings.append(f"Fix both with: {YTDLP_UPGRADE_COMMAND}")
    return warnings


_PERM_WARNED: set[str] = set()


def _check_file_permissions(path: Path) -> None:
    """Warn to stderr (once per path per process) if a secrets file is
    world/group readable."""
    key = str(path)
    if key in _PERM_WARNED:
        return
    try:
        mode = path.stat().st_mode
        if mode & 0o044:
            _PERM_WARNED.add(key)
            sys.stderr.write(
                f"[watch] WARNING: {path} is readable by other users. "
                f"Run: chmod 600 {path}\n"
            )
            sys.stderr.flush()
    except OSError:
        pass


def _read_env_key(name: str) -> str | None:
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    if not CONFIG_FILE.exists():
        return None
    _check_file_permissions(CONFIG_FILE)
    try:
        for line in CONFIG_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            if key.strip() != name:
                continue
            raw = raw.strip()
            if len(raw) >= 2 and raw[0] in ('"', "'") and raw[-1] == raw[0]:
                raw = raw[1:-1]
            return raw or None
    except OSError:
        return None
    return None


def _have_api_key() -> tuple[bool, str | None]:
    if _read_env_key("GROQ_API_KEY"):
        return True, "groq"
    if _read_env_key("OPENAI_API_KEY"):
        return True, "openai"
    return False, None


def is_first_run() -> bool:
    """True if the installer hasn't completed successfully yet."""
    return _read_env_key("SETUP_COMPLETE") != "true"


def _scaffold_env() -> bool:
    """Create ~/.config/watch/.env with placeholders if missing."""
    if CONFIG_FILE.exists():
        return False
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(ENV_TEMPLATE, encoding="utf-8")
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass
    return True


def _write_setup_complete() -> None:
    """Idempotently append SETUP_COMPLETE=true to .env.

    Used only after a fully successful install (deps + key). Future sessions
    detect this marker to skip wizard-style UI and stay silent.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = ""
    if CONFIG_FILE.exists():
        existing = CONFIG_FILE.read_text(encoding="utf-8")
        for line in existing.splitlines():
            if line.strip().startswith("SETUP_COMPLETE="):
                return
        if existing and not existing.endswith("\n"):
            existing += "\n"
        CONFIG_FILE.write_text(existing + "SETUP_COMPLETE=true\n", encoding="utf-8")
    else:
        CONFIG_FILE.write_text(ENV_TEMPLATE + "\nSETUP_COMPLETE=true\n", encoding="utf-8")
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass


def _brew_pkg(missing: list[str]) -> list[str]:
    pkgs: list[str] = []
    for bin_name in missing:
        if bin_name in ("ffmpeg", "ffprobe"):
            if "ffmpeg" not in pkgs:
                pkgs.append("ffmpeg")
        elif bin_name == "yt-dlp":
            if "yt-dlp" not in pkgs:
                pkgs.append("yt-dlp")
        else:
            pkgs.append(bin_name)
    return pkgs


def _install_macos(missing: list[str]) -> tuple[bool, str]:
    if _which("brew") is None:
        return False, (
            "Homebrew is not installed. Install it from https://brew.sh, then re-run setup. "
            "Or install manually: `brew install " + " ".join(_brew_pkg(missing)) + "`"
        )
    pkgs = _brew_pkg(missing)
    if not pkgs:
        return True, "nothing to install"
    cmd = ["brew", "install", *pkgs]
    print(f"[setup] running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return False, f"brew install failed with exit code {result.returncode}"
    return True, f"installed via brew: {', '.join(pkgs)}"


def _install_hint_linux(missing: list[str]) -> str:
    pkgs = _brew_pkg(missing)
    hints = []
    if "ffmpeg" in pkgs:
        hints.append("apt: `sudo apt install ffmpeg` or dnf: `sudo dnf install ffmpeg`")
    if "yt-dlp" in pkgs:
        hints.append("`pipx install yt-dlp` (recommended) or `pip install --user yt-dlp`")
    return "\n  ".join(hints) if hints else "nothing to install"


def _install_hint_windows(missing: list[str]) -> str:
    pkgs = _brew_pkg(missing)
    hints = []
    if "ffmpeg" in pkgs:
        hints.append("winget: `winget install Gyan.FFmpeg`")
    if "yt-dlp" in pkgs:
        hints.append("winget: `winget install yt-dlp.yt-dlp` or pip: `pip install --user yt-dlp`")
    return "\n  ".join(hints) if hints else "nothing to install"


def _status() -> dict:
    """Structured preflight snapshot.

    `status` describes the *ideal* state (a Whisper key is encouraged), so a
    keyless install still reports `needs_key` on the very first run — that's
    the agent's cue to encourage adding one.

    `can_proceed` is the operational gate: /watch can run as long as the
    binaries are present AND the user has either set a key or already finished
    setup (consciously opting out of Whisper). A keyless user who completed
    setup is NOT nagged on every call.
    """
    missing = _check_binaries()
    has_key, backend = _have_api_key()
    setup_complete = not is_first_run()

    if not missing and has_key:
        status = "ready"
    elif missing and not has_key:
        status = "needs_install_and_key"
    elif missing:
        status = "needs_install"
    else:
        status = "needs_key"

    can_proceed = (not missing) and (has_key or setup_complete)

    # A stale or impersonation-less yt-dlp predicts a failed download but does
    # not block one, so it is a warning and never gates can_proceed — plenty of
    # sources download fine from an old build.
    probe = probe_ytdlp()
    warnings = ytdlp_warnings(probe)

    cfg = get_config()
    return {
        "status": status,
        "can_proceed": can_proceed,
        "first_run": not setup_complete,
        "setup_complete": setup_complete,
        "missing_binaries": missing,
        "whisper_backend": backend,
        "has_api_key": has_key,
        "config_file": str(CONFIG_FILE),
        "watch_detail": cfg["detail"],
        "watch_quality": cfg["quality"],
        "ytdlp": probe,
        "warnings": warnings,
        "platform": platform.system(),
    }


def cmd_check() -> int:
    """Silent-on-success preflight.

    Exit 0 with no output when /watch can run. A keyless user who already
    finished setup (SETUP_COMPLETE=true) counts as ready — Whisper is
    encouraged, not required — so they are never nagged on follow-up calls.

    On a state that blocks /watch, print one actionable line to stderr:
      2 → binaries missing
      3 → genuine first run with no API key (encourage one)
      4 → both missing
    """
    s = _status()

    # Warnings print whether or not setup is otherwise complete: a ready install
    # with an expired yt-dlp is exactly the case that currently fails halfway
    # through a download with a message that names no cause.
    for warning in s["warnings"]:
        sys.stderr.write(f"[watch] warning: {warning}\n")
    if s["warnings"]:
        sys.stderr.flush()

    if s["can_proceed"]:
        return 0

    parts = []
    if s["missing_binaries"]:
        parts.append(f"missing binaries: {', '.join(s['missing_binaries'])}")
    if not s["has_api_key"] and not s["setup_complete"]:
        parts.append("no Whisper API key (GROQ_API_KEY or OPENAI_API_KEY)")
    installer = Path(__file__).resolve()
    sys.stderr.write(
        f"[watch] setup incomplete ({'; '.join(parts)}). "
        f"Run: python3 {installer}\n"
    )
    sys.stderr.flush()

    if s["missing_binaries"] and not s["has_api_key"]:
        return 4
    if s["missing_binaries"]:
        return 2
    return 3


def cmd_json() -> int:
    json.dump(_status(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_install() -> int:
    missing = _check_binaries()
    installed_deps = False
    if missing:
        system = platform.system()
        if system == "Darwin":
            ok, msg = _install_macos(missing)
            print(f"[setup] {msg}", file=sys.stderr)
            if not ok:
                return 2
            still_missing = _check_binaries()
            if still_missing:
                print(f"[setup] still missing after install: {', '.join(still_missing)}", file=sys.stderr)
                return 2
            installed_deps = True
        elif system == "Linux":
            print("[setup] dependencies missing on Linux — please install:", file=sys.stderr)
            print("  " + _install_hint_linux(missing), file=sys.stderr)
            return 2
        elif system == "Windows":
            print("[setup] dependencies missing on Windows — please install:", file=sys.stderr)
            print("  " + _install_hint_windows(missing), file=sys.stderr)
            return 2
        else:
            print(f"[setup] unsupported platform ({system}) for auto-install. Install manually:", file=sys.stderr)
            print(f"  missing: {', '.join(missing)}", file=sys.stderr)
            return 2

    created = _scaffold_env()
    if created:
        print(f"[setup] created config: {CONFIG_FILE}")
    else:
        print(f"[setup] config exists: {CONFIG_FILE}")

    # Re-probe rather than trusting the cache: the installer is exactly when a
    # yt-dlp may have just been installed or upgraded.
    for warning in ytdlp_warnings(probe_ytdlp(force=True)):
        print(f"[setup] warning: {warning}", file=sys.stderr)

    has_key, backend = _have_api_key()
    if has_key:
        _write_setup_complete()
        print(f"[setup] ready. whisper backend: {backend}")
        if installed_deps:
            print("[setup] installed dependencies; /watch is fully set up.")
        return 0

    print("")
    print("[setup] one step left: add a Whisper API key.")
    print("")
    print(f"  Edit {CONFIG_FILE} and set either:")
    print("    GROQ_API_KEY=...    (preferred — cheaper, faster; get one at console.groq.com/keys)")
    print("    OPENAI_API_KEY=...  (fallback; get one at platform.openai.com/api-keys)")
    print("")
    print("  Without a key, /watch still works but videos without captions come back frames-only.")
    return 3


def main() -> int:
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--check":
            return cmd_check()
        if arg == "--json":
            return cmd_json()
    return cmd_install()


if __name__ == "__main__":
    raise SystemExit(main())
