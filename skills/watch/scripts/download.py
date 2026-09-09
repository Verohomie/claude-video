#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from config import DEFAULT_QUALITY  # noqa: E402


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}

# Keep the tail of yt-dlp's output for failure diagnosis. Bounded so a long
# fragment-retry storm can't grow unboundedly in memory; the useful error is
# always at the end.
_LOG_TAIL_LINES = 200


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        print(
            f"[watch] warning: {p.suffix} is not a known video extension, proceeding anyway",
            file=sys.stderr,
        )
    return {
        "video_path": str(p),
        "subtitle_path": None,
        "info": {"title": p.name, "url": str(p)},
        "downloaded": False,
    }


def _pick_subtitle(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    preferred = [
        c for c in candidates
        if any(marker in c.name for marker in (".en.", ".en-US.", ".en-GB.", ".en-orig."))
    ]
    return preferred[0] if preferred else candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".opus"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def fetch_captions(url: str, out_dir: Path) -> dict:
    """Fetch metadata and best available VTT captions without downloading video."""
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]
    subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)
    return {
        "video_path": None,
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": False,
    }


def _read_info(info_path: Path, url: str) -> dict:
    info: dict = {}
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
            info = {
                "title": raw.get("title"),
                "uploader": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "url": raw.get("webpage_url") or url,
            }
        except Exception as exc:
            print(f"[watch] info.json parse failed: {exc}", file=sys.stderr)
            info = {"url": url}
    return info


def format_selector(quality: str = DEFAULT_QUALITY, audio_only: bool = False) -> str:
    """Build the yt-dlp ``-f`` expression for a max-height ceiling.

    ``quality`` is a height in pixels as a string, or ``"best"`` for no ceiling.
    The trailing ``/bv+ba/b`` fallback keeps a video that only exists above the
    ceiling downloadable rather than failing outright — a ceiling is a
    preference, not a requirement.
    """
    if audio_only:
        return "ba/bestaudio"
    if str(quality).lower() == "best":
        return "bv*+ba/b"
    height = str(quality)
    return f"bv*[height<={height}]+ba/b[height<={height}]/bv+ba/b"


# Substrings yt-dlp prints for failures a user can actually act on, each mapped
# to what to do about it. Ordered: the first match wins, so put the specific
# bot-check signatures ahead of the generic 403.
_FAILURE_HINTS: list[tuple[tuple[str, ...], str]] = [
    (
        ("sign in to confirm you're not a bot", "sign in to confirm youre not a bot",
         "confirm you're not a bot", "only images are available"),
        "YouTube's bot check refused this download. It usually means your yt-dlp "
        "is out of date or was built without browser impersonation.",
    ),
    (
        ("http error 403", "403: forbidden", "403 forbidden"),
        "The host returned 403 Forbidden for the video data. Reading titles and "
        "caption lists still works, which is why this looks confusing — the usual "
        "cause is an out-of-date yt-dlp that the site no longer accepts.",
    ),
    (
        ("http error 429", "too many requests"),
        "The host is rate-limiting this machine (HTTP 429). Wait a few minutes "
        "and retry; there is nothing to fix locally.",
    ),
    (
        ("sign in to confirm your age", "age-restricted", "age restricted"),
        "This video is age-restricted and needs a signed-in session. Pass "
        "cookies to yt-dlp, or download it by hand and run /watch on the file.",
    ),
    (
        ("private video", "members-only", "join this channel", "video unavailable"),
        "The host says this video is not publicly available (private, "
        "members-only, or removed). Nothing local will fix it.",
    ),
    (
        ("not available in your country", "geo restricted", "geo-restricted"),
        "This video is geo-restricted and the host refused it from this location.",
    ),
    (
        ("requested format is not available",),
        "No stream matched the requested quality ceiling. Re-run with "
        "`--quality best` to take whatever the host offers.",
    ),
]

_FIX_COMMAND = (
    "pipx install --force 'yt-dlp[default,curl-cffi]'   "
    "(or: python3 -m pip install --upgrade 'yt-dlp[default,curl-cffi]')"
)


def diagnose_failure(output: str) -> str | None:
    """Map yt-dlp's output onto a plain-language cause, or None if unrecognized."""
    if not output:
        return None
    haystack = output.lower()
    for needles, message in _FAILURE_HINTS:
        if any(n in haystack for n in needles):
            return message
    return None


def _ytdlp_state_line() -> str | None:
    """One line describing the installed yt-dlp, for a failure message.

    Imported lazily and fail-open: diagnosing a download failure must never
    itself raise, and a normal run should not pay for the probe.
    """
    try:
        from setup import probe_ytdlp  # noqa: PLC0415

        probe = probe_ytdlp()
    except Exception:
        return None
    version = probe.get("version")
    if not version:
        return None
    age = probe.get("age_days")
    age_note = f", {age} days old" if isinstance(age, int) else ""
    impersonation = probe.get("impersonation")
    imp_note = (
        "no browser impersonation available" if impersonation is False
        else "impersonation available" if impersonation is True
        else "impersonation state unknown"
    )
    return f"Your yt-dlp: {version}{age_note} at {probe.get('path')} — {imp_note}."


def _run_streaming(cmd: list[str]) -> tuple[int, str]:
    """Run ``cmd``, echoing its output to stderr live while keeping the tail.

    yt-dlp's progress needs to stay visible, but the failure path needs the text
    to diagnose. Tee rather than choosing one.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stderr.write(line)
        tail.append(line)
        if len(tail) > _LOG_TAIL_LINES:
            del tail[0]
    sys.stderr.flush()
    return proc.wait(), "".join(tail)


def _download_failure_message(
    url: str, out_dir: Path, returncode: int, log_tail: str
) -> str:
    """Say what went wrong and what to type, instead of only that nothing appeared."""
    cause = diagnose_failure(log_tail)
    lines = [f"yt-dlp did not produce a video file for {url} (exit {returncode})."]
    if cause:
        lines.append("")
        lines.append(f"Cause: {cause}")
        state = _ytdlp_state_line()
        if state:
            lines.append(state)
        if "yt-dlp" in cause:
            lines.append("")
            lines.append("Fix — install a current yt-dlp with impersonation support:")
            lines.append(f"  {_FIX_COMMAND}")
            lines.append(
                "Homebrew's build lags and omits curl_cffi, so its impersonation "
                "targets all read unavailable. Treat yt-dlp as perishable: sites "
                "change their defences every few weeks."
            )
    else:
        lines.append("")
        lines.append(
            "No recognized cause in yt-dlp's output — the full log is above. "
            f"Partial files (if any) are in {out_dir}."
        )
    return "\n".join(lines)


def download_url(
    url: str,
    out_dir: Path,
    audio_only: bool = False,
    quality: str = DEFAULT_QUALITY,
) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    fmt = format_selector(quality, audio_only=audio_only)
    cmd = [
        "yt-dlp",
        "-N", "8",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]

    # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
    # the video itself downloaded fine. Treat "video file present" as success.
    returncode, log_tail = _run_streaming(cmd)
    video = _pick_video(out_dir)
    if video is None:
        raise SystemExit(_download_failure_message(url, out_dir, returncode, log_tail))

    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": True,
    }


def download(
    source: str,
    out_dir: Path,
    audio_only: bool = False,
    quality: str = DEFAULT_QUALITY,
) -> dict:
    if is_url(source):
        return download_url(source, out_dir, audio_only=audio_only, quality=quality)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
