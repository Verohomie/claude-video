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
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from config import DEFAULT_QUALITY  # noqa: E402
from setup import (  # noqa: E402
    YTDLP_UPGRADE_COMMAND,
    probe_ytdlp,
    resolve_ytdlp,
    ytdlp_warnings,
)


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
    # Resolve by capability rather than PATH order — see resolve_ytdlp. A bare
    # "yt-dlp" here is what let Homebrew's stale, curl_cffi-less build shadow
    # the pipx copy that setup itself tells the user to install.
    ytdlp = resolve_ytdlp()
    if shutil.which(ytdlp) is None:
        raise SystemExit(
            "yt-dlp is not installed. Install with:\n"
            f"  {YTDLP_UPGRADE_COMMAND}\n"
            "(Homebrew's build lags and omits curl_cffi, so downloads get refused.)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")
    cmd = [
        ytdlp,
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


def fetch_duration(url: str, timeout: float = 30.0) -> float | None:
    """Length of a video in seconds, read from its metadata; None if unknown.

    Metadata only -- nothing is downloaded. Used to price a Gemini run before
    it is sent. Fails open: a metadata refusal must not block a Gemini run,
    since Gemini exists precisely for the case where YouTube refuses us.
    """
    try:
        proc = subprocess.run(
            [resolve_ytdlp(), "--skip-download", "--no-playlist", "--no-warnings",
             "--print", "duration", "--", url],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        value = float(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    return value if value > 0 else None


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
# to a key and to what to do about it. Ordered: the first match wins, so put the
# specific bot-check signatures ahead of the generic 403.
#
# The two refusal cases deliberately state no cause. Whether a refusal means
# "your tool is stale" or "your address is throttled" cannot be read off the
# error text at all — both produce the identical message — so the cause is
# decided later, from the local yt-dlp's measured state. See
# _refusal_advice.
_FAILURE_HINTS: list[tuple[str, tuple[str, ...], str]] = [
    (
        "bot_check",
        ("sign in to confirm you're not a bot", "sign in to confirm youre not a bot",
         "confirm you're not a bot", "only images are available"),
        "YouTube's bot check refused this download. Either the local yt-dlp is a "
        "build it will not accept, or this machine's address is being throttled.",
    ),
    (
        "forbidden",
        ("http error 403", "403: forbidden", "403 forbidden"),
        "The host returned 403 Forbidden for the video data. Reading titles and "
        "caption lists still works, which is why this looks confusing: metadata "
        "and video data are gated separately.",
    ),
    (
        "rate_limit",
        ("http error 429", "too many requests"),
        "The host is rate-limiting this machine (HTTP 429). Wait a few minutes "
        "and retry; there is nothing to fix locally.",
    ),
    (
        "age_restricted",
        ("sign in to confirm your age", "age-restricted", "age restricted"),
        "This video is age-restricted and needs a signed-in session. Download it "
        "by hand and run /watch on the file. (Passing your browser's cookies to "
        "yt-dlp also works, but hands it a live login — see the cookie warning "
        "in SKILL.md before doing that.)",
    ),
    (
        "unavailable",
        ("private video", "members-only", "join this channel", "video unavailable"),
        "The host says this video is not publicly available (private, "
        "members-only, or removed). Nothing local will fix it.",
    ),
    (
        "geo_blocked",
        ("not available in your country", "geo restricted", "geo-restricted"),
        "This video is geo-restricted and the host refused it from this location.",
    ),
    (
        "no_format",
        ("requested format is not available",),
        "No stream matched the requested quality ceiling. Re-run with "
        "`--quality best` to take whatever the host offers.",
    ),
]

# Refusals where the local tool might or might not be the cause, so the message
# has to measure before it prescribes.
_REFUSAL_KEYS = {"bot_check", "forbidden"}

_FIX_COMMAND = (
    "pipx install --force 'yt-dlp[default,curl-cffi]'   "
    "(or: python3 -m pip install --upgrade 'yt-dlp[default,curl-cffi]')"
)


def classify_failure(output: str) -> str | None:
    """Key of the first failure signature present in yt-dlp's output."""
    if not output:
        return None
    haystack = output.lower()
    for key, needles, _message in _FAILURE_HINTS:
        if any(n in haystack for n in needles):
            return key
    return None


def diagnose_failure(output: str) -> str | None:
    """Map yt-dlp's output onto a plain-language cause, or None if unrecognized."""
    key = classify_failure(output)
    if key is None:
        return None
    return next(message for k, _needles, message in _FAILURE_HINTS if k == key)


_YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be", "www.youtu.be",
}
_OEMBED_URL = "https://www.youtube.com/oembed"
_OEMBED_TIMEOUT = 6


def _is_youtube(url: str) -> bool:
    try:
        return urlparse(url).netloc.lower() in _YOUTUBE_HOSTS
    except ValueError:
        return False


def probe_video_liveness(url: str) -> tuple[str, str | None]:
    """Is the video itself still there? ``("live"|"gone"|"unknown", detail)``.

    A refused download and a dead video produce indistinguishable output from
    the download path, which leaves the user unable to tell "wait and try
    later" from "this will never work". YouTube's oEmbed endpoint answers that
    question on a different gate to the video stream: it kept returning correct
    titles throughout a bot-check block that refused every download.

    One request, only on the failure path, only for YouTube, and only to the
    host the URL already points at. Fail-open to "unknown" — a diagnosis that
    cannot itself be diagnosed must never replace the real error.
    """
    if not _is_youtube(url):
        return "unknown", None
    query = urlencode({"url": url, "format": "json"})
    request = Request(
        f"{_OEMBED_URL}?{query}",
        headers={"User-Agent": "claude-video-watch-skill"},
    )
    try:
        with urlopen(request, timeout=_OEMBED_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        # 401 private, 403 restricted, 404 removed. 400 is what YouTube returns
        # for an ID it cannot resolve at all — measured, not assumed: a bad ID
        # gives 400 where a good one gives 200. Anything else is the endpoint
        # having its own problem and proves nothing about the video.
        if exc.code in (400, 401, 403, 404):
            return "gone", f"oEmbed returned HTTP {exc.code}"
        return "unknown", None
    except Exception:
        return "unknown", None
    title = payload.get("title")
    author = payload.get("author_name")
    if not title:
        return "unknown", None
    detail = f'"{title}"' + (f" by {author}" if author else "")
    return "live", detail


def _refusal_advice(url: str, probe: dict) -> list[str]:
    """What to actually do about a refused download, measured rather than assumed.

    The old advice said a refusal is "almost always an out-of-date yt-dlp". That
    is true often enough to be worth saying first, and wrong in a way that costs
    an hour when it is wrong: on 2026-09-22 a current yt-dlp with working
    impersonation was refused on every player client, every PO-token route and a
    cookie-less browser profile, because the refusal was per-address throttling
    and had nothing to do with the tool. Probing harder made it worse — a control
    video that had just succeeded failed on the next attempt.

    So: if the local install has a known fault, that is the lead. If it does not,
    say so plainly and tell the user to stop, because attempts are what is being
    counted.
    """
    lines: list[str] = []
    state, detail = probe_video_liveness(url)
    if state == "live":
        lines.append(
            f"The video itself is fine — YouTube still returns {detail} for it. "
            "So this is the download being blocked, not a dead or private video."
        )
    elif state == "gone":
        lines.append(
            f"YouTube will not describe this video either ({detail}), so it is "
            "most likely private, removed, or the ID is wrong. Nothing local "
            "will fix that."
        )
        return lines

    faults = ytdlp_warnings(probe)
    if faults:
        lines.append("")
        lines.append("Your yt-dlp is the likely cause:")
        lines.extend(f"  - {fault}" for fault in faults)
        lines.append("")
        lines.append("Fix — install a current yt-dlp with impersonation support:")
        lines.append(f"  {_FIX_COMMAND}")
        lines.append(
            "Homebrew's build lags and omits curl_cffi, so its impersonation "
            "targets all read unavailable. Treat yt-dlp as perishable: sites "
            "change their defences every few weeks."
        )
        return lines

    lines.append("")
    lines.append(
        "Your yt-dlp is current and has working browser impersonation, so this "
        "is NOT the out-of-date-tool case and reinstalling it will change "
        "nothing. The host is refusing this machine's address."
    )
    lines.append("")
    lines.append(
        "Do not retry in a loop, and do not sweep player clients: attempts are "
        "the thing being counted, and probing tightens the block rather than "
        "getting around it. Wait — tens of minutes at least — and try again "
        "later. If it is urgent, download the file by hand on another network "
        "and run /watch on the local file."
    )
    return lines


def _ytdlp_state_line() -> str | None:
    """One line describing the yt-dlp that actually ran, for a failure message.

    Fail-open: diagnosing a download failure must never itself raise. Naming the
    binary matters here — with several copies installed, "which one ran?" is the
    first thing anyone needs to know.
    """
    try:
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
    line = f"Ran yt-dlp {version}{age_note} at {probe.get('path')} — {imp_note}."
    shadowed = probe.get("shadowed") or []
    if shadowed:
        line += f" (Chosen over {len(shadowed)} copy/copies earlier on PATH.)"
    return line


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
    key = classify_failure(log_tail)
    cause = diagnose_failure(log_tail)
    lines = [f"yt-dlp did not produce a video file for {url} (exit {returncode})."]
    if cause:
        lines.append("")
        lines.append(f"Cause: {cause}")
        state = _ytdlp_state_line()
        if state:
            lines.append(state)
        if key in _REFUSAL_KEYS:
            try:
                lines.extend(_refusal_advice(url, probe_ytdlp()))
            except Exception:
                # Diagnosing a failure must never itself raise and swallow the
                # error the user actually needs to see.
                lines.append("")
                lines.append(f"If yt-dlp is out of date, upgrade it: {_FIX_COMMAND}")
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
    # Resolve by capability rather than PATH order — see resolve_ytdlp. A bare
    # "yt-dlp" here is what let Homebrew's stale, curl_cffi-less build shadow
    # the pipx copy that setup itself tells the user to install.
    ytdlp = resolve_ytdlp()
    if shutil.which(ytdlp) is None:
        raise SystemExit(
            "yt-dlp is not installed. Install with:\n"
            f"  {YTDLP_UPGRADE_COMMAND}\n"
            "(Homebrew's build lags and omits curl_cffi, so downloads get refused.)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    fmt = format_selector(quality, audio_only=audio_only)
    cmd = [
        ytdlp,
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
