#!/usr/bin/env python3
"""/watch entry point: download video, extract frames, parse transcript.

Prints a markdown report to stdout listing frame paths + transcript. Claude
then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from config import QUALITIES, frame_cap, get_config  # noqa: E402
from download import download, fetch_captions, is_url  # noqa: E402
from frames import DEFAULT_RESOLUTION, MAX_FPS, SCREEN_FRAME_CAP, SCREEN_RESOLUTION, auto_fps, auto_fps_focus, detect_screen_recording, extract_at_timestamps, extract_keyframes, extract_scene_or_uniform, format_time, get_metadata, merge_frames, parse_time, parse_timestamps  # noqa: E402
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
from whisper import load_api_key, transcribe_video  # noqa: E402


def write_manifest(
    work: Path,
    *,
    source: str,
    info: dict,
    meta: dict,
    frames: list[dict],
    detail: str,
    quality: str | None,
    resolution: int,
    resolution_source: str,
    screen_evidence: dict,
    effective_start: float,
    effective_end: float,
    focused: bool,
    transcript_source: str | None,
    transcript_segments: list[dict],
) -> Path | None:
    """Write frames.json next to the extracted images.

    The frame → timestamp mapping otherwise exists only in the stdout report,
    which is routinely piped or truncated (it is mostly transcript), leaving a
    directory of JPEGs with no way to say when any of them happened. That is
    fine for a summary and useless the moment a frame is cited as evidence.

    Best-effort: a manifest that cannot be written must never fail the run,
    since every frame and the whole report are already produced.
    """
    payload = {
        "source": source,
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "duration_seconds": meta.get("duration_seconds"),
        "video_width": meta.get("width"),
        "video_height": meta.get("height"),
        "download_quality_ceiling": quality,
        "detail": detail,
        "frame_width": resolution,
        "frame_width_source": resolution_source,
        "screen_recording": screen_evidence or None,
        "range": {
            "focused": focused,
            "start_seconds": round(effective_start, 3),
            "end_seconds": round(effective_end, 3),
        },
        "transcript": {
            "source": transcript_source,
            "segments": len(transcript_segments),
        },
        "frames": [
            {
                "file": Path(frame["path"]).name,
                "path": frame["path"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "timestamp": format_time(frame["timestamp_seconds"]),
                "reason": frame.get("reason", "selected"),
                # Present only when this frame absorbed near-duplicates: the
                # source was unchanged from its timestamp through to here.
                **(
                    {"represents_until_seconds": frame["represents_until"]}
                    if frame.get("represents_until") is not None
                    else {}
                ),
            }
            for frame in frames
        ],
    }
    path = work / "frames.json"
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[watch] could not write {path}: {exc}", file=sys.stderr)
        return None
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="watch",
        description="Download a video, extract auto-scaled frames, and surface the transcript.",
    )
    ap.add_argument("source", help="Video URL or local file path")
    ap.add_argument("--max-frames", type=int, default=None, help="Override frame cap")
    ap.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=f"Frame width in pixels. Default: auto — {DEFAULT_RESOLUTION} for ordinary "
             f"footage, {SCREEN_RESOLUTION} when the source is detected as a screen "
             "recording (small on-screen text is unreadable below that).",
    )
    ap.add_argument(
        "--quality",
        type=str,
        default=None,
        choices=sorted(QUALITIES),
        help="Max download height in pixels, or 'best' for no ceiling. "
             "Default 1080 (override with WATCH_QUALITY in the config file).",
    )
    ap.add_argument("--fps", type=float, default=None, help="Override auto-fps")
    ap.add_argument(
        "--detail",
        choices=["transcript", "efficient", "balanced", "token-burner"],
        default=None,
        help="Fidelity/speed dial: transcript (no frames), efficient (fast keyframes, cap 50), "
             "balanced (scene, cap 100), token-burner (scene, uncapped).",
    )
    ap.add_argument(
        "--timestamps",
        type=str,
        default=None,
        help="Comma-separated absolute timestamps (SS, MM:SS, HH:MM:SS) to grab a frame at, "
             "e.g. transcript-flagged 'look here' moments. Added on top of the detail frames "
             "(reserved against the cap); with --detail transcript these become the only frames.",
    )
    ap.add_argument("--start", type=str, default=None, help="Range start (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--end", type=str, default=None, help="Range end (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--out-dir", type=str, default=None, help="Working directory (default: tmp)")
    ap.add_argument(
        "--no-whisper",
        action="store_true",
        help="Disable Whisper fallback. Report frames-only if no captions available.",
    )
    ap.add_argument(
        "--whisper",
        choices=["groq", "openai"],
        default=None,
        help="Force a specific Whisper backend. Default: prefer Groq, fall back to OpenAI.",
    )
    ap.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable near-duplicate frame removal. Keeps visually identical "
             "frames (static screen recordings, held slides) instead of collapsing them.",
    )
    args = ap.parse_args()

    config = get_config()
    detail = args.detail or str(config["detail"])
    quality = args.quality or str(config["quality"])
    max_fps = float(config["max_fps"])
    configured_cap = frame_cap(detail)
    if args.max_frames is not None:
        max_frames = args.max_frames
    else:
        max_frames = configured_cap
    if max_frames is not None and max_frames < 1:
        raise SystemExit("--max-frames must be greater than zero")
    budget_cap = max_frames if max_frames is not None else 100
    cue_timestamps = parse_timestamps(args.timestamps)

    if args.out_dir:
        work = Path(args.out_dir).expanduser().resolve()
    else:
        work = Path(tempfile.mkdtemp(prefix="watch-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[watch] working dir: {work}", file=sys.stderr)

    url_source = is_url(args.source)
    dl: dict = {"subtitle_path": None, "info": {}, "downloaded": False}
    transcript_segments: list[dict] = []
    transcript_text: str | None = None
    transcript_source: str | None = None
    video_path: str | None = None

    if url_source:
        print("[watch] checking metadata/captions via yt-dlp…", file=sys.stderr)
        dl = fetch_captions(args.source, work / "download")
        if dl.get("subtitle_path"):
            try:
                transcript_segments = parse_vtt(dl["subtitle_path"])
                transcript_text = format_transcript(transcript_segments)
                transcript_source = "captions"
            except Exception as exc:
                print(f"[watch] subtitle parse failed: {exc}", file=sys.stderr)
                transcript_segments = []

    # --timestamps needs the video for frame grabs, so it overrides the
    # transcript-mode download skip (and forces a full, not audio-only, fetch).
    audio_only = detail == "transcript" and not cue_timestamps
    if detail == "transcript" and transcript_segments and not cue_timestamps:
        video_path = None
    else:
        if url_source:
            print(
                "[watch] downloading audio via yt-dlp…" if audio_only
                else f"[watch] downloading video via yt-dlp (quality ceiling {quality})…",
                file=sys.stderr,
            )
            dl = download(
                args.source,
                work / "download",
                audio_only=audio_only,
                quality=quality,
            )
        else:
            print("[watch] using local file…", file=sys.stderr)
            dl = download(args.source, work / "download")
        video_path = dl["video_path"]

    meta = get_metadata(video_path) if video_path else {
        "duration_seconds": float((dl.get("info") or {}).get("duration") or 0),
        "width": None,
        "height": None,
        "codec": None,
        "has_audio": False,
        "has_video": False,
    }
    full_duration = meta["duration_seconds"]

    # An audio-only source (a local .mp3/.m4a, or a download that only produced
    # audio) has nothing to extract frames from. Say so once and carry on in
    # transcript-only mode: the audio is still worth transcribing, and dying here
    # would throw away a download that already succeeded.
    if video_path and not meta.get("has_video", True):
        print(
            "[watch] no video stream in this source — continuing transcript-only",
            file=sys.stderr,
        )
        audio_source = video_path
        video_path = None
    else:
        audio_source = video_path

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    if start_sec is not None and start_sec < 0:
        raise SystemExit("--start must be non-negative")
    if end_sec is not None and start_sec is not None and end_sec <= start_sec:
        raise SystemExit("--end must be greater than --start")
    if full_duration > 0 and start_sec is not None and start_sec >= full_duration:
        raise SystemExit(f"--start {start_sec:.1f}s is past end of video ({full_duration:.1f}s)")

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)
    focused = start_sec is not None or end_sec is not None

    # Frame width: honour an explicit --resolution, otherwise measure whether the
    # source is a screen recording. A tutorial's whole payload is small on-screen
    # text, which 512px destroys — and a better download does not help, because
    # the loss happens at extraction. Raising the width ~9x the pixels per frame,
    # so a detected screen recording also lowers the frame cap to keep the image
    # token bill near where it was; an explicit --max-frames still wins.
    screen_evidence: dict = {}
    if args.resolution is not None:
        resolution = args.resolution
        resolution_source = "explicit"
    elif video_path is None:
        resolution = DEFAULT_RESOLUTION
        resolution_source = "default"
    else:
        is_screen, screen_evidence = detect_screen_recording(
            video_path, meta, start_seconds=start_sec, end_seconds=end_sec
        )
        resolution = SCREEN_RESOLUTION if is_screen else DEFAULT_RESOLUTION
        resolution_source = "screen-recording" if is_screen else "default"
        if is_screen:
            print(
                f"[watch] screen recording detected ({screen_evidence['reason']}) — "
                f"frames at {resolution}px so on-screen text stays readable",
                file=sys.stderr,
            )
            if args.max_frames is None:
                capped = SCREEN_FRAME_CAP if max_frames is None else min(max_frames, SCREEN_FRAME_CAP)
                if capped != max_frames:
                    print(
                        f"[watch] frame cap {max_frames or 'unlimited'} → {capped} to offset the "
                        "~9x image-token cost per frame (override with --max-frames)",
                        file=sys.stderr,
                    )
                max_frames = capped
                budget_cap = capped

    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=budget_cap, max_fps=max_fps)
    else:
        fps, target = auto_fps(effective_duration, max_frames=budget_cap, max_fps=max_fps)
    if args.fps is not None:
        if args.fps > max_fps:
            print(
                f"[watch] --fps {args.fps:g} exceeds the {max_fps:g} fps ceiling — using "
                f"{max_fps:g}. Raise it with WATCH_MAX_FPS in ~/.config/watch/.env, and pair "
                "it with --start/--end so the frame bill stays small.",
                file=sys.stderr,
            )
        fps = min(args.fps, max_fps)
        target = max(1, int(round(fps * effective_duration)))

    if transcript_segments and focused:
        transcript_segments = filter_range(transcript_segments, start_sec, end_sec)
        transcript_text = format_transcript(transcript_segments)

    scope = (
        f"{format_time(effective_start)}-{format_time(effective_end)} ({effective_duration:.1f}s)"
        if focused else f"full {effective_duration:.1f}s"
    )
    frames: list[dict] = []
    frame_meta: dict = {"engine": "none", "candidate_count": 0, "selected_count": 0, "fallback": False}
    cue_frames: list[dict] = []
    cue_meta: dict = {}

    # Transcript cues are pinned: extracted first and counted against the cap so
    # the detail engine never evicts the moments the user explicitly asked for.
    if cue_timestamps and video_path:
        cue_frames, cue_meta = extract_at_timestamps(
            video_path,
            work / "frames",
            cue_timestamps,
            resolution=resolution,
            max_frames=max_frames,
            start_seconds=start_sec,
            end_seconds=end_sec,
        )
        if cue_meta.get("dropped_out_of_window"):
            print(
                f"[watch] {cue_meta['dropped_out_of_window']} cue timestamp(s) outside the "
                "focus range — dropped",
                file=sys.stderr,
            )

    detail_budget = max_frames if max_frames is None else max(0, max_frames - len(cue_frames))
    if detail != "transcript" and video_path and detail_budget != 0:
        cap_label = "unlimited" if detail_budget is None else str(detail_budget)
        engine_label = "keyframes" if detail == "efficient" else "scene-aware frames"
        print(
            f"[watch] extracting {engine_label} over {scope} "
            f"(target {target}, cap {cap_label})…",
            file=sys.stderr,
        )
        if detail == "efficient":
            frames, frame_meta = extract_keyframes(
                video_path,
                work / "frames",
                resolution=resolution,
                max_frames=detail_budget,
                start_seconds=start_sec,
                end_seconds=end_sec,
                dedup=not args.no_dedup,
            )
        else:  # balanced, token-burner
            frames, frame_meta = extract_scene_or_uniform(
                video_path,
                work / "frames",
                fps=fps,
                target_frames=target,
                resolution=resolution,
                max_frames=detail_budget,
                start_seconds=start_sec,
                end_seconds=end_sec,
                dedup=not args.no_dedup,
                # Already probed for the metadata block above; passing it keeps
                # the coverage check and the fps clamp from re-running ffprobe.
                span_seconds=effective_duration,
            )

    if cue_frames:
        frames = merge_frames(frames, cue_frames)

    if not transcript_segments and dl.get("subtitle_path"):
        try:
            all_segments = parse_vtt(dl["subtitle_path"])
            transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
            transcript_text = format_transcript(transcript_segments)
            transcript_source = "captions"
        except Exception as exc:
            print(f"[watch] subtitle parse failed: {exc}", file=sys.stderr)

    # audio_source, not video_path: an audio-only file clears video_path so the
    # frame engines are skipped, but its audio is exactly what we want to send.
    if not transcript_segments and not args.no_whisper and audio_source and meta.get("has_audio"):
        backend, api_key = load_api_key(args.whisper)
        if backend and api_key:
            try:
                all_segments, used_backend = transcribe_video(
                    audio_source,
                    work / "audio.mp3",
                    backend=backend,
                    api_key=api_key,
                )
                transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
                transcript_text = format_transcript(transcript_segments)
                transcript_source = f"whisper ({used_backend})"
            except SystemExit as exc:
                print(f"[watch] whisper fallback failed: {exc}", file=sys.stderr)
        else:
            hint = (
                f"--whisper {args.whisper} was set but the matching API key is missing"
                if args.whisper else
                "no subtitles and no Whisper API key found"
            )
            setup_py = SCRIPT_DIR / "setup.py"
            print(
                f"[watch] {hint} — run `python3 {setup_py}` to enable the Whisper fallback",
                file=sys.stderr,
            )
    elif not transcript_segments and video_path and not meta.get("has_audio"):
        print("[watch] no audio stream found — proceeding without transcription", file=sys.stderr)

    info = dl.get("info") or {}

    print()
    print("# watch: video report")
    print()
    print(f"- **Source:** {args.source}")
    if info.get("title"):
        print(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        print(f"- **Uploader:** {info['uploader']}")
    print(f"- **Duration:** {format_time(full_duration)} ({full_duration:.1f}s)")
    if focused:
        print(
            f"- **Focus range:** {format_time(effective_start)} → {format_time(effective_end)} "
            f"({effective_duration:.1f}s)"
        )
    if meta.get("width") and meta.get("height"):
        ceiling = "" if not dl.get("downloaded") else f", downloaded at a {quality} ceiling"
        print(
            f"- **Resolution:** {meta['width']}x{meta['height']} "
            f"({meta.get('codec') or 'unknown codec'}{ceiling})"
        )
    range_mode = "focused" if focused else "full"
    print(f"- **Detail:** {detail}")
    detail_count = frame_meta.get("selected_count", 0)
    if detail != "transcript":
        cap_label = "unlimited" if detail_budget is None else str(detail_budget)
        engine = frame_meta.get("engine", "scene")
        # Say WHY it fell back. "sparse" means the detector found almost no cuts;
        # "coverage" means it found enough but left a stretch of the video with no
        # frame at all — a different thing to know when reading the answer.
        reason = frame_meta.get("fallback_reason")
        if frame_meta.get("fallback") and reason == "coverage":
            rejected = frame_meta.get("fallback_candidate_count")
            fallback = f" — fell back from scene: {rejected} cuts left a gap in the range"
        elif frame_meta.get("fallback") and reason == "sparse":
            fallback = " — fell back from scene: too few cuts to cover the range"
        elif frame_meta.get("fallback"):
            fallback = " with uniform fallback"
        else:
            fallback = ""
        deduped = frame_meta.get("deduped_count", 0)
        dedup_note = f", {deduped} near-duplicate{'s' if deduped != 1 else ''} dropped" if deduped else ""
        print(
            f"- **Frames:** {detail_count} selected from {frame_meta.get('candidate_count', detail_count)} "
            f"candidates ({engine}{fallback}{dedup_note}, {range_mode} range, budget {target}, cap {cap_label})"
        )
        # State the span outright. Without this the only coverage signal is the
        # last frame's timestamp, which understates it whenever duplicates were
        # dropped -- a static tail collapses to one frame and the run looks like
        # it stopped early when it had in fact sampled the whole range.
        if frames:
            covered_to = max(
                float(fr.get("represents_until") or fr["timestamp_seconds"]) for fr in frames
            )
            print(
                f"- **Frame coverage:** sampled across {format_time(effective_start)}–"
                f"{format_time(effective_end)}; frames represent up to {format_time(covered_to)}"
            )
    elif not cue_frames:
        print("- **Frames:** skipped (transcript detail)")
    if cue_frames:
        dropped = cue_meta.get("dropped_out_of_window", 0)
        drop_note = f", {dropped} dropped outside range" if dropped else ""
        print(
            f"- **Cue frames:** {len(cue_frames)} at transcript-flagged timestamps "
            f"(transcript-cue{drop_note})"
        )
    if frames:
        size_note = {
            "explicit": " (set with --resolution)",
            "screen-recording": f" (auto-raised: {screen_evidence.get('reason', 'screen recording detected')})",
        }.get(resolution_source, "")
        print(f"- **Frame size:** max {resolution}px wide, max 1998px tall{size_note}")
    if transcript_segments:
        in_range = " in range" if focused else ""
        print(
            f"- **Transcript:** {len(transcript_segments)} segments{in_range} "
            f"(via {transcript_source or 'captions'})"
        )
    else:
        print("- **Transcript:** none available")

    if detail == "token-burner" and len(frames) > 250:
        print()
        print(
            f"> **Warning:** token-burner detail selected {len(frames)} frames. "
            "This may use a large number of image tokens."
        )

    if not focused and full_duration > 600 and detail not in ("transcript", "token-burner"):
        mins = int(full_duration // 60)
        print()
        print(
            f"> **Warning:** This is a {mins}-minute video. Frame coverage is sparse at this length "
            f"under `{detail}` detail — its cap spreads thin across the full clip. For better results, "
            "re-run with `--start HH:MM:SS --end HH:MM:SS` to zoom into a section, or use "
            "`--detail token-burner` to keep every scene-change frame across the whole video."
        )

    manifest_path = write_manifest(
        work,
        source=args.source,
        info=info,
        meta=meta,
        frames=frames,
        detail=detail,
        quality=quality if dl.get("downloaded") else None,
        resolution=resolution,
        resolution_source=resolution_source,
        screen_evidence=screen_evidence,
        effective_start=effective_start,
        effective_end=effective_end,
        focused=focused,
        transcript_source=transcript_source,
        transcript_segments=transcript_segments,
    )

    print()
    print("## Frames")
    print()
    if frames:
        print(f"Frames live at: `{work / 'frames'}`")
        if manifest_path:
            print(f"Frame index (filename → timestamp → reason): `{manifest_path}`")
        print()
        print(
            "**Read each frame path below with the Read tool to view the image.** "
            "Frames are in chronological order; `t=MM:SS` is the absolute timestamp in the source video."
        )
        print()
        for frame in frames:
            # A frame that absorbed duplicates stands for everything up to
            # represents_until. Saying so on the line itself is what stops a late
            # last-timestamp being read as "the rest was never looked at".
            until = frame.get("represents_until")
            held = ""
            if isinstance(until, (int, float)) and until > frame["timestamp_seconds"] + 1.0:
                held = f" → {format_time(until)} unchanged"
            print(
                f"- `{frame['path']}` "
                f"(t={format_time(frame['timestamp_seconds'])}{held}, "
                f"reason={frame.get('reason', 'selected')})"
            )
    else:
        print("_No frames extracted._")

    print()
    print("## Transcript")
    print()
    if transcript_text:
        label = transcript_source or "captions"
        if focused:
            print(f"_Source: {label}. Filtered to {format_time(effective_start)} → {format_time(effective_end)}:_")
        else:
            print(f"_Source: {label}._")
        print()
        print("```")
        print(transcript_text)
        print("```")
    elif detail == "transcript":
        print(
            "_No transcript available at transcript detail. Captions were missing and Whisper was "
            "unavailable or failed, so there is no visual fallback here. Re-run with "
            "`--detail balanced` for frames._"
        )
    elif focused and dl.get("subtitle_path"):
        print(f"_No transcript lines fell inside {format_time(effective_start)} → {format_time(effective_end)}._")
    else:
        setup_py = SCRIPT_DIR / "setup.py"
        print(
            "_No transcript available — proceed with frames only. "
            "Captions were missing and the Whisper fallback was unavailable "
            "(no API key set, or `--no-whisper` was used). "
            f"Run `python3 {setup_py}` to enable Whisper, then re-run._"
        )

    print()
    print("---")
    print(f"_Work dir: `{work}` — delete when done._")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
