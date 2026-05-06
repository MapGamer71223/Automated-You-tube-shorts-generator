"""
part3.py  ── UPGRADED render pipeline

Architecture
------------
1. Load script.json  → list of lines (one sentence per element)
2. TTS per line      → individual audio files + Whisper word timings
3. Merge timings     → flat word list with EXACT global timestamps
4. Clip selection    → 1 clip per line; duration == actual TTS duration (TTS is master)
5. FFmpeg render     → clips trimmed to audio length → concat → bg.mp4
6. ASS captions      → 1-word highlight captions, pixel-perfect Whisper sync, no drift
7. Final composite   → bg.mp4 + voice + music → final_masterpiece.mp4

Key fixes applied
-----------------
✓ TTS IS THE MASTER TIMELINE — clip duration = speech duration, never reversed
✓ No -t flag truncating audio in padded audio step
✓ Audio is ONLY padded with silence if clip > speech; never cut
✓ Caption offsets use AUDIO cumulative sum (not clamped clip durations)
✓ No artificial +0.1 delay on caption chunk end
✓ 1-word captions, important words get natural Whisper-measured duration
✓ Debug log: clip+timestamp, audio vs clip duration, final video vs audio duration
✓ No NameError / None crashes — all vars initialised before branches
"""

import asyncio
import json
import logging
import os
import random
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("ShortsPipeline")

from tts_engine import generate_tts, WordTiming
from clip_engine import select_clips_for_lines, clamp_duration
import settings as cfg


# ──────────────────────────────────────────────────────────────────────────────
# Directory setup
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = cfg.OUTPUT_DIR
ASSETS_DIR  = getattr(cfg, "MUSIC_DIR", "assets")
SCRIPT_PATH = os.path.join(OUTPUT_DIR, "script.json")


# ──────────────────────────────────────────────────────────────────────────────
# FFprobe helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_duration(file_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get duration for {file_path}: {e}")
        return cfg.CLIP_TARGET_DUR


def is_vertical(video_path: str) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path],
        stdout=subprocess.PIPE, text=True
    )
    try:
        w, h = map(int, result.stdout.strip().split(","))
        return h > w
    except Exception:
        return False


def ffmpeg_sub_path(path: str) -> str:
    path = os.path.abspath(path)
    path = path.replace("\\", "/")
    path = path.replace(":", "\\:")
    return f"'{path}'"


# ──────────────────────────────────────────────────────────────────────────────
# NVENC / CPU encoder selection
# ──────────────────────────────────────────────────────────────────────────────

def _vcodec_args(cq: int = None, preset: str = None):
    cq     = cq     or cfg.FFMPEG_CRF
    preset = preset or cfg.FFMPEG_PRESET
    if getattr(cfg, "USE_NVENC", True):
        return ["-c:v", "h264_nvenc", "-preset", preset, "-cq", str(cq)]
    else:
        return ["-c:v", "libx264", "-preset", "fast", "-crf", str(cq)]


# ──────────────────────────────────────────────────────────────────────────────
# Per-line timing extraction
# ──────────────────────────────────────────────────────────────────────────────

def _line_duration_from_timings(timings: list[WordTiming]) -> float:
    """Actual spoken duration for a line. Returns 0.0 if timings empty."""
    if not timings:
        return 0.0
    return timings[-1].end_sec - timings[0].start_sec


def _merge_timings_with_offset(
    all_timings: list[list[WordTiming]],
    line_audio_offsets: list[float],
) -> list[dict]:
    """
    Flatten per-line WordTiming lists into one global word list.
    Offsets are based on ACTUAL audio file durations (TTS is master).
    Returns list of {"word": str, "start": float, "duration": float}.
    """
    merged: list[dict] = []
    for timings, offset in zip(all_timings, line_audio_offsets):
        for wt in timings:
            merged.append({
                "word":     wt.word,
                "start":    round(wt.start_sec + offset, 3),
                "duration": round(wt.end_sec - wt.start_sec, 3),
            })
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# ASS caption engine — 1-word, exact Whisper sync, no drift
# ──────────────────────────────────────────────────────────────────────────────

# Words that should stay slightly longer on screen (important / emphasis)
_EMPHASIS_WORDS = {
    "never", "always", "every", "most", "least", "only", "just", "now",
    "secret", "truth", "fact", "real", "actually", "literally", "insane",
    "shocking", "warning", "stop", "wait", "listen", "remember", "believe",
}


def _word_display_duration(word: dict, next_word: dict | None) -> float:
    """
    Duration for which a single word is shown on screen.
    Follows Whisper timing exactly:
      • if next word exists: show until next word's start
      • otherwise: show for Whisper-measured duration
    No artificial +0.1 s buffers.
    """
    if next_word is not None:
        return next_word["start"] - word["start"]
    return word["duration"]


def generate_chunked_ass(words: list[dict], out_path: str) -> None:
    """
    Write an ASS subtitle file with 1-word captions (cfg.SUB_WORDS_PER_LINE).

    Sync rules:
      • Each word appears at its exact Whisper start time.
      • Word disappears when the NEXT word starts (tight, no gap).
      • Last word disappears after its own Whisper duration.
      • NO artificial +0.1 delay. NO overlap.
    """
    highlight = cfg.SUB_HIGHLIGHT_COLOR
    font_name = cfg.SUB_FONT_NAME
    font_size = cfg.SUB_FONT_SIZE
    margin_v  = cfg.SUB_MARGIN_V
    outline   = cfg.SUB_OUTLINE
    wpk       = cfg.SUB_WORDS_PER_LINE   # 1 is the default and recommended

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Shadow, Bold, BorderStyle, Outline, Alignment, MarginV
Style: Caption,{font_name},{font_size},{cfg.SUB_PRIMARY_COLOR},{cfg.SUB_OUTLINE_COLOR},0,-1,1,{outline},2,{margin_v}

[Events]
Format: Layer, Start, End, Style, Text
"""

    def fmt(t: float) -> str:
        t = max(0.0, t)
        h  = int(t // 3600)
        m  = int((t % 3600) // 60)
        s  = t % 60
        cs = int(round((s - int(s)) * 100))
        cs = min(cs, 99)
        return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"

    events = ""
    i = 0

    while i < len(words):
        chunk = words[i: i + wpk]

        for j, w in enumerate(chunk):
            word_start = w["start"]

            # End time: start of next word (in chunk or globally), no buffer
            if j + 1 < len(chunk):
                # next word in same chunk
                word_end = chunk[j + 1]["start"]
            elif i + wpk < len(words):
                # first word of next chunk
                word_end = words[i + wpk]["start"]
            else:
                # last word of entire video
                word_end = word_start + w["duration"]

            # Skip degenerate timing
            if word_end <= word_start:
                word_end = word_start + max(0.1, w["duration"])

            # Emphasis: important words get 10% longer display
            clean_word = w["word"].strip(".,!?;:'\"").lower()
            if clean_word in _EMPHASIS_WORDS and next(
                (words[k] for k in range(i + wpk, min(i + wpk + 1, len(words)))), None
            ) is not None:
                # Slight extension only when not the last word (can't steal next word's time)
                pass  # Whisper timing is already correct; don't artificially inflate

            # Build styled line — highlight current word in color
            chunk_words = [cw["word"] for cw in chunk]
            styled_parts = []
            for k, cw in enumerate(chunk_words):
                important_words = {
                    "no", "never", "only", "just", "all", "because",
                    "cleanest", "trash", "home", "yours"
                }

                clean = cw.strip(".,!?").lower()

                if k == j and clean in important_words:
                    color_tag = highlight if highlight.endswith("&") else highlight + "&"
                    styled_parts.append(
                        f"{{\\c{color_tag}\\b1}}{cw}{{\\c{cfg.SUB_PRIMARY_COLOR}&\\b0}}"
                    )
                else:
                    styled_parts.append(cw)

            line_text = "{\\an2}" + " ".join(styled_parts)
            events += f"Dialogue: 0,{fmt(word_start)},{fmt(word_end)},Caption,{line_text}\n"

        i += wpk

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + events)

    logger.info(f"ASS captions written: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Clip processing (FFmpeg per-clip)
# ──────────────────────────────────────────────────────────────────────────────

def _encode_clip(
    clip_path: str,
    start_sec: float,
    duration: float,
    out_path: str,
    index: int,
) -> None:
    """
    Trim, resize, colour-grade, encode one clip segment.
    Resolution: 720×1280 (9:16 vertical), FPS: 30.
    duration == actual TTS speech duration (TTS is master).
    """
    if clip_path == "__blank__":
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s=720x1280:r=30:d={duration}",
            *_vcodec_args(cq=18),
            out_path,
        ]
    else:
        vf = (
            "scale=720:1280:force_original_aspect_ratio=increase,"
            "crop=720:1280,"
            "fps=30,"
            "eq=contrast=1.08:saturation=1.15:brightness=0.02"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-t",  str(duration),   # clip duration == TTS speech duration
            "-i",  clip_path,
            "-an",
            "-vf", vf,
            *_vcodec_args(),
            out_path,
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        actual = get_duration(out_path)
        logger.debug(f"  Clip {index:02d}: requested={duration:.2f}s | encoded={actual:.2f}s")
    except subprocess.CalledProcessError as e:
        logger.error(f"Clip {index} encode failed: {e.stderr.decode()[-300:]}")
        raise
def build_word_groups(words, max_gap=0.4, max_words=5):
    groups = []
    current = [words[0]]

    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i-1]["start"]
        word = words[i]["word"].lower()

        if (
            gap > max_gap
            or len(current) >= max_words
            or word in ["and", "but", "because", "so"]
        ):
            groups.append(current)
            current = []

        current.append(words[i])

    if current:
        groups.append(current)

    return groups
#──────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

async def make_video() -> None:
    # ── 0. Load script ─────────────────────────────────────────────────────────
    if not os.path.exists(SCRIPT_PATH):
        logger.error(f"script.json not found at {SCRIPT_PATH}")
        return

    with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines: list[str] = data.get("script") or []
    if not lines:
        logger.error("script.json has no 'script' lines")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. TTS per line ────────────────────────────────────────────────────────
    logger.info(f"Generating TTS for {len(lines)} lines…")


    full_text = " ".join(lines)
    full_audio_path, full_timings = await generate_tts(
    full_text,
    output_path=os.path.join(OUTPUT_DIR, "full_audio.mp3")
)
    global_words = [
    {
        "word": wt.word,
        "start": wt.start_sec,
        "duration": wt.end_sec - wt.start_sec
    }
    for wt in full_timings
]
    logger.info(f"TTS audio generated: {full_audio_path} | duration={get_duration(full_audio_path):.2f}s")

    # ── 2. Compute per-line ACTUAL audio durations (TTS is master) ─────────────


    # ── 3. Clip selection (TTS-driven durations) ──────────────────────────────
    logger.info("Selecting clips (CLIP embeddings, best-segment detection)…")
    groups = build_word_groups(global_words)

    texts = []
    durations = []

    for i, g in enumerate(groups):
        text = " ".join([w["word"] for w in g])
        start = g[0]["start"]
        end = g[-1]["start"] + g[-1]["duration"]

        if i + 1 < len(groups):
            next_start = groups[i+1][0]["start"]
            dur = next_start - start
        else:
            dur = end - start

        dur = max(0.3, dur)  # stability fix

        texts.append(text)
        durations.append(dur)
    # 🔥 NOW call clip engine
    segments = select_clips_for_lines(texts, durations)

    expanded_segments = segments
    # ── 4. Encode clip segments ────────────────────────────────────────────────
    processed_clips = []

    for i, (clip_path, start_sec, clip_dur) in enumerate(expanded_segments):
        out = os.path.join(OUTPUT_DIR, f"clip_{i}.mp4")

        logger.info(
            f"  Clip {i:02d}: path={os.path.basename(clip_path)} "
            f"| start={start_sec:.2f}s | dur={clip_dur:.2f}s"
        )

        _encode_clip(clip_path, start_sec, clip_dur, out, index=i)
        processed_clips.append(out)

    # ── 5. Concatenate clips → bg.mp4 ─────────────────────────────────────────
    concat_file = os.path.join(OUTPUT_DIR, "concat.txt")
    with open(concat_file, "w") as f:
        for p in processed_clips:
            abs_p = os.path.abspath(p).replace("\\", "/")
            f.write(f"file '{abs_p}'\n")

    temp_video = os.path.join(OUTPUT_DIR, "bg.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_file, "-c", "copy", temp_video],
        check=True
    )
    video_dur = get_duration(temp_video)
    logger.info(f"Background video: {temp_video} | duration={video_dur:.2f}s")

    # ── 6. Build padded audio — NEVER cut speech, only pad silence ────────────
    #
    # Rule: if clip_dur > audio_dur → pad with silence
    #       if clip_dur < audio_dur → DO NOT trim (this should never happen
    #                                 since clip_dur = speech_dur + 0.05)
    #
    full_voice = full_audio_path
    audio_total_dur = get_duration(full_voice)

    logger.info(f"Full voice audio: {full_voice} | duration={audio_total_dur:.2f}s")

    # ── 7. Build global word timings — offsets must follow VIDEO timeline ─────
    #
    # CRITICAL: caption offsets must match where each audio chunk sits in the
    # FINAL video, which is determined by clip_dur (not raw audio file dur).
    # clip_dur = speech_dur + 0.05 tail-pad → use that as the segment clock.
    #
   
    # shift global words based on clip timeline
    shifted_words = []

    for word in global_words:
        shifted_words.append({
            "word": word["word"],
            "start": round(word["start"], 3),   # already global from TTS
            "duration": word["duration"]
        })

    if not global_words:
        logger.warning("No word timings — captions will be empty")

    # ── 8. Generate ASS captions ───────────────────────────────────────────────
    ass_path = os.path.join(OUTPUT_DIR, "final.ass")
    if global_words:
        generate_chunked_ass(shifted_words, ass_path)
    else:
        with open(ass_path, "w") as f:
            f.write("[Script Info]\nScriptType: v4.00+\n[Events]\nFormat: Layer, Start, End, Style, Text\n")

    # ── 9. Pick background music ───────────────────────────────────────────────
    audio_exts = {".mp3", ".wav", ".ogg", ".m4a"}
    music_files = [
        os.path.join(ASSETS_DIR, f) for f in os.listdir(ASSETS_DIR)
        if os.path.splitext(f)[1].lower() in audio_exts
    ] if os.path.isdir(ASSETS_DIR) else []

    music_path = random.choice(music_files) if music_files else None

    # ── 10. Final render ───────────────────────────────────────────────────────
    final_video = os.path.join(OUTPUT_DIR, "final_masterpiece.mp4")
    safe_ass    = ffmpeg_sub_path(os.path.abspath(ass_path))
    vf_chain    = f"subtitles={safe_ass}" if global_words else "null"

    if music_path:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", full_voice,
            "-i", music_path,
            "-filter_complex",
            "[1:a]loudnorm=I=-16:LRA=7:TP=-1.5[voice];"
            f"[2:a]volume=0.6,afade=t=in:st=0:d=0.4,afade=t=out:st={max(1.0, audio_total_dur - 1.5):.2f}:d=1.5[music];"
            "[voice][music]amix=inputs=2:duration=shortest:weights=1 0.6[a]",
            "-map", "0:v",
            "-map", "[a]",
            "-vf", vf_chain,
            *_vcodec_args(cq=15),
            "-c:a", "aac",
            final_video,
        ], check=True)
    else:
        logger.warning("No music file found — rendering without background music")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", full_voice,
            "-filter_complex", "[1:a]loudnorm=I=-16:LRA=7:TP=-1.5[a]",
            "-map", "0:v",
            "-map", "[a]",
            "-vf", vf_chain,
            *_vcodec_args(cq=15),
            "-c:a", "aac",
            final_video,
        ], check=True)

    final_dur = get_duration(final_video)
    logger.info(f"✅ Done → {final_video}")
    logger.info(
        f"Final check: video={video_dur:.2f}s | audio={audio_total_dur:.2f}s "
        f"| final={final_dur:.2f}s | delta={(final_dur - audio_total_dur):+.2f}s"
    )

    # ── FINAL SYNC VALIDATION ─────────────────────────────────────────────────
    drift = abs(final_dur - audio_total_dur)
    logger.info("=" * 60)
    logger.info(f"   Background video : {video_dur:.3f}s")
    logger.info(f"   Full voice audio : {audio_total_dur:.3f}s")
    logger.info(f"   Final output     : {final_dur:.3f}s")
    logger.info(f"   A/V drift        : {drift:+.3f}s {'✓ OK' if drift < 0.2 else '⚠ DRIFT DETECTED — check caption offsets'}")
    logger.info("=" * 60)

    _print_summary(texts, segments, durations, audio_total_dur)

# ──────────────────────────────────────────────────────────────────────────────
# Debug summary
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(
    lines: list[str],
    segments: list,
    speech_durations: list[float],
    total_audio: float,
) -> None:
    # Guard: ensure segments list matches lines length
    if not segments:
        logger.warning("_print_summary: no segments to summarise")
        return
    if len(segments) != len(lines):
        logger.warning(
            f"_print_summary: len(segments)={len(segments)} != len(lines)={len(lines)} — truncating"
        )
        n = min(len(segments), len(lines), len(speech_durations))
        segments         = segments[:n]
        speech_durations = speech_durations[:n]
        lines            = lines[:n]

    print("\n" + "─" * 80)
    print(f"{'LINE':<4} {'SPEECH':>7} {'CLIP_SEG':>8} {'DELTA':>7}  {'START':>6}  TEXT")
    print("─" * 80)

    total_clips = 0.0
    for i, (line, sd, seg) in enumerate(zip(lines, speech_durations, segments)):
        clip_path, start_sec, clip_dur = seg
        delta     = clip_dur - sd
        flag      = "⚠ " if abs(delta) > 0.5 else "  "
        clip_name = os.path.basename(clip_path)[:20]
        print(
            f"{i:<4} {sd:>7.2f}s {clip_dur:>8.2f}s {delta:>+7.2f}s  "
            f"{start_sec:>5.2f}s  {flag}{line[:40]}"
        )
        total_clips += clip_dur

    print("─" * 80)
    print(f"     {sum(speech_durations):>7.2f}s {total_clips:>8.2f}s           TOTAL SPEECH / TOTAL CLIPS")
    print(f"     Audio file concat: {total_audio:.2f}s")
    delta_av = total_clips - total_audio
    ok = "✓" if abs(delta_av) < 0.2 else "⚠ DRIFT DETECTED"
    print(f"     Audio vs clips: {delta_av:+.2f}s  {ok}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(make_video())