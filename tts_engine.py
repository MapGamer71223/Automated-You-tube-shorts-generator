"""
modules/tts_engine.py  ── OPTIMIZED for Tortoise TTS + Whisper alignment

Timing resolution priority
--------------------------
1. Whisper forced alignment on the generated audio  (always used when possible)
2. Synthetic proportional timings                   (fallback only)

Tortoise TTS produces slower, more expressive speech than Edge TTS.
All clip-duration clamping must happen AFTER timings are known
(i.e., in part3.py / make_video), not here.

This module is a pure audio + timing producer.
"""

from __future__ import annotations
import torch

import asyncio
import json
import os
import subprocess
import tempfile
import torchaudio
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from logger import log_tts, log_error, log_debug
import settings as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data model (unchanged — subtitle engine depends on this shape)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WordTiming:
    word:      str
    start_sec: float   # seconds from audio start
    end_sec:   float   # seconds from audio start


# ─────────────────────────────────────────────────────────────────────────────
# Whisper singleton
# ─────────────────────────────────────────────────────────────────────────────

_whisper_model = None


def _get_whisper_model():
    """Lazy-load faster-whisper once per process. GPU if available."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log_error("faster-whisper not installed — run: pip install faster-whisper")
        return None

    model_size = getattr(cfg, "WHISPER_MODEL_SIZE", "small")

    try:
        import torch
        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "int8"
    except ImportError:
        device, compute_type = "cpu", "int8"

    log_tts(f"Whisper: loading '{model_size}' on {device} ({compute_type})")
    try:
        _whisper_model = WhisperModel(
            model_size, device=device, compute_type=compute_type, num_workers=1
        )
        log_tts(f"Whisper '{model_size}' ready on {device}")
    except Exception as e:
        log_error(f"Whisper load failed: {e}", exc=True)
        _whisper_model = None

    return _whisper_model


# ─────────────────────────────────────────────────────────────────────────────
# Whisper alignment
# ─────────────────────────────────────────────────────────────────────────────

def _run_whisper_sync(audio_path: str) -> List[WordTiming]:
    """Blocking — run via run_in_executor from async callers."""
    model = _get_whisper_model()
    if model is None:
        return []

    t0 = time.perf_counter()
    try:
        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=5,
            language="en",
            vad_filter=False,
            condition_on_previous_text=False,
        )
        timings: List[WordTiming] = []
        for seg in segments:
            if not seg.words:
                continue
            for w in seg.words:
                clean = w.word.strip()
                if not clean:
                    continue
                timings.append(WordTiming(
                    word=clean,
                    start_sec=round(w.start, 3),
                    end_sec=round(w.end, 3),
                ))
        log_tts(
            f"Whisper: {len(timings)} words in {time.perf_counter()-t0:.2f}s "
            f"(lang={info.language}, p={info.language_probability:.2f})"
        )
        return timings
    except Exception as e:
        log_error(f"Whisper transcription failed: {e}", exc=True)
        return []


async def align_with_whisper(audio_path: str) -> List[WordTiming]:
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        log_error(f"align_with_whisper: file missing or empty — {audio_path}")
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_whisper_sync, audio_path)


# ─────────────────────────────────────────────────────────────────────────────
# Tortoise TTS
# ─────────────────────────────────────────────────────────────────────────────

def _tortoise_generate_sync(text: str, voice: str, preset: str, output_path: str) -> bool:
    """
    Blocking call to Tortoise TTS.
    Returns True on success, False on failure.

    Tortoise is slow (5–30 s per sentence at ultra_fast).
    For short-form video, use preset='ultra_fast' or 'fast'.
    GPU strongly recommended — CPU is 5–10× slower.

    Requires: pip install tortoise-tts
    """
    try:
        import torch
        from tortoise.api import TextToSpeech

        # Singleton TTS object (avoid reloading weights per sentence)
        if not hasattr(_tortoise_generate_sync, "_tts"):
            log_tts("Loading Tortoise TTS model (first call — may take 30-60 s)…")
            _tortoise_generate_sync._tts = TextToSpeech(
                use_deepspeed=False,   # set True if DeepSpeed installed
                kv_cache=True,
                half=torch.cuda.is_available(),
            )
            log_tts("Tortoise TTS model loaded")

        tts = _tortoise_generate_sync._tts

        log_tts(f"Tortoise TTS: preset={preset}, voice={voice}, text={text[:60]!r}…")
        t0 = time.perf_counter()
        seed_val = getattr(cfg, "seed", 1234)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)

        gen = tts.tts_with_preset(
    text,
    voice_samples=None,
    conditioning_latents=None,
    preset=preset,
)
        # gen is a tensor; save as WAV then convert to MP3
        tmp_wav = output_path.replace(".mp3", "_raw.wav")
        torchaudio.save(tmp_wav, gen.squeeze(0).cpu(), 24000)

        # Convert WAV → MP3 via ffmpeg (keeps pipeline MP3-native)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_wav, "-q:a", "2", output_path],
            capture_output=True, check=True, timeout=30
        )
        os.remove(tmp_wav)

        elapsed = time.perf_counter() - t0
        log_tts(f"Tortoise TTS done in {elapsed:.1f}s → {output_path}")
        return True

    except ImportError as e:
        log_error(f"Tortoise import failed: {e}")
        return False

    except Exception as e:
        import traceback
        traceback.print_exc()
        log_error(f"Tortoise runtime error: {e}", exc=True)
        return False
    except Exception as e:
        log_error(f"Tortoise TTS failed: {e}", exc=True)
        import traceback
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Edge TTS (kept as fallback)
# ─────────────────────────────────────────────────────────────────────────────

async def _edge_tts_generate(text: str, voice: str, rate: str, volume: str, output_path: str) -> bool:
    """Run Edge TTS. Returns True on success."""
    try:
        import edge_tts
    except ImportError:
        log_error("edge_tts not installed — run: pip install edge-tts")
        return False
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
        chunks = []
        async for event in communicate.stream():
            if event["type"] == "audio":
                chunks.append(event["data"])
        if not chunks:
            log_error("Edge TTS returned no audio chunks")
            return False
        with open(output_path, "wb") as f:
            for c in chunks:
                f.write(c)
        log_tts(f"Edge TTS audio written: {output_path}")
        return True
    except Exception as e:
        log_error(f"Edge TTS failed: {e}", exc=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic timing fallback
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_timings(text: str, audio_duration: float) -> List[WordTiming]:
    """
    Proportional character-length word timings.
    Used ONLY when Whisper fails — never the primary path.

    For Tortoise audio, inflate estimated duration slightly since
    Tortoise speaks slower than the neutral TTS baseline.
    """
    words = text.split()
    if not words:
        return []
    lengths = [max(1, len(w)) for w in words]
    total   = sum(lengths)
    timings = []
    cursor  = 0.05
    for w, L in zip(words, lengths):
        dur = max(0.06, (L / total) * (audio_duration - 0.1))
        timings.append(WordTiming(
            word=w,
            start_sec=round(cursor, 3),
            end_sec=round(cursor + dur, 3),
        ))
        cursor += dur
    log_tts(f"Synthetic timings for {len(words)} words (fallback)")
    return timings


# ─────────────────────────────────────────────────────────────────────────────
# Audio duration util
# ─────────────────────────────────────────────────────────────────────────────

def _get_audio_duration(path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
            capture_output=True, text=True, timeout=10
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:
        return 5.0


def _write_silence(path: str, duration: float) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", "anullsrc=r=44100:cl=stereo",
             "-t", str(duration),
             "-q:a", "9", "-acodec", "libmp3lame", path],
            capture_output=True, timeout=30
        )
    except Exception as e:
        log_error(f"Could not write silence: {e}")
        with open(path, "wb") as f:
            f.write(b"\xff\xfb" + b"\x00" * 413)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def generate_tts(
    text: str,
    output_path: Optional[str] = None,
    voice: Optional[str]  = None,
    rate:  Optional[str]  = None,
    volume: Optional[str] = None,
) -> Tuple[str, List[WordTiming]]:
    """
    Generate TTS audio and return (audio_path, word_timings).

    Engine selection (cfg.TTS_ENGINE):
      "tortoise" → Tortoise TTS (default; high quality, GPU recommended)
      "edge"     → Microsoft Edge TTS (fast, internet required)

    Timing pipeline:
      1. Generate audio via selected engine
      2. Align with Whisper (word-level accuracy against real waveform)
      3. Fall back to synthetic timings only if Whisper fails

    Never raises — degrades through fallback chain.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

    engine = getattr(cfg, "TTS_ENGINE", "edge").lower()
    voice  = voice  or (cfg.TTS_VOICE if engine == "tortoise" else cfg.TTS_EDGE_VOICE)
    rate   = rate   or cfg.TTS_RATE
    volume = volume or cfg.TTS_VOLUME

    log_tts(f"TTS engine={engine}, voice={voice}, text={text[:60]!r}…")

    # ── Step 1: Generate audio ────────────────────────────────────────────────
    audio_ok = False

    if engine == "tortoise":
        preset = getattr(cfg, "TTS_TORTOISE_PRESET", "ultra_fast")
        loop   = asyncio.get_running_loop()
        audio_ok = await loop.run_in_executor(
            None, _tortoise_generate_sync, text, voice, preset, output_path
        )
        if not audio_ok:
            log_tts("Tortoise failed — falling back to Edge TTS", level="warning")
            audio_ok = await _edge_tts_generate(text, cfg.TTS_EDGE_VOICE, rate, volume, output_path)

    else:  # edge
        audio_ok = await _edge_tts_generate(text, voice, rate, volume, output_path)

    # ── Step 2: Whisper alignment (always preferred over boundary events) ─────
    if audio_ok and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        whisper_timings = await align_with_whisper(output_path)
        if whisper_timings:
            log_tts(f"Using Whisper timings: {len(whisper_timings)} words")
            return output_path, whisper_timings
        log_tts("Whisper returned no timings — using synthetic fallback", level="warning")
        duration = _get_audio_duration(output_path)
        return output_path, _synthetic_timings(text, duration)

    # ── Step 3: Full fallback — silence + synthetic ───────────────────────────
    log_tts("No audio produced — writing silence", level="warning")
    words    = text.split()
    # Tortoise-like rate: ~130 wpm → ~0.46 s/word average
    duration = max(3.0, len(words) * 0.46)
    _write_silence(output_path, duration)
    return output_path, _synthetic_timings(text, duration)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk helpers (used by subtitle engine)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_timings(
    timings: List[WordTiming],
    words_per_chunk: int = None,
) -> List[List[WordTiming]]:
    """
    Group word timings into subtitle display chunks.
    cfg.SUB_WORDS_PER_LINE = 2  (1–2 words MAX for Short-form retention)
    """
    wpk = words_per_chunk or cfg.SUB_WORDS_PER_LINE
    return [timings[i: i + wpk] for i in range(0, len(timings), wpk)]