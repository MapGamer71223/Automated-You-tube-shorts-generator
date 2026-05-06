"""
config/settings.py  ── OPTIMIZED for Tortoise TTS + strict pacing
Central configuration for the YouTube Shorts pipeline.
Edit this file to tune behaviour without touching module code.
"""

# ─── LLM ──────────────────────────────────────────────────────────────────────
LLM_BASE_URL    = "http://localhost:1234/v1"
LLM_MODEL       = "mistral"
LLM_TIMEOUT     = 30
LLM_MAX_TOKENS  = 200
LLM_TEMPERATURE = 0.85

SCRIPT_MIN_WORDS  = 8
SCRIPT_MAX_WORDS  = 35
SCRIPT_IDEAL_MIN  = 12
SCRIPT_IDEAL_MAX  = 28
SCRIPT_BATCH_COUNT = 5
SCRIPT_FALLBACKS  = [
    "Did you know the average person checks their phone 96 times a day? That's once every ten minutes.",
    "Scientists discovered that a day on Venus is longer than its year. Space is wild.",
    "The first video ever uploaded to YouTube was just 18 seconds of someone at the zoo.",
    "Your brain uses 20 percent of your body's energy despite being only 2 percent of its weight.",
    "The shortest war in history lasted 38 minutes between Britain and Zanzibar in 1896.",
]

# ─── TTS ── TORTOISE ──────────────────────────────────────────────────────────
TTS_ENGINE         = "edge"          # "tortoise" | "edge"
TTS_VOICE          = "emma"            # Tortoise preset: random/fast/daniel/emma …
TTS_TORTOISE_PRESET = "fast"       # ultra_fast / fast / standard / high_quality
TTS_EDGE_VOICE     = "en-US-JennyNeural" # fallback if Tortoise unavailable
TTS_RATE           = "+12%"               # Tortoise ignores this; kept for Edge fallback
TTS_VOLUME         = "+0%"
# inside your tts engine
seed = 1234  # Set a fixed seed for reproducibility. Change this value to get different random voices each time.
# Tortoise speaks ~20% slower than Edge TTS.
# These multipliers compensate clip duration accordingly.
TTS_SPEED_RATIO    = 0.82                # tortoise is ~82% the speed of Edge at +10%

# ─── PACING ── THE CORE RULE ──────────────────────────────────────────────────
# One clip per script line.  Duration = TTS speech time, clamped to this window.
CLIP_MIN_DUR   = 1.6    # seconds — never shorter (looks choppy)
CLIP_MAX_DUR   = 3.0    # seconds — never longer  (loses retention)
CLIP_TARGET_DUR = 2.0   # fallback when speech timing unavailable

# ─── STRUCTURE WINDOW (seconds from video start) ─────────────────────────────
HOOK_WINDOW_END   = 2.0    # 0–2 s   → hook
BUILD_WINDOW_END  = 7.0   # 2–7 s  → build curiosity
TENSION_WINDOW_END = 12.0  # 7–12 s → tension
REVEAL_WINDOW_END  = 17.0  # 12–17 s → reveal / payoff

# ─── SUBTITLES ────────────────────────────────────────────────────────────────
SUB_FONT_NAME      = "Anton"          # 🔥 MAIN FIX
SUB_FONT_SIZE      = 120
SUB_PRIMARY_COLOR  = "&H00FFFFFF&"    # white
SUB_HIGHLIGHT_COLOR = "&H0000FFFF&"   # 🔥 yellow (viral style)
SUB_OUTLINE_COLOR  = "&H00000000"
SUB_BACK_COLOR     = "&H80000000"
SUB_OUTLINE        = 10
SUB_SHADOW         = 1
SUB_ALIGNMENT      = 2              # bottom-centre (ASS code)
SUB_MARGIN_V       = 350
SUB_WORDS_PER_LINE =1               # 1–2 words at a time MAX

# ─── CLIP ENGINE ──────────────────────────────────────────────────────────────
CLIP_DIR          = "clips"
import os as _os
CLIP_CACHE_DIR    = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".clip_cache")
del _os
DEFAULT_CLIP_DIR  = "assets/default_clips"
CLIP_USE_SEMANTIC = True            # False → keyword-only

# Keyword→category map for hybrid clip selection
CLIP_KEYWORD_MAP = {
    "indoor":  ["store", "restaurant", "cafe", "office", "room", "house", "inside"],
    "outdoor": ["park", "street", "city", "road", "outside", "nature", "forest"],
    "calm":    ["quiet", "alone", "peace", "silent", "walk", "slow"],
    "reaction":["confusing", "strange", "weird", "shocked", "surprised", "wait"],
    "action":  ["run", "fast", "chase", "fight", "move", "hurry", "rush"],
    "face":    ["person", "people", "man", "woman", "face", "eye", "look"],
    "data":    ["percent", "%", "million", "billion", "number", "study", "research"],
}

# ─── HOOK CLIP PREFERENCES ───────────────────────────────────────────────────
HOOK_PREFER_TAGS   = ["face", "motion", "close-up", "reaction"]
HOOK_PENALIZE_TAGS = ["landscape", "text", "static"]

# ─── VIDEO / FFMPEG ──────────────────────────────────────────────────────────
OUTPUT_WIDTH   = 720
OUTPUT_HEIGHT  = 1280
OUTPUT_FPS     = 30
OUTPUT_DIR     = "output"
FFMPEG_PRESET  = "p6"       # NVENC preset (GPU). Use "fast" for CPU builds.
FFMPEG_CRF     = 14         # Quality (lower = better). NVENC uses -cq instead.
FFMPEG_THREADS = 2
USE_NVENC      = True       # Set False to fall back to libx264 (CPU)

ENABLE_ZOOM    = False
MUSIC_DIR      = "assets"
MUSIC_VOLUME   = 0.3

# ─── WHISPER ─────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "small"  # tiny/base/small/medium — small is best RTX 3050 tradeoff

# ─── QUALITY PRESETS ─────────────────────────────────────────────────────────
QUALITY_PRESETS = {
    "fast":     {"FFMPEG_PRESET": "p3", "FFMPEG_CRF": 24, "CLIP_USE_SEMANTIC": False},
    "balanced": {"FFMPEG_PRESET": "p5", "FFMPEG_CRF": 18, "CLIP_USE_SEMANTIC": True},
    "high":     {"FFMPEG_PRESET": "p6", "FFMPEG_CRF": 15, "CLIP_USE_SEMANTIC": True},
}

# ─── LOGGING ─────────────────────────────────────────────────────────────────
LOG_DIR     = "logs"
LOG_LEVEL   = "DEBUG"
LOG_TO_FILE = True
DEBUG_MODE  = False


def apply_preset(preset_name: str) -> None:
    import sys
    mod = sys.modules[__name__]
    preset = QUALITY_PRESETS.get(preset_name, QUALITY_PRESETS["balanced"])
    for k, v in preset.items():
        setattr(mod, k, v)