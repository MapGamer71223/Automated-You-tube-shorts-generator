"""
clip_engine.py  ── UPGRADED: FAISS moment-level retrieval + best-segment detection

Changes from previous version
------------------------------
1. FAISS global moment search replaces per-clip embedding ranking
   → select_clips_for_lines() now queries ALL indexed frames via FAISS
   → returns (clip_path, start_time, duration) with start_time from metadata
2. Category preference scoring: hook > face > motion > reaction > luxury > neutral
3. Combined ranking: FAISS similarity + category score + motion energy
4. Safe fallback chain: FAISS → per-clip embedding → stem scoring
5. Singleton FAISS index — loaded once at startup
6. All original public API preserved (select_clips, select_clips_for_lines, etc.)
7. TTS-driven durations unchanged — audio is still master

Imports build_index for FAISS retrieval. Run build_index.py first.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from logger import log_clip, log_error, log_debug
import settings as cfg

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

ClipSegment = Tuple[str, float, float]   # (path, start_sec, duration_sec)
_SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ── Category preference order (higher index = lower preference) ───────────────
# Used to break ties when FAISS similarity is close
CATEGORY_RANK: Dict[str, int] = {
    "hook":     6,
    "face":     5,
    "motion":   4,
    "reaction": 3,
    "luxury":   2,
    "neutral":  1,
    "unknown":  0,
}

# Index directory (can be overridden via settings)
_INDEX_DIR: str = getattr(cfg, "FAISS_INDEX_DIR", "index")


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index  (singleton — loaded once)
# ─────────────────────────────────────────────────────────────────────────────

_faiss_ready: bool = False


def _ensure_faiss() -> bool:
    """
    Load FAISS index on first call. Returns True if ready.
    Falls back gracefully if build_index is missing or index not built.
    """
    global _faiss_ready
    if _faiss_ready:
        return True
    try:
        from build_index import load_faiss_index
        _faiss_ready = load_faiss_index(_INDEX_DIR)
        if not _faiss_ready:
            log_clip("FAISS index not available — falling back to per-clip embeddings", level="warning")
    except ImportError:
        log_clip("build_index.py not found — FAISS disabled, using per-clip embeddings", level="warning")
        _faiss_ready = False
    return _faiss_ready


def _faiss_search(text: str, top_k: int = 30) -> List[Dict]:
    key = (text.strip().lower(), top_k)
    # 🔥 CACHE HIT
    if key in _faiss_result_cache:
        log_debug(f"FAISS cache hit: '{text[:30]}'")
        return _faiss_result_cache[key]

    try:
        from build_index import search_index
        results = search_index(text, top_k=top_k, index_dir=_INDEX_DIR)

        # 🔥 STORE CACHE
        _faiss_result_cache[key] = results
        return results

    except Exception as e:
        log_error(f"FAISS search error: {e}")
        return []

# ─────────────────────────────────────────────────────────────────────────────
# FFprobe helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_clip_duration(path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", path],
            capture_output=True, text=True, timeout=10
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Clip catalogue  (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

_clip_catalogue_cache: dict[str, list[dict]] = {}


def scan_clips(clip_dir: str = None) -> List[Dict]:
    """
    Walk clip_dir and return metadata dicts:
      { path, filename, stem, duration }
    Skips clips shorter than 2s. Results cached per directory.
    """
    clip_dir = clip_dir or cfg.CLIP_DIR

    if clip_dir in _clip_catalogue_cache:
        log_clip(f"Catalogue cache hit: {len(_clip_catalogue_cache[clip_dir])} clips")
        return _clip_catalogue_cache[clip_dir]

    clips: List[Dict] = []
    for root, _, files in os.walk(clip_dir):
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if ext not in _SUPPORTED_EXTS:
                continue
            path = os.path.join(root, fname)
            dur  = _get_clip_duration(path)
            if dur < 2.0:
                log_clip(f"Skip short clip ({dur:.1f}s): {fname}", level="debug")
                continue
            stem = Path(fname).stem.replace("_", " ").replace("-", " ").lower()
            clips.append({
                "path":     path,
                "filename": fname,
                "stem":     stem,
                "duration": dur,
                # Derive category from parent folder name
                "category": Path(root).name
                            if Path(root).name in CATEGORY_RANK else "unknown",
            })

    if not clips:
        log_clip(f"No clips in '{clip_dir}' — trying default_clips…", level="warning")
        if clip_dir != cfg.DEFAULT_CLIP_DIR:
            clips = scan_clips(cfg.DEFAULT_CLIP_DIR)

    log_clip(f"Catalogue: {len(clips)} clips")
    _clip_catalogue_cache[clip_dir] = clips
    return clips

def _load_recent_used():
    global _RECENT_USED

    if os.path.exists(_RECENT_USED_PATH):
        try:
            with open(_RECENT_USED_PATH, "r") as f:
                _RECENT_USED = json.load(f)
        except Exception:
            _RECENT_USED = {}
            
def _save_recent_used():
    # keep only meaningful recent entries
    cleaned = {
        k: v for k, v in _RECENT_USED.items()
        if v > 0.05
    }

    _RECENT_USED.clear()
    _RECENT_USED.update(cleaned)
    try:
        with open(_RECENT_USED_PATH, "w") as f:
            json.dump(_RECENT_USED, f)
    except Exception:
        pass
# ─────────────────────────────────────────────────────────────────────────────
# CLIP model  (singleton — unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

_clip_model      = None
_clip_preprocess = None
_clip_device     = "cpu"

_RECENT_USED_PATH = os.path.join(_INDEX_DIR, "recent_used.json")
def _load_clip_model() -> bool:
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is not None:
        return True
    try:
        import torch
        import clip as openai_clip
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preprocess = openai_clip.load("ViT-B/32", device=_clip_device)
        log_clip(f"CLIP model loaded on {_clip_device}")
        return True
    except ImportError:
        log_clip("openai-clip not installed — stem scoring fallback", level="warning")
        return False
    except Exception as e:
        log_error(f"CLIP model load failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Frame embedding helpers  (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_frame_at(clip_path: str, timestamp: float) -> Optional[np.ndarray]:
    """Extract one frame at `timestamp` seconds → normalised CLIP image embedding."""
    try:
        import torch
        from PIL import Image

        cap = cv2.VideoCapture(clip_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        inp = _clip_preprocess(img).unsqueeze(0).to(_clip_device)
        with torch.no_grad():
            emb = _clip_model.encode_image(inp).cpu().numpy().flatten()
        emb /= (np.linalg.norm(emb) + 1e-8)
        return emb
    except Exception as e:
        log_error(f"Frame embed failed at {timestamp:.2f}s: {e}")
        return None


def _text_embedding(text: str) -> Optional[np.ndarray]:
    """Return normalised CLIP text embedding."""
    try:
        import torch
        import clip as openai_clip
        tokens = openai_clip.tokenize([text]).to(_clip_device)
        with torch.no_grad():
            emb = _clip_model.encode_text(tokens).cpu().numpy().flatten()
        emb /= (np.linalg.norm(emb) + 1e-8)
        return emb
    except Exception as e:
        log_error(f"Text embed failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Whole-clip embedding cache  (unchanged — used for per-clip fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(clip_path: str) -> str:
    h = hashlib.md5(clip_path.encode()).hexdigest()[:12]
    return os.path.join(cfg.CLIP_CACHE_DIR, f"{h}.npy")


def _get_clip_embedding(clip_path: str) -> Optional[np.ndarray]:
    """Averaged CLIP embedding for whole clip (3-frame sample). Disk-cached."""
    cache = _cache_path(clip_path)
    if os.path.exists(cache):
        try:
            return np.load(cache)
        except Exception:
            pass

    dur    = _get_clip_duration(clip_path)
    embs   = []
    for t in [dur * 0.2, dur * 0.5, dur * 0.8]:
        emb = _embed_frame_at(clip_path, t)
        if emb is not None:
            embs.append(emb)

    if not embs:
        return None

    avg = np.mean(embs, axis=0)
    avg /= (np.linalg.norm(avg) + 1e-8)
    os.makedirs(cfg.CLIP_CACHE_DIR, exist_ok=True)
    np.save(cache, avg)
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: per-clip semantic scoring  (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def _semantic_scores(script_line: str, clips: List[Dict]) -> List[float]:
    """Rank clips by cosine sim between script line and clip image embedding."""
    if not _load_clip_model():
        line_words = set(script_line.lower().split())
        return [float(len(line_words & set(c["stem"].split()))) for c in clips]

    text_emb = _text_embedding(script_line)
    if text_emb is None:
        return [0.0] * len(clips)

    scores = []
    for clip in clips:
        emb = _get_clip_embedding(clip["path"])
        if emb is None:
            scores.append(0.0)
            continue
        scores.append(float(np.dot(text_emb, emb)))
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Motion score  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _motion_score_at(clip_path: str, timestamp: float, window: float = 0.3) -> float:
    """Estimate motion energy around `timestamp`. Returns 0.0–1.0."""
    try:
        cap    = cv2.VideoCapture(clip_path)
        frames = []
        for t in [max(0, timestamp - window / 2), timestamp + window / 2]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if ok and fr is not None:
                small = cv2.resize(fr, (64, 64))
                gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                frames.append(gray)
        cap.release()
        if len(frames) < 2:
            return 0.0
        diff = np.mean(np.abs(frames[1] - frames[0]))
        return float(min(1.0, diff / 50.0))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADED: FAISS-based moment selection
# ─────────────────────────────────────────────────────────────────────────────
_motion_cache = {}
_faiss_result_cache = {}
_RECENT_USED = {}
def _score_faiss_result(
    result:           Dict,
    segment_dur:      float,
    used_paths:       Set[str],
    prefer_unused:    bool = True,
    speech_duration:  float = 0.0,
    prev_category:    str = "",
    script_embedding: Optional[np.ndarray] = None,
) -> float:
    """
    Compute composite score for a FAISS frame result:
      = 0.45 * faiss_similarity   (CLIP cosine score, 0–1)
      + 0.25 * text_similarity    (caption_hint vs script, 0–1)
      + 0.15 * motion_score       (0–1)
      + 0.15 * category_score     (normalised 0–1, duration-aware boost)
      - reuse_penalty             (0.30 if already used, else 0)
      + continuity_bonus          (0.05 if same category as previous)
    """
    clip_path = result["clip_path"]
    timestamp = result["timestamp"]
    sim_score = result.get("score", 0.0)
    category  = result.get("category", "unknown")

    # ── Normalised category score (0–1) ──────────────────────────────────────
    max_rank   = max(CATEGORY_RANK.values(), default=1)
    cat_score  = CATEGORY_RANK.get(category, 0) / max_rank   # already 0–1

    # ── Duration-aware category boost ────────────────────────────────────────
    dur_multiplier = 1.0
    if speech_duration > 0:
        if speech_duration < 1.2 and category in ("hook", "motion"):
            dur_multiplier = 1.35
        elif speech_duration > 1.8 and category in ("face", "reaction"):
            dur_multiplier = 1.35
    cat_score = min(1.0, cat_score * dur_multiplier)

    # ── Motion score ──────────────────────────────────────────────────────────
    key = (clip_path, round(timestamp, 1))

    if key in _motion_cache:
        mot_score = _motion_cache[key]
    else:
        mot_score = _motion_score_at(clip_path, timestamp)
        _motion_cache[key] = mot_score

    # ── Text similarity via caption_hint ─────────────────────────────────────
    text_sim = 0.0
    caption_hint = result.get("caption_hint", "")
    if script_embedding is not None and caption_hint:
        try:
            from build_index import embed_text as _bi_embed_text
            hint_emb = _bi_embed_text(caption_hint)
            text_sim = float(np.dot(script_embedding, hint_emb))
            text_sim = max(0.0, text_sim)   # clamp negative cosine to 0
        except Exception:
            text_sim = 0.0

    # ── Reuse penalty ─────────────────────────────────────────────────────────
    reuse_pen = 0.30 if (prefer_unused and clip_path in used_paths) else 0.0
    global_decay = _RECENT_USED.get(clip_path, 0) * 0.04

    # ── Category continuity bonus ─────────────────────────────────────────────
    continuity_bonus = 0.05 if (prev_category and category == prev_category) else 0.0

    return (
    0.45 * sim_score
    + 0.25 * text_sim
    + 0.15 * mot_score
    + 0.15 * cat_score
    - reuse_pen
    - global_decay
    + continuity_bonus
)

def _select_via_faiss(
    line:            str,
    segment_dur:     float,
    used_paths:      Set[str],
    top_k:           int = 30,
    clip_dir:        str = None,
    speech_duration: float = 0.0,
    prev_category:   str = "",
    script_embedding: Optional[np.ndarray] = None,
) -> Optional[Tuple[str, float, str]]:
    """
    Use FAISS to find the best (clip_path, start_time, category) for `line`.
    Returns None if FAISS unavailable or no suitable result found.
    start_time is temporally centered on the matched frame.
    """
    if not _ensure_faiss():
        return None

    results = _faiss_search(line, top_k=top_k)
    if not results:
        return None

    # Score and sort results
    scored = []
    dur_cache: Dict[str, float] = {}

    for r in results:
        cp = r["clip_path"]

        # Resolve clip duration (cache to avoid repeated ffprobe calls)
        if cp not in dur_cache:
            dur_cache[cp] = _get_clip_duration(cp)
        clip_dur = dur_cache[cp]

        # ── TEMPORAL FIX: centre segment on matched frame ─────────────────────
        raw_ts    = r["timestamp"]
        start_time = raw_ts - (segment_dur / 2)
        start_time = max(0.0, start_time)

        # Ensure segment fits within clip
        if start_time + segment_dur > clip_dur:
            if clip_dur < segment_dur:
                continue   # clip too short — skip entirely
            start_time = clip_dur - segment_dur - 0.05
            start_time = max(0.0, start_time)

        r = dict(r)
        r["timestamp"] = round(start_time, 3)

        score = _score_faiss_result(
            r,
            segment_dur,
            used_paths,
            speech_duration=speech_duration,
            prev_category=prev_category,
            script_embedding=script_embedding,
        )
        scored.append((score, r))

    if not scored:
        return None

    import random

    scored.sort(key=lambda x: x[0], reverse=True)

    # take top candidates
    top_choices = scored[:6]

    # weighted random selection
    weights = [max(0.01, s) for s, _ in top_choices]

    best_score, best_result = random.choices(
        top_choices,
        weights=weights,
        k=1
    )[0]

    log_clip(
        f"  FAISS → {Path(best_result['clip_path']).name}  "
        f"ts={best_result['timestamp']:.2f}s  "
        f"cat={best_result['category']}  score={best_score:.3f}"
    )
    return best_result["clip_path"], best_result["timestamp"], best_result.get("category", "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Best-segment detection  (unchanged — used for fallback path)
# ─────────────────────────────────────────────────────────────────────────────

def find_best_segment(
    clip_path:        str,
    clip_duration:    float,
    segment_duration: float,
    text_emb:         Optional[np.ndarray],
    n_samples:        int = 8,
) -> float:
    """
    Scan `n_samples` candidate start times, score by semantic_sim + motion.
    Returns the best start time. Fallback to 0.0 on failure.
    """
    if clip_duration <= 0 or segment_duration <= 0:
        return 0.0

    max_start  = max(0.0, clip_duration - segment_duration)
    safe_start = clip_duration * 0.05
    safe_end   = clip_duration * 0.95 - segment_duration
    if safe_end <= safe_start:
        safe_start, safe_end = 0.0, max_start

    candidates = np.linspace(safe_start, min(safe_end, max_start), n_samples)
    best_start = float(candidates[0])
    best_score = -999.0

    for start in candidates:
        mid = start + segment_duration / 2

        sem = 0.0
        if text_emb is not None and _load_clip_model():
            emb = _embed_frame_at(clip_path, mid)
            if emb is not None:
                sem = float(np.dot(text_emb, emb))

        mot   = _motion_score_at(clip_path, mid)
        score = 0.65 * sem + 0.35 * mot

        if score > best_score:
            best_score = score
            best_start = float(start)

    log_clip(
        f"  best_segment: start={best_start:.2f}s  "
        f"score={best_score:.3f}  clip={Path(clip_path).name}"
    )
    return round(best_start, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Hook boost  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _hook_boost(clips: List[Dict], scores: List[float]) -> List[float]:
    prefer   = set(getattr(cfg, "HOOK_PREFER_TAGS",   ["face", "motion", "close-up", "reaction"]))
    penalise = set(getattr(cfg, "HOOK_PENALIZE_TAGS", ["landscape", "text", "static"]))
    boosted  = list(scores)
    for i, clip in enumerate(clips):
        stem = clip["stem"]
        if any(t in stem for t in prefer):
            boosted[i] *= 1.5
        if any(t in stem for t in penalise):
            boosted[i] *= 0.5
    return boosted


# ─────────────────────────────────────────────────────────────────────────────
# Duration helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def clamp_duration(speech_dur: float) -> float:
    """Legacy compat. Returns speech_dur clamped to configured pacing window."""
    lo = getattr(cfg, "CLIP_MIN_DUR", 1.6)
    hi = getattr(cfg, "CLIP_MAX_DUR", 2.4)
    return max(lo, min(hi, speech_dur))


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADED: Public API — line-aware clip selection
# ─────────────────────────────────────────────────────────────────────────────
_text_emb_cache = {}
def select_clips_for_lines(
    lines:          List[str],
    line_durations: List[float],
    clip_dir:       str = None,
) -> List[ClipSegment]:
    """
    UPGRADED: FAISS moment-level retrieval → per-clip fallback → stem fallback.

    Selection priority
    ------------------
    1. FAISS global moment search (if index available)
       → ranks by: CLIP similarity + category rank + motion + reuse penalty
       → start_time comes directly from frame metadata (no best_segment scan)

    2. Per-clip embedding fallback (if FAISS unavailable)
       → same logic as previous clip_engine version
       → best_segment used to find start_time within chosen clip

    3. Stem word-overlap fallback (if CLIP model also unavailable)

    TTS is master: segment_duration == speech_duration (+ 0.05s tail pad).
    """
    clips = scan_clips(clip_dir)
    if not clips:
        log_error("No clips found — returning blank segments")
        return _blank_segments(lines, line_durations)

    clip_model_ok = _load_clip_model()
    faiss_ok      = _ensure_faiss()
    _load_recent_used()

    used_paths:    Set[str]           = set()
    segments:      List[ClipSegment]  = []
    dur_cache:     Dict[str, float]   = {}
    prev_category: str                = ""   # continuity tracking

    for idx, (line, speech_dur) in enumerate(zip(lines, line_durations)):
        segment_dur = speech_dur + 0.05   # add small tail pad to ensure we don't cut off speech

        # Pre-compute script embedding once per line (reused in scorer)
        script_emb: Optional[np.ndarray] = None
        if clip_model_ok:
            try:
                if line in _text_emb_cache:
                    script_emb = _text_emb_cache[line]
                else:
                    script_emb = _text_embedding(line)
                    _text_emb_cache[line] = script_emb
            except Exception:
                script_emb = None

        # ── Path 1: FAISS moment search ───────────────────────────────────────
        if faiss_ok:
            faiss_result = _select_via_faiss(
                line=line,
                segment_dur=segment_dur,
                used_paths=used_paths,
                top_k=30,
                clip_dir=clip_dir,
                speech_duration=speech_dur,
                prev_category=prev_category,
                script_embedding=script_emb,
            )
            if faiss_result is not None:
                chosen_path, start_sec, chosen_cat = faiss_result
                segments.append((chosen_path, start_sec, segment_dur))
                used_paths.add(chosen_path)
                _RECENT_USED[chosen_path] = _RECENT_USED.get(chosen_path, 0) + 1
                prev_category = chosen_cat
                log_clip(
                    f"Line {idx:02d} | FAISS | speech={speech_dur:.2f}s"
                    f" | start={start_sec:.2f}s | {Path(chosen_path).name}"
                )
                continue

        # ── Path 2: Per-clip embedding fallback ───────────────────────────────
        log_clip(f"Line {idx}: FAISS miss — per-clip embedding fallback", level="debug")

        text_emb: Optional[np.ndarray] = script_emb   # reuse pre-computed embedding

        scores = _semantic_scores(line, clips)
        if idx == 0:
            scores = _hook_boost(clips, scores)

        ranked = sorted(enumerate(clips), key=lambda x: scores[x[0]], reverse=True)

        chosen: Optional[Dict] = None
        for _, clip in ranked:
            if clip["path"] not in used_paths and clip["duration"] >= segment_dur:
                chosen = clip
                break
        if chosen is None:
            for _, clip in ranked:
                if clip["duration"] >= segment_dur:
                    log_clip(f"Line {idx}: reuse {clip['filename']} (pool exhausted)", level="warning")
                    chosen = clip
                    break
        if chosen is None:
            chosen      = max(clips, key=lambda c: c["duration"])
            segment_dur = min(segment_dur, chosen["duration"] - 0.05)
            log_clip(f"Line {idx}: absolute fallback → {chosen['filename']}", level="warning")

        used_paths.add(chosen["path"])
        _RECENT_USED[chosen["path"]] = _RECENT_USED.get(chosen["path"], 0) + 1
        start_sec = find_best_segment(
            clip_path=chosen["path"],
            clip_duration=chosen["duration"],
            segment_duration=segment_dur,
            text_emb=text_emb,
            n_samples=8,
        )
        segments.append((chosen["path"], start_sec, segment_dur))
        prev_category = chosen.get("category", "unknown")
        log_clip(
            f"Line {idx:02d} | FALLBACK | speech={speech_dur:.2f}s"
            f" | start={start_sec:.2f}s | {chosen['filename']}"
        )
    # slowly decay reuse memory
    for k in list(_RECENT_USED.keys()):
        _RECENT_USED[k] *= 0.85

        # cleanup tiny values
        if _RECENT_USED[k] < 0.05:
            del _RECENT_USED[k]
    _save_recent_used()
    return segments


def _blank_segments(lines: List[str], line_durations: List[float]) -> List[ClipSegment]:
    target = getattr(cfg, "CLIP_TARGET_DUR", 2.0)
    return [
        ("__blank__", 0.0, max(0.5, d if d > 0 else target))
        for d in line_durations
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Legacy API shim  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def select_clips(
    script:        str,
    total_duration: float,
    clip_dir:      str   = None,
    use_semantic:  bool  = None,
    seg_dur:       float = None,
) -> List[ClipSegment]:
    """DEPRECATED — use select_clips_for_lines(). Kept for backward compat."""
    seg_dur = seg_dur or getattr(cfg, "CLIP_TARGET_DUR", 2.0)
    n_clips = max(1, int(total_duration / seg_dur) + 1)
    lines   = [script] * n_clips
    durs    = [seg_dur] * n_clips
    return select_clips_for_lines(lines, durs, clip_dir)