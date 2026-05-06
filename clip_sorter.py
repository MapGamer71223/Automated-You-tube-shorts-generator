"""
clip_sorter_v2.py  ── Production-grade CLIP-based video organiser (UPGRADED)

Upgrades over original v2
--------------------------
✓ New required category set: hook / face / luxury / motion / reaction / neutral
✓ Stronger, more discriminative CLIP prompts (3–4 per category)
✓ Per-category prompt averaging to prevent prompt-count bias
✓ Confidence-based fallback to "neutral" (replaces "misc")
✓ All original features preserved:
    - Multi-frame analysis (5 frames, evenly spaced)
    - Re-sort support (--rescan)
    - Dry-run mode (--dry-run)
    - Face detection boost (optional)
    - Motion score (optional)
    - Loop-safe state tracking
    - Category summary after sort

Usage
-----
  python clip_sorter_v2.py               # sort everything
  python clip_sorter_v2.py --dry-run     # preview without moving
  python clip_sorter_v2.py --rescan      # force re-classify already-sorted clips
  python clip_sorter_v2.py --dir /path   # custom clips directory
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CLIPS_DIR            = "clips"
CONFIDENCE_THRESHOLD = 0.05   # Below this → "neutral" fallback
NUM_FRAMES           = 5        # Frames to sample per clip (evenly spaced)
SUPPORTED_EXTS       = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
STATE_FILE           = ".sorter_state.json"
ENABLE_FACE_SCORE    = True
ENABLE_MOTION_SCORE  = True
CLIP_MODEL_NAME      = "openai/clip-vit-base-patch32"


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADED Category definitions
#
# New required categories: hook, face, luxury, motion, reaction, neutral
# Prompt engineering rules:
#   1. Be specific about what's IN frame (objects, environment, lighting)
#   2. Include compositional cues ("close-up", "wide shot", "from behind")
#   3. 3–4 prompts per category, visually distinct across categories
#   4. Avoid overlap — prompts for one category shouldn't describe another
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIES: Dict[str, List[str]] = {
    "hook": [
    "a person pointing at camera close up",
    "a face with intense eye contact",
    "a sudden movement towards camera",
    "a zoomed in dramatic face expression"
],

    "face": [
        "a clear human face portrait centered in frame",
        "a person looking directly into camera neutral expression",
        "a static talking head close-up"
    ],

    "reaction": [
    "a surprised face with wide eyes and open mouth",
    "a person laughing or smiling strongly",
    "a shocked facial expression close up",
    "a person reacting emotionally to something"
],
   
    "luxury": [
        # Upscale, premium — expensive materials, soft light, aspirational
       
        "a man showing expensive watches or jewelry",
        "a luxury car like Lamborghini or Rolls Royce close up",
        "rich lifestyle with money, cash, private jet, mansion",
        "high-end environment with wealth and status"
    ],

    "motion": [
        # Kinetic energy — movement, speed, blurred backgrounds, dynamic subjects
        "fast movement scene with camera motion or action",
        "a person running, jumping or in dynamic motion",   
        "walking, running or dynamic movement",
        "cinematic movement shot with motion blur"
    ],

   

    "neutral": [
        # Calm, ambient, non-specific — talking head, static scene, background filler
        "background scene with no clear subject",
        "empty environment or generic visuals",
        "low detail or unclear subject"
    ],
}


# ── Flat text list for batch inference (built from CATEGORIES above) ─────────

ALL_TEXTS:    List[str] = []
CATEGORY_MAP: List[str] = []

for _cat, _prompts in CATEGORIES.items():
    for _p in _prompts:
        ALL_TEXTS.append(_p)
        CATEGORY_MAP.append(_cat)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (singleton)
# ─────────────────────────────────────────────────────────────────────────────

_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_model:  Optional[CLIPModel]     = None
_processor: Optional[CLIPProcessor] = None


def _load_model() -> Tuple[CLIPModel, CLIPProcessor]:
    global _model, _processor
    if _model is not None:
        return _model, _processor
    print(f"[sorter] Loading CLIP on {_device}…")
    _model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device)
    _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    _model.eval()
    print("[sorter] CLIP model ready")
    return _model, _processor


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int = NUM_FRAMES) -> List[Image.Image]:
    """
    Sample `num_frames` evenly from the clip, skipping first and last 5%
    to avoid intro/outro black frames. Returns list of PIL Images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    start_idx = max(0, int(total * 0.05))
    end_idx   = min(total - 1, int(total * 0.95))
    if end_idx <= start_idx:
        start_idx, end_idx = 0, total - 1

    indices = np.linspace(start_idx, end_idx, num_frames, dtype=int)
    frames: List[Image.Image] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Optional: face detection score  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

_face_cascade: Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(path):
            _face_cascade = cv2.CascadeClassifier(path)
            return _face_cascade
    except Exception:
        pass
    return None


def face_score(video_path: str, sample_frames: int = 3) -> float:
    """Returns 0.0–1.0 — fraction of sampled frames containing a detected face."""
    if not ENABLE_FACE_SCORE:
        return 0.0
    cascade = _get_face_cascade()
    if cascade is None or cascade.empty():
        return 0.0

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0.0

    indices = np.linspace(0, total - 1, sample_frames, dtype=int)
    hits    = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(faces) > 0:
            hits += 1

    cap.release()
    return hits / sample_frames


# ─────────────────────────────────────────────────────────────────────────────
# Optional: motion score  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def motion_score(video_path: str, sample_frames: int = 8) -> float:
    """Returns 0.0–1.0 normalised motion energy via mean abs frame difference."""
    if not ENABLE_MOTION_SCORE:
        return 0.0

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return 0.0

    indices     = np.linspace(0, total - 1, sample_frames, dtype=int)
    frames_gray = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            small = cv2.resize(frame, (64, 64))
            frames_gray.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))

    cap.release()
    if len(frames_gray) < 2:
        return 0.0

    diffs    = [np.mean(np.abs(frames_gray[i+1] - frames_gray[i]))
                for i in range(len(frames_gray) - 1)]
    avg_diff = float(np.mean(diffs))
    return float(min(1.0, avg_diff / 50.0))


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADED: Core classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_clip(
    video_path: str,
    model:      CLIPModel,
    processor:  CLIPProcessor,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Classify a clip using multi-frame CLIP analysis with per-category averaging.

    Per-category averaging ensures categories with more prompts are not
    unfairly boosted over categories with fewer prompts.

    Returns
    -------
    (category, confidence, per_category_scores)
      category   : best matching category string
      confidence : averaged probability of best category (0–1)
      scores     : {category: avg_probability}
    """
    frames = extract_frames(video_path, NUM_FRAMES)
    if not frames:
        return "neutral", 0.0, {}

    inputs = processor(
        text=ALL_TEXTS,
        images=frames,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(_device)

    with torch.no_grad():
        outputs  = model(**inputs)
        logits   = outputs.logits_per_image   # [num_frames, num_texts]
        probs    = logits.softmax(dim=1)      # normalise across all texts

    avg_probs = probs.mean(dim=0).cpu().numpy()   # [num_texts]

    # ── Per-category averaging ────────────────────────────────────────────────
    cat_sum:    Dict[str, float] = {}
    cat_counts: Dict[str, int]   = {}

    for prob, cat in zip(avg_probs, CATEGORY_MAP):
        cat_sum[cat]    = cat_sum.get(cat, 0.0) + float(prob)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    cat_scores: Dict[str, float] = {
        cat: cat_sum[cat] / cat_counts[cat]
        for cat in cat_sum
    }

    best_cat   = max(cat_scores, key=cat_scores.get)
    best_score = cat_scores[best_cat]

    # ── UPGRADED: face/motion post-correction ────────────────────────────────
    # If CLIP is uncertain AND a strong face is detected, nudge toward "face"
    # If CLIP is uncertain AND strong motion detected, nudge toward "motion"
    # These are soft nudges — they only apply below 1.5× the threshold
    if best_score < 0.12:
        f = face_score(video_path)
        m = motion_score(video_path)
        if f >= 0.6 and cat_scores.get("face", 0) > cat_scores.get("neutral", 0):
            best_cat   = "face"
            best_score = cat_scores["face"]
        elif m >= 0.7 and cat_scores.get("motion", 0) > cat_scores.get("neutral", 0):
            best_cat   = "motion"
            best_score = cat_scores["motion"]
    # prevent neutral dominance
    if best_cat == "neutral":
        sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_cats[1][1] > 0.04:
            best_cat = sorted_cats[1][0]
            best_score = sorted_cats[1][1]
    # ── Low-confidence fallback ───────────────────────────────────────────────
    # Only fallback if EVERYTHING is garbage
    if max(cat_scores.values()) < 0.03:
        return "neutral", best_score, cat_scores
        
    return best_cat, best_score, cat_scores


# ─────────────────────────────────────────────────────────────────────────────
# State persistence  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _load_state(clips_dir: str) -> Dict[str, str]:
    path = os.path.join(clips_dir, STATE_FILE)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(clips_dir: str, state: Dict[str, str]) -> None:
    with open(os.path.join(clips_dir, STATE_FILE), "w") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Clip discovery  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def discover_clips(clips_dir: str) -> List[Dict]:
    """
    Walk the full clips directory tree. Returns:
      [{path, filename, current_category}]
    current_category = None if clip is in root (unsorted).
    """
    found    = []
    root_path = Path(clips_dir)

    for fpath in root_path.rglob("*"):
        if fpath.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if fpath.name.startswith("."):
            continue

        relative = fpath.relative_to(root_path)
        parts    = relative.parts

        if len(parts) == 1:
            current_cat = None
        elif len(parts) >= 2 and parts[0] in CATEGORIES:
            current_cat = parts[0]
        else:
            continue   # unknown subfolder — skip

        found.append({
            "path":             str(fpath),
            "filename":         fpath.name,
            "current_category": current_cat,
        })

    return found


# ─────────────────────────────────────────────────────────────────────────────
# Main sort/re-sort orchestrator  (unchanged interface from v1)
# ─────────────────────────────────────────────────────────────────────────────

def sort_clips(
    clips_dir: str  = CLIPS_DIR,
    dry_run:   bool = False,
    rescan:    bool = False,
    verbose:   bool = True,
) -> None:
    """
    Sort (and optionally re-sort) all clips in clips_dir.

    Parameters
    ----------
    clips_dir : Root clips directory
    dry_run   : Preview moves without executing
    rescan    : Re-classify already-sorted clips
    verbose   : Print per-clip details
    """
    model, processor = _load_model()
    state  = _load_state(clips_dir)
    clips  = discover_clips(clips_dir)

    if not clips:
        print(f"[sorter] No clips found in {clips_dir}")
        return

    unsorted = [c for c in clips if c["current_category"] is None]
    sorted_  = [c for c in clips if c["current_category"] is not None]

    to_process = list(unsorted)
    if rescan:
        to_process += sorted_
        print(f"[sorter] Re-scan mode: {len(unsorted)} unsorted + {len(sorted_)} already-sorted clips")
    else:
        print(f"[sorter] Processing {len(unsorted)} unsorted clips "
              f"({len(sorted_)} already sorted — use --rescan to re-evaluate)")

    if not to_process:
        print("[sorter] Nothing to process.")
        return

    moved  = 0
    stayed = 0
    errors = 0

    for clip in to_process:
        src_path    = clip["path"]
        filename    = clip["filename"]
        current_cat = clip["current_category"]

        t0 = time.perf_counter()
        try:
            new_cat, confidence, all_scores = classify_clip(src_path, model, processor)
        except Exception as e:
            print(f"[sorter] ERROR classifying {filename}: {e}")
            errors += 1
            continue

        elapsed = time.perf_counter() - t0

        f_score = face_score(src_path)   if ENABLE_FACE_SCORE   else 0.0
        m_score = motion_score(src_path) if ENABLE_MOTION_SCORE else 0.0

        if verbose:
            top3     = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top3_str = "  ".join(f"{k}={v:.3f}" for k, v in top3)
            face_s   = f"  face={f_score:.2f}" if ENABLE_FACE_SCORE   else ""
            mot_s    = f"  motion={m_score:.2f}" if ENABLE_MOTION_SCORE else ""
            print(
                f"  {filename[:35]:<35}  → {new_cat:<10} conf={confidence:.3f}"
                f"  [{top3_str}]{face_s}{mot_s}  ({elapsed:.1f}s)"
            )

        if current_cat == new_cat:
            stayed += 1
            state[filename] = new_cat
            continue

        dst_dir  = os.path.join(clips_dir, new_cat)
        dst_path = os.path.join(dst_dir, filename)

        if dry_run:
            action = "MOVE" if current_cat else "SORT"
            print(f"    [dry-run] {action}: {current_cat or 'root'} → {new_cat}")
            stayed += 1
            continue

        # Handle filename collision
        if os.path.exists(dst_path):
            stem, ext = os.path.splitext(filename)
            counter   = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(dst_dir, f"{stem}_{counter}{ext}")
                counter += 1

        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src_path, dst_path)
        state[filename] = new_cat
        moved += 1

        if current_cat:
            print(f"    ↳ RE-SORTED: {current_cat}/ → {new_cat}/")
        else:
            print(f"    ↳ SORTED → {new_cat}/")

    _save_state(clips_dir, state)
    print(f"\n[sorter] Done. moved={moved}  stayed={stayed}  errors={errors}")
    _print_category_summary(clips_dir)


def _print_category_summary(clips_dir: str) -> None:
    """Print count of clips per category after sorting."""
    print("\n[sorter] Category summary:")
    root = Path(clips_dir)
    for cat in sorted(CATEGORIES.keys()):
        cat_dir = root / cat
        if cat_dir.exists():
            count = sum(
                1 for f in cat_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTS
            )
            bar = "█" * min(count, 30)
            print(f"  {cat:<12} {count:>4}  {bar}")
    unsorted = sum(
        1 for f in root.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )
    if unsorted:
        print(f"  {'(unsorted)':<12} {unsorted:>4}  ← still needs sorting")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI clip sorter v2 (upgraded)")
    parser.add_argument("--dir",     default=CLIPS_DIR, help="Clips directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--rescan",  action="store_true", help="Re-classify sorted clips")
    parser.add_argument("--quiet",   action="store_true", help="Less output")
    args = parser.parse_args()

    print(f"[sorter] Starting — dir={args.dir}  dry_run={args.dry_run}  rescan={args.rescan}")
    sort_clips(
        clips_dir=args.dir,
        dry_run=args.dry_run,
        rescan=args.rescan,
        verbose=not args.quiet,
    )