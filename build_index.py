"""
build_index.py  ── CLIP + FAISS video frame indexer

Scans all clips recursively, extracts smart timestamps, computes CLIP embeddings,
and builds a FAISS IndexFlatIP (cosine similarity) index for moment-level retrieval.

Usage
-----
  python build_index.py                     # index clips/ directory
  python build_index.py --dir /path/clips   # custom directory
  python build_index.py --force             # re-index everything (ignore resume state)
  python build_index.py --frames 10         # frames per clip (default 8)
  python build_index.py --out index_dir     # custom output directory

Output
------
  index_dir/
    index.faiss       — FAISS IndexFlatIP (d=512, normalised vectors)
    metadata.json     — [{clip_path, timestamp, category, clip_hash}, ...]
    indexed_clips.json — {clip_path: clip_hash} — resume state

Requirements
------------
  pip install faiss-cpu transformers torch opencv-python pillow numpy
  (or faiss-gpu if CUDA available)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("[index] ERROR: Pillow required — pip install pillow")
    sys.exit(1)

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    print("[index] ERROR: transformers + torch required — pip install transformers torch")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("[index] ERROR: faiss required — pip install faiss-cpu")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CLIPS_DIR  = "clips"
DEFAULT_INDEX_DIR  = "index"
DEFAULT_N_FRAMES   = 3        # Frames per clip (smart timestamps)
SUPPORTED_EXTS     = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CLIP_MODEL_NAME    = "openai/clip-vit-base-patch32"
EMBEDDING_DIM      = 512        # ViT-B/32 output dimension
BATCH_SIZE         = 16         # Frames per CLIP batch

# Known category folder names (matches clip_sorter_v2 + upgraded sorter)
KNOWN_CATEGORIES   = {
    "hook", "face", "luxury", "motion", "reaction", "neutral",
    "shy", "talking", "walking", "city", "nature", "data_visual", "misc",
}


# ─────────────────────────────────────────────────────────────────────────────
# CLIP model singleton
# ─────────────────────────────────────────────────────────────────────────────

_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_model:  Optional[CLIPModel]     = None
_proc:   Optional[CLIPProcessor] = None


def _load_model() -> Tuple[CLIPModel, CLIPProcessor]:
    global _model, _proc
    if _model is not None:
        return _model, _proc
    print(f"[index] Loading CLIP ({CLIP_MODEL_NAME}) on {_device}…")
    _model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device)
    _proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    _model.eval()
    print("[index] CLIP model ready")
    return _model, _proc


# ─────────────────────────────────────────────────────────────────────────────
# Smart frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def _clip_duration(path: str) -> float:
    """Fast duration read via OpenCV."""
    cap = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0 and total > 0:
        return float(total / fps)
    return 0.0


def _clip_hash(path: str) -> str:
    """Stable hash from path + mtime + size (no full file read)."""
    try:
        stat = os.stat(path)
        raw  = f"{path}|{stat.st_mtime}|{stat.st_size}"
        return md5(raw.encode()).hexdigest()[:16]
    except Exception:
        return md5(path.encode()).hexdigest()[:16]


def smart_timestamps(duration: float, n: int = DEFAULT_N_FRAMES) -> List[float]:
    """
    Return `n` evenly-spaced timestamps in [10%, 90%] of clip duration.
    Avoids black intro/outro frames.
    """
    start = duration * 0.10
    end   = duration * 0.90
    if end <= start:
        start, end = 0.0, duration
    return list(np.linspace(start, end, n))


def extract_frame(path: str, timestamp_sec: float) -> Optional[Image.Image]:
    """Extract a single frame at `timestamp_sec`. Returns PIL Image or None."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding computation (batched)
# ─────────────────────────────────────────────────────────────────────────────

def embed_images_batch(images: List[Image.Image]) -> np.ndarray:
    """
    Compute CLIP image embeddings for a batch of PIL Images.
    Returns float32 array [N, EMBEDDING_DIM], L2-normalised.
    """
    model, proc = _load_model()

    inputs = proc(images=images, return_tensors="pt", padding=True).to(_device)

    with torch.no_grad():
        feats = model.get_image_features(**inputs)   # [N, 512]

    embs = feats.cpu().float().numpy()

    # L2 normalise each row (required for cosine via IndexFlatIP)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs  = embs / norms

    return embs


def embed_text(text: str) -> np.ndarray:
    """
    Compute normalised CLIP text embedding. Returns float32 [EMBEDDING_DIM].
    Useful for query-time retrieval (exported for clip_engine.py to import).
    """
    model, proc = _load_model()
    inputs = proc(text=[text], return_tensors="pt",
                  padding=True, truncation=True, max_length=77).to(_device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    emb = feats.cpu().float().numpy().flatten()
    emb /= (np.linalg.norm(emb) + 1e-8)
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Clip discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_clips(clips_dir: str) -> List[Dict]:
    """
    Walk `clips_dir` recursively. For each video file return:
      { path, category, duration }
    category = parent folder name if it matches KNOWN_CATEGORIES, else "unknown"
    """
    found = []
    root  = Path(clips_dir)

    for fpath in sorted(root.rglob("*")):
        if fpath.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if fpath.name.startswith("."):
            continue

        rel    = fpath.relative_to(root)
        parts  = rel.parts
        cat    = parts[0] if len(parts) > 1 and parts[0] in KNOWN_CATEGORIES else "unknown"

        dur = _clip_duration(str(fpath))
        if dur < 1.0:
            print(f"[index] Skip (too short {dur:.1f}s): {fpath.name}")
            continue

        found.append({
            "path":     str(fpath),
            "category": cat,
            "duration": dur,
        })

    return found


# ─────────────────────────────────────────────────────────────────────────────
# Resume state
# ─────────────────────────────────────────────────────────────────────────────

def _load_resume(index_dir: str) -> Dict[str, str]:
    """Load {clip_path: clip_hash} from disk for resume support."""
    path = os.path.join(index_dir, "indexed_clips.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_resume(index_dir: str, state: Dict[str, str]) -> None:
    path = os.path.join(index_dir, "indexed_clips.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_existing_index(index_dir: str) -> Tuple[Optional[faiss.Index], List[Dict]]:
    """Load existing FAISS index + metadata for resume/append mode."""
    idx_path  = os.path.join(index_dir, "index.faiss")
    meta_path = os.path.join(index_dir, "metadata.json")

    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        return None, []

    try:
        idx      = faiss.read_index(idx_path)
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[index] Loaded existing index: {idx.ntotal} vectors, {len(meta)} metadata entries")
        return idx, meta
    except Exception as e:
        print(f"[index] Could not load existing index ({e}) — rebuilding from scratch")
        return None, []


# ─────────────────────────────────────────────────────────────────────────────
# Main indexing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    clips_dir:  str  = DEFAULT_CLIPS_DIR,
    index_dir:  str  = DEFAULT_INDEX_DIR,
    n_frames:   int  = DEFAULT_N_FRAMES,
    force:      bool = False,
    verbose:    bool = True,
) -> None:
    """
    Full indexing pipeline:
      1. Discover clips
      2. Skip already-indexed clips (resume mode) unless --force
      3. Extract smart frames, compute CLIP embeddings
      4. Append to FAISS IndexFlatIP
      5. Save index.faiss + metadata.json + indexed_clips.json
    """
    os.makedirs(index_dir, exist_ok=True)

    # ── Load existing state ───────────────────────────────────────────────────
    resume_state             = {} if force else _load_resume(index_dir)
    existing_index, all_meta = (None, []) if force else _load_existing_index(index_dir)

    # ── Discover clips ────────────────────────────────────────────────────────
    clips = discover_clips(clips_dir)
    print(f"[index] Found {len(clips)} clips in '{clips_dir}'")

    if not clips:
        print("[index] Nothing to index.")
        return

    # ── Filter already-indexed clips ──────────────────────────────────────────
    to_index = []
    skipped  = 0

    for clip in clips:
        h = _clip_hash(clip["path"])
        if not force and clip["path"] in resume_state and resume_state[clip["path"]] == h:
            skipped += 1
            continue
        clip["hash"] = h
        to_index.append(clip)

    print(f"[index] Skipping {skipped} already-indexed clips | Processing {len(to_index)} new/changed clips")

    if not to_index and existing_index is not None:
        print("[index] All clips already indexed. Use --force to rebuild.")
        return

    # ── Pre-load model ────────────────────────────────────────────────────────
    _load_model()

    # ── Collect all new embeddings ─────────────────────────────────────────────
    new_embeddings: List[np.ndarray] = []
    new_meta:       List[Dict]       = []
    errors = 0

    total_clips = len(to_index)
    for ci, clip in enumerate(to_index):
        t0 = time.perf_counter()

        timestamps = smart_timestamps(clip["duration"], n_frames)
        clip_embs  = []
        clip_metas = []

        # Extract frames in batch for this clip
        frame_batch: List[Image.Image] = []
        valid_ts:    List[float]        = []

        for ts in timestamps:
            img = extract_frame(clip["path"], ts)
            if img is not None:
                frame_batch.append(img)
                valid_ts.append(ts)

        if not frame_batch:
            print(f"[index] WARNING: No frames extracted from {clip['path']} — skipping")
            errors += 1
            continue

        # Process in sub-batches to avoid OOM
        for bi in range(0, len(frame_batch), BATCH_SIZE):
            batch_imgs = frame_batch[bi : bi + BATCH_SIZE]
            batch_ts   = valid_ts[bi : bi + BATCH_SIZE]

            try:
                batch_embs = embed_images_batch(batch_imgs)   # [B, 512]
            except Exception as e:
                print(f"[index] Embed error for {clip['path']} batch {bi}: {e}")
                errors += 1
                continue

            for emb, ts in zip(batch_embs, batch_ts):
                clip_embs.append(emb)
                # Build caption_hint: "category stem" — used for text_similarity scoring
                stem = Path(clip["path"]).stem.replace("_", " ").replace("-", " ").lower()
                caption_hint = f"{clip['category']} {stem}"
                clip_metas.append({
                    "clip_path":    clip["path"],
                    "timestamp":    round(float(ts), 3),
                    "category":     clip["category"],
                    "clip_hash":    clip["hash"],
                    "caption_hint": caption_hint,
                })

        new_embeddings.extend(clip_embs)
        new_meta.extend(clip_metas)
        resume_state[clip["path"]] = clip["hash"]

        elapsed = time.perf_counter() - t0
        if verbose:
            print(
                f"  [{ci+1:>4}/{total_clips}] {Path(clip['path']).name[:40]:<40}"
                f"  frames={len(clip_embs)}  cat={clip['category']}  ({elapsed:.1f}s)"
            )

    if not new_embeddings and existing_index is None:
        print("[index] No embeddings produced — cannot build index.")
        return

    # ── Build / update FAISS index ────────────────────────────────────────────
    print(f"[index] Building FAISS index ({EMBEDDING_DIM}d, IndexFlatIP)…")

    if existing_index is not None and new_embeddings:
        # Append new vectors to existing index
        new_mat = np.array(new_embeddings, dtype=np.float32)
        existing_index.add(new_mat)
        idx = existing_index
        all_meta.extend(new_meta)
    elif new_embeddings:
        # Fresh index
        new_mat  = np.array(new_embeddings, dtype=np.float32)
        idx      = faiss.IndexFlatIP(EMBEDDING_DIM)
        idx.add(new_mat)
        all_meta = new_meta
    else:
        # Nothing new, but existing index is fine
        idx = existing_index

    total_vectors = idx.ntotal
    print(f"[index] FAISS index: {total_vectors} total vectors ({len(all_meta)} metadata entries)")

    # ── Save everything ───────────────────────────────────────────────────────
    idx_path  = os.path.join(index_dir, "index.faiss")
    meta_path = os.path.join(index_dir, "metadata.json")

    faiss.write_index(idx, idx_path)
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, separators=(",", ":"))   # compact JSON

    _save_resume(index_dir, resume_state)

    print(f"[index] Saved:")
    print(f"  {idx_path}   ({os.path.getsize(idx_path) // 1024} KB)")
    print(f"  {meta_path}  ({os.path.getsize(meta_path) // 1024} KB)")
    print(f"[index] Done. errors={errors}  total_vectors={total_vectors}")


# ─────────────────────────────────────────────────────────────────────────────
# FAISS retrieval  (importable by clip_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

_faiss_index:    Optional[faiss.Index] = None
_faiss_metadata: Optional[List[Dict]]  = None
_faiss_index_dir: str                   = DEFAULT_INDEX_DIR


def load_faiss_index(index_dir: str = DEFAULT_INDEX_DIR) -> bool:
    """
    Load FAISS index + metadata into module-level singletons.
    Call once at startup; subsequent calls are no-ops.
    Returns True if loaded successfully.
    """
    global _faiss_index, _faiss_metadata, _faiss_index_dir

    if _faiss_index is not None:
        return True

    idx_path  = os.path.join(index_dir, "index.faiss")
    meta_path = os.path.join(index_dir, "metadata.json")

    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        print(f"[index] FAISS index not found at '{index_dir}' — run build_index.py first")
        return False

    try:
        _faiss_index = faiss.read_index(idx_path)
        with open(meta_path) as f:
            _faiss_metadata = json.load(f)
        _faiss_index_dir = index_dir
        print(f"[index] FAISS loaded: {_faiss_index.ntotal} vectors, {len(_faiss_metadata)} frames")
        return True
    except Exception as e:
        print(f"[index] FAISS load error: {e}")
        return False


def search_index(
    text: str,
    top_k: int = 20,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> List[Dict]:
    """
    Encode `text` with CLIP, search FAISS, return top_k frame metadata dicts
    each augmented with a `score` (cosine similarity, higher = better).

    Returns [] on failure (caller should fall back to legacy system).
    """
    global _faiss_index, _faiss_metadata

    if not load_faiss_index(index_dir):
        return []

    if _faiss_index is None or not _faiss_metadata:
        return []

    try:
        q_emb = embed_text(text).reshape(1, -1).astype(np.float32)
        scores, indices = _faiss_index.search(q_emb, min(top_k, _faiss_index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(_faiss_metadata):
                continue
            entry = dict(_faiss_metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results
    except Exception as e:
        print(f"[index] FAISS search error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CLIP+FAISS index for video clips")
    parser.add_argument("--dir",    default=DEFAULT_CLIPS_DIR, help="Clips root directory")
    parser.add_argument("--out",    default=DEFAULT_INDEX_DIR,  help="Output index directory")
    parser.add_argument("--frames", type=int, default=DEFAULT_N_FRAMES,
                        help=f"Frames per clip (default {DEFAULT_N_FRAMES})")
    parser.add_argument("--force",  action="store_true",
                        help="Re-index all clips (ignore resume state)")
    parser.add_argument("--quiet",  action="store_true", help="Less output")
    args = parser.parse_args()

    print(f"[index] Starting — dir={args.dir}  out={args.out}  frames={args.frames}  force={args.force}")
    build_index(
        clips_dir=args.dir,
        index_dir=args.out,
        n_frames=args.frames,
        force=args.force,
        verbose=not args.quiet,
    )