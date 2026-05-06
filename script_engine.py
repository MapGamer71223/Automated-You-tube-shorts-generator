"""
modules/script_engine.py

Responsibilities
----------------
• Generate scripts via local LLM (LM Studio / Mistral)
• Batch-request N scripts in a single LLM call  (reduces round-trips)
• Validate + auto-repair every script
• Score and rank scripts
• ALWAYS return at least one usable script (fallback chain)

Never raises an unhandled exception — returns a fallback instead.
"""

from __future__ import annotations

import json
import random
import re
import string
import time
from typing import List, Optional, Tuple

import requests

# ── Local imports ──────────────────────────────────────────────────────────────
from logger import (
    log_llm, log_filter, log_score, log_selected, log_error, log_debug
)
import settings as cfg

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

_BATCH_SYSTEM = (
    "You are an elite viral content creator for TikTok and YouTube Shorts.\n"
    "You create scripts that feel like realizations, not explanations.\n"
    "Every script must create curiosity, tension, or contradiction.\n"
    "Avoid generic facts. Make it feel like the viewer misunderstood something.\n"
    "Short, punchy, emotionally engaging.\n"
    "No boring educational tone."
)

def _build_batch_prompt(topic: str, n: int) -> str:
    return (
        f"Write exactly {n} different viral YouTube Shorts scripts about: \"{topic}\".\n\n"
        f"STRICT RULES:\n"
        f"- Each script: {cfg.SCRIPT_MIN_WORDS}–{cfg.SCRIPT_MAX_WORDS} words\n"
        f"- One script per line\n"
        f"- No numbering, no bullet points, no quotes\n"
        f"- Plain sentences only\n"
        f"- Each must end with a period, exclamation mark, or question mark\n\n"
        f"Output {n} lines, one script per line:"
    )

def _build_single_prompt(topic: str) -> str:
    return (
        f"Write one viral YouTube Shorts script about: \"{topic}\".\n\n"
        f"Rules: {cfg.SCRIPT_MIN_WORDS}–{cfg.SCRIPT_MAX_WORDS} words, "
        f"plain sentence, ends with punctuation. No commentary."
    )

# ──────────────────────────────────────────────────────────────────────────────
# LLM call (with retry)
# ──────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, system: str = _BATCH_SYSTEM, retries: int = 2) -> Optional[str]:
    payload = {
        "model": cfg.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":   cfg.LLM_MAX_TOKENS,
        "temperature":  cfg.LLM_TEMPERATURE,
    }
    url = f"{cfg.LLM_BASE_URL}/chat/completions"

    for attempt in range(retries + 1):
        try:
            log_llm(f"Calling LLM (attempt {attempt + 1})…")
            r = requests.post(url, json=payload, timeout=cfg.LLM_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            log_llm(f"LLM returned {len(text)} chars")
            return text
        except requests.exceptions.ConnectionError:
            log_error("LLM not reachable — is LM Studio running?")
            return None
        except Exception as e:
            log_error(f"LLM call failed (attempt {attempt + 1}): {e}", exc=True)
            if attempt < retries:
                time.sleep(1.5)
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Validation + Repair
# ──────────────────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())

def _has_terminal_punct(text: str) -> bool:
    return text.rstrip()[-1:] in ".!?"

def _repair_script(text: str) -> str:
    """Best-effort repair: strip junk, fix length, add punctuation."""
    # Remove leading numbers / bullets  e.g. "1. " "- " "• "
    text = re.sub(r"^\s*[\d]+[.)]\s*|^\s*[-•]\s*", "", text).strip()
    # Remove surrounding quotes
    text = text.strip('"\'')
    # Collapse whitespace
    text = " ".join(text.split())
    # Truncate to max words
    words = text.split()
    if len(words) > cfg.SCRIPT_MAX_WORDS:
        # cut at last sentence boundary within limit
        joined = " ".join(words[:cfg.SCRIPT_MAX_WORDS])
        # try to end on punctuation
        m = re.search(r"[.!?][^.!?]*$", joined)
        if m:
            text = joined[: m.start() + 1]
        else:
            text = joined.rstrip(string.punctuation) + "."
    # Ensure terminal punctuation
    if not _has_terminal_punct(text):
        text = text.rstrip() + "."
    # Capitalise first letter
    if text:
        text = text[0].upper() + text[1:]
    return text
def _improve_script(text: str) -> str:
    FILLERS = ["you know", "basically", "kind of", "sort of", "i mean", "like,", "actually,", "so,", "well,"]
    
    low = text.lower()
    for f in FILLERS:
        low = low.replace(f, "")

    text = low.strip()
    if text:
        text = text[0].upper() + text[1:]

    return text


def _mutate_hook(text: str) -> str:
    patterns = [
        "They don’t say no directly",
        "It sounds like yes… but it isn’t",
        "You think they agreed… they didn’t",
        "This is how rejection actually sounds",
        "You misunderstood what they meant",
    ]

    words = text.split()
    if len(words) > 5:
        return random.choice(patterns) + " " + " ".join(words[5:])
    
    return text
def _is_valid(text: str) -> Tuple[bool, str]:
    """Return (valid, reason). Reason is empty string when valid."""
    wc = _word_count(text)
    if wc < cfg.SCRIPT_MIN_WORDS:
        return False, f"too short ({wc} words)"
    if wc > cfg.SCRIPT_MAX_WORDS:
        return False, f"too long ({wc} words)"
    if not _has_terminal_punct(text):
        return False, "missing terminal punctuation"
    return True, ""

# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

# Trigger words associated with high-performing Shorts
_HOOK_WORDS = {
    "you", "your", "secret", "never", "always", "most", "every",
    "shocking", "why", "what", "how", "this", "nobody", "actually",
    "fact", "discovered", "found", "proven", "truth", "real",
    "did you know", "turns out", "scientists", "study", "billion",
    "million", "percent", "%", "minutes", "seconds",
}

def _score_script(text: str) -> float:
    """
    Returns a float 0.0–1.0.
    Higher = better viral potential.
    """
    lower = text.lower()
    wc    = _word_count(text)

    # Length score (peak around ideal range)
    ideal_mid = (cfg.SCRIPT_IDEAL_MIN + cfg.SCRIPT_IDEAL_MAX) / 2
    length_score = max(0.0, 1.0 - abs(wc - ideal_mid) / ideal_mid)

    # Hook word presence
    hook_count  = sum(1 for w in _HOOK_WORDS if w in lower)
    hook_score  = min(1.0, hook_count / 4)

    # Question hook bonus
    question_bonus = 0.15 if text.startswith(("Did", "What", "Why", "How", "Can", "Is ")) else 0.0
    curiosity_bonus = 0.1 if "..." in text or "?" in text else 0
    contrast_bonus = 0.1 if "but" in lower or "actually" in lower else 0
    # Exclamation bonus (excitement)
    excl_bonus = 0.05 if text.endswith("!") else 0.0

    score = (length_score * 0.5) + (hook_score * 0.35) + question_bonus + excl_bonus + curiosity_bonus + contrast_bonus
    return round(min(score, 1.0), 3)

# ──────────────────────────────────────────────────────────────────────────────
# Parse batch LLM output into individual scripts
# ──────────────────────────────────────────────────────────────────────────────

def _parse_batch(raw: str) -> List[str]:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    scripts = []
    for line in lines:
        repaired = _repair_script(line)
        valid, reason = _is_valid(repaired)
        if valid:
            scripts.append(repaired)
            log_filter(f"ACCEPTED: {repaired!r}")
        else:
            log_filter(f"REJECTED ({reason}): {line!r}")
    return scripts

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_scripts(topic: str, n: int = None) -> List[str]:
    """
    Generate, validate, score and rank scripts for a topic.
    ALWAYS returns at least one script (uses fallbacks if necessary).

    Parameters
    ----------
    topic : str
        The video topic / keyword.
    n : int
        Number of scripts to request. Defaults to cfg.SCRIPT_BATCH_COUNT.

    Returns
    -------
    List[str]
        Scripts sorted best-first by score.
    """
    if n is None:
        n = cfg.SCRIPT_BATCH_COUNT

    log_llm(f"Generating {n} scripts for topic: {topic!r}")

    scripts: List[str] = []

    # ── Attempt 1: batch request ───────────────────────────────────────────
    raw = _call_llm(_build_batch_prompt(topic, n))
    if raw:
        scripts = _parse_batch(raw)

    # ── Attempt 2: single request fallback (if batch gave nothing) ────────
    if not scripts:
        log_llm("Batch gave no valid scripts — trying single request…", level="warning")
        raw = _call_llm(_build_single_prompt(topic))
        if raw:
            repaired = _repair_script(raw.splitlines()[0] if raw.splitlines() else raw)
            valid, reason = _is_valid(repaired)
            if valid:
                scripts = [repaired]
            else:
                log_filter(f"Single request also invalid ({reason}) — will use fallback")

    # ── Attempt 3: hard-coded fallbacks ───────────────────────────────────
    if not scripts:
        log_error("LLM produced no usable script — switching to built-in fallbacks")
        scripts = list(cfg.SCRIPT_FALLBACKS[:n])

    # ── Score + rank ───────────────────────────────────────────────────────
    enhanced = []

    for s in scripts:
        improved = _improve_script(s)
        mutated = _mutate_hook(improved)

        enhanced.append(improved)
        enhanced.append(mutated)

    # remove duplicates
    enhanced = list(set(enhanced))

    scored = [(s, _score_script(s)) for s in enhanced]
    scored.sort(key=lambda x: x[1], reverse=True)

    for s, sc in scored:
        log_score(f"{sc:.3f}  {s!r}")

    best = [s for s, _ in scored]
    log_selected(f"Best script: {best[0]!r}")
    return best


def select_best_script(scripts: List[str]) -> str:
    """
    From an already-generated list, return the highest-scoring one.
    Convenience wrapper used by the GUI.
    """
    if not scripts:
        return random.choice(cfg.SCRIPT_FALLBACKS)
    scored = sorted(scripts, key=_score_script, reverse=True)
    return scored[0]


def normalize_script(text: str) -> str:
    """
    Normalize any user-supplied text (manual mode) into the same single-script
    internal format.  Multi-line input is joined into one sentence.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = " ".join(lines)
    return _repair_script(joined)