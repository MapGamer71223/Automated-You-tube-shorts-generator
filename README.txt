<div align="center">

# AI Shorts Pipeline

```text
Semantic Retrieval • Automated Rendering • AI Video Generation
```

End-to-end automated short-form video generation system built around:
- LLM-generated scripts
- semantic clip retrieval
- subtitle synchronization
- automated pacing
- FFmpeg rendering pipelines

</div>

---

# Pipeline Architecture

```text
topic
  ↓
LLM script generation
  ↓
script scoring + refinement
  ↓
semantic clip retrieval
  ↓
TTS generation
  ↓
Whisper alignment
  ↓
ASS subtitle generation
  ↓
FFmpeg rendering
  ↓
final vertical short
```

---

# Core Systems

## Script Engine

Generates and scores short-form scripts using local LLM workflows with:
- hook mutation
- retention-focused scoring
- metadata generation
- batch generation pipelines

```text
prompt
   ↓
script generation
   ↓
scoring
   ↓
hook refinement
   ↓
final narration
```

**Core files**
- `script_engine.py`
- `prompts.json`

---

## CLIP + FAISS Retrieval Engine

Semantic moment-level retrieval system using:
- CLIP embeddings
- FAISS indexing
- motion-aware reranking
- category-aware scoring
- multi-frame analysis

```text
script line
   ↓
CLIP embedding
   ↓
FAISS similarity search
   ↓
moment retrieval
   ↓
motion reranking
   ↓
best matching segment
```

**Core files**
- `build_index.py`
- `clip_engine.py`
- `clip_sorter.py`

---

## Rendering Pipeline

Custom FFmpeg-based rendering system with:
- automated clip timing
- subtitle compositing
- vertical formatting
- pacing synchronization
- ASS subtitle rendering

```text
audio timing
      ↓
clip sequencing
      ↓
subtitle timing
      ↓
FFmpeg compositing
      ↓
final render
```

**Core files**
- `part3.py`
- `part4.py`

---

## TTS + Alignment System

Voice synthesis and alignment pipeline using:
- Edge TTS
- Whisper alignment
- word-level timestamps
- subtitle synchronization

```text
generated narration
        ↓
speech synthesis
        ↓
word timestamps
        ↓
subtitle alignment
```

**Core files**
- `tts_engine.py`

---

# Features

- Semantic video retrieval
- Automated short generation
- CLIP + FAISS indexing
- Word-level subtitle synchronization
- FFmpeg rendering orchestration
- Retention-focused scripting
- Automated pacing systems

---

# Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-0d1117?style=for-the-badge&logo=python)
![FAISS](https://img.shields.io/badge/FAISS-0d1117?style=for-the-badge)
![CLIP](https://img.shields.io/badge/CLIP-0d1117?style=for-the-badge)
![Whisper](https://img.shields.io/badge/Whisper-0d1117?style=for-the-badge)
![FFmpeg](https://img.shields.io/badge/FFmpeg-0d1117?style=for-the-badge&logo=ffmpeg)
![PyTorch](https://img.shields.io/badge/PyTorch-0d1117?style=for-the-badge&logo=pytorch)

</div>

---

# Running

```bash
pip install -r requirements.txt
python gui.py
```
