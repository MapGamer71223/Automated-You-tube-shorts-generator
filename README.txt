#YouTube Shorts Generator (YT Bot) — Automation Pipeline

Automated pipeline to create AI-assisted YouTube Shorts (9:16) with:
- TTS voiceovers (Edge TTS or custom)
- Auto-generated subtitles
- Motion / blur and image effects
- Optional segmentation/foreground processing (DeepLabV3)
- Video assembly via MoviePy



Features
- Generate TTS audio asynchronously for text segments
- Stitch images / clips into a 9:16 short with captions and motion
- Optional semantic segmentation (DeepLabv3) for foreground/background separation
- Basic effects: blur, overlays, caption rendering, font support



Repo structure (example)

yt-bot/
├── part1.py # TTS generation, prompts, text chunking
├── part2.py # image generation / procession
├── part3.py # video assembly (make_video)
├── fonts/ # fonts used for captions
├── clips/ # short clips / assets
├── output/ # generated videos
├── temp/ # temporary files
├── requirements.txt
├── prompts.json
└── start.bat / run.sh


Setup

1. Create venv & install deps
bash
python -m venv venv
# linux / mac
source venv/bin/activate
# windows
venv\Scripts\activate

pip install -r requirements.txt
Install FFmpeg (system)

Linux: sudo apt install ffmpeg

Windows: download FFmpeg static build and add to PATH

Fonts & assets

Put any custom fonts in fonts/ and reference them in your caption functions.

Put background clips into clips/.

Running the pipeline
Minimal usage (example):

bash
Copy code
python part1.py          # generate TTS files and is the file you need to connect the ollama or lm studio to use the prompts from the prompt.jason to create script
python part3.py          # assemble final video using available assets
Or run your main orchestrator (example):

bash
Copy code
python main.py
Notes / Tips
Edge TTS requires an internet connection to use Microsoft Edge voices. If you want fully offline TTS, integrate a local TTS model (Coqui or others).

Segmentation (DeepLab) uses a pre-trained model from torchvision; first inference may be slow if CPU-only.

Keep temp/ in .gitignore to avoid committed audio/video blobs.

Use tqdm to visualize generation progress in loops.

System requirements
Python 3.10+ recommended

FFmpeg installed and available in PATH

For Torch GPU acceleration: CUDA-compatible GPU and matching torch wheel



Author:
Punyansh Sharma — AI / automation developer



4) Small dev notes & suggestions (copy to README or dev.md)
- Break your pipeline into three clear modules (as you already do):
  - `part1.py` — prepare text chunks & TTS generation (async), save audio files
  - `part2.py` — image/background generation & per-frame assets
  - `part3.py` — assemble audio + frames → final video using MoviePy
- Keep an index file `main.py` that orchestrates:
  1. Load prompts.json  
  2. call `part1.generate_tts` for each chunk (await concurrently)  
  3. prepare visuals (part2)  
  4. call `part3.make_video` to assemble and export
- When using `librosa` for audio features, export stable 16k/22k sample WAV for moviepy compatibility.



5) .gitignore quick content
venv/
pycache/
.pyc
output/
temp/
.wav
.mp4

