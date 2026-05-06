import asyncio
import json
import os
from tts_engine import generate_tts
from part3 import make_video

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(ROOT, "output", "script.json")
async def run_pipeline():
    print(" Rendering video...")
    await make_video()
    print("Pipeline complete")
if __name__ == "__main__":
    asyncio.run(run_pipeline())
