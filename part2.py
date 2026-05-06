import json
import os
from PIL import Image, ImageDraw, ImageFont
import textwrap
from moviepy.video.fx import blur

# Emoji keywords for automatic addition to captions
EMOJI_MAP = {
    "love": "😍", "kiss": "😘", "sad": "💔",
    "cute": "🥺", "hot": "🔥", "girl": "👧", 
    "romantic": "✨", "shy": "😳", "wow": "😲",
}

def add_emojis(text):
    for word, emoji in EMOJI_MAP.items():
        if word in text.lower():
            text += " " + emoji
    return text

# Stylized text overlay for frames (used when generating captions)
def add_text_overlay(img, text, index):
    img = img.convert("RGB")
    try:
        font = ImageFont.truetype("arialbd.ttf", size=64)
    except:
        print("⚠️ Font load failed, using default.")
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    wrapped = textwrap.fill(text, width=20)

    bbox = draw.textbbox((0, 0), wrapped, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (img.width - text_width) // 2
    text_y = img.height - text_height - 150

    lines = wrapped.split("\n")
    spacing = 12

    for i, line in enumerate(lines):
        y = text_y + i * (font.size + spacing)
        draw.text((text_x + 2, y + 2), line, font=font, fill="black")  # Shadow
        draw.text((text_x, y), line, font=font, fill="white")         # Main

    out_path = f"output/frame_{index:03d}.png"
    img.save(out_path)
    print(f"✅ Saved stylized frame: {out_path}")
