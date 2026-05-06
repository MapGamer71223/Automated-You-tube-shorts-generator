from email.mime import text
import sys
import json
import os
import subprocess
import threading
import random
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QTextEdit, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox
from script_engine import generate_scripts
from script_engine import normalize_script

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "output")
SCRIPT_PATH = os.path.join(OUTPUT_DIR, "script.json")
critic_cache = {}
import requests
def log(stage, msg):
    print(f"[{stage}] {msg}")
def load_history(n=20):
    try:
        with open("history.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
            return [json.loads(l)["script"] for l in lines]
    except:
        return []
    
def extract_patterns(scripts):
    hooks = []
    endings = []
    structures = []

    for s in scripts:
        if len(s) < 2:
            continue

        hooks.append(s[0])
        endings.append(s[-1])

        # capture pattern shape
        structures.append([
            len(line.split()) for line in s
        ])

    return {
        "hooks": hooks[:5],
        "endings": endings[:5],
        "structures": structures[:5]
    }
    
def critique_script(script):
    key = tuple(script)

    if key in critic_cache:
        return critic_cache[key]

    url = "http://localhost:1234/v1/chat/completions"

    prompt = """You are a viral content expert.

Rate this script (1–10) based on:
- Hook strength
- Curiosity gap
- Emotional tension
- Loop effectiveness

Script:
{}

Return ONLY number.""".format("\n".join(script))

    payload = {
        "model": "mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 10
    }

    try:
        res = requests.post(url, json=payload)
        data = res.json()["choices"][0]["message"]["content"]
        score = int(''.join(filter(str.isdigit, data)))
    except:
        score = 5

    critic_cache[key] = score
    return score
    
def combined_score(script):
    base = score_script(script)
    critic = critique_script(script)
    return (base * 1.5) + critic
def save_script(script, meta):
    with open("history.jsonl", "a", encoding="utf-8") as f:
        json.dump({
            "script": script,
            "meta": meta
        }, f)
        f.write("\n")

def score_script(script):
    score = 0

    if not script:
        return 0

    text = " ".join(script).lower()
    first = script[0].lower()

    # curiosity triggers
    triggers = ["don't", "won’t", "never", "why", "actually", "nobody"]
    if any(t in first for t in triggers):
        score += 4

    # short hook = strong
    if len(first.split()) <= 5:
        score += 3

    # emotional tension
    if any(w in text for w in ["miss", "feel", "confusing", "strange", "notice"]):
        score += 2

    # pattern interrupt words
    if any(w in first for w in ["nobody", "most people", "actually"]):
        score += 3

    return score

def mutate_hook(script):
    if not script:
        return script

    patterns = [
        "They don’t say no directly",
        "It sounds like yes… but it isn’t",
        "You think they agreed… they didn’t",
        "This is how rejection actually sounds",
    ]

    return [random.choice(patterns)] + script[1:]
def improve_script(script):
    """Remove filler phrases and tighten, but never truncate meaning."""
    FILLERS = [
        "you know", "basically", "kind of", "sort of",
        "i mean", "like,", "actually,", "so,", "well,"
    ]
    improved = []
    for line in script:
        low = line.lower()
        for f in FILLERS:
            low = low.replace(f, "")
        # Re-capitalise first char
        line = low.strip()
        if line:
            line = line[0].upper() + line[1:]
        improved.append(line)

    # Strengthen hook only if it's genuinely weak (no tension trigger)
    TENSION_WORDS = ["don't", "won't", "never", "actually", "nobody", "think", "wrong", "isn't", "they"]
    hook = improved[0] if improved else ""
    if not any(t in hook.lower() for t in TENSION_WORDS):
        hook = random.choice([
            "You think you understood this.",
            "Nobody talks about what this actually means.",
            "This isn't what it looks like.",
            "You missed the real signal.",
        ])
        improved[0] = hook

    return improved

def get_hook_templates():
    return [
        "They don’t say no directly",
        "It sounds like yes… but it isn’t",
        "You think they agreed… they didn’t",
        "This is how rejection actually sounds",
        "Politeness is how they reject you",
        "You misunderstood what they meant",
        "This is where you got it wrong"
    ]

def generate_metadata(script):
    url = "http://localhost:1234/v1/chat/completions"

    prompt = f"""Create a YouTube Shorts title and description.

Script:
{chr(10).join(script)}

Rules:
- Title must be catchy and curiosity-driven
- Max 60 characters
- Description 1–2 lines max
- Add 3–5 hashtags

Return format:
TITLE:
...
DESCRIPTION:
...
"""

    payload = {
        "model": "mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.8,
        "max_tokens": 100
    }

    try:
        res = requests.post(url, json=payload)
        data = res.json()["choices"][0]["message"]["content"]

        return data.strip()
    except:
        return "Failed to generate metadata"
def generate_topics():
    return [
        "why japanese girls avoid eye contact when they like you",
        "why japanese people never say no directly",
        "hidden habits in japan tourists don’t notice",
        "why silence in japan feels different",
        "why japanese dating feels confusing"
    ]
    
def generate_script_from_llm(prompt, style):
    history = load_history()
    patterns = extract_patterns(history)

    pattern_text = ""


    if patterns["hooks"]:
        pattern_text += "\nHook styles (examples):\n"
        pattern_text += "\n".join([f"- {h}" for h in patterns["hooks"]])

    if patterns["endings"]:
        pattern_text += "\n\nEnding styles (loop patterns):\n"
        pattern_text += "\n".join([f"- {e}" for e in patterns["endings"]])

    if patterns["structures"]:
        pattern_text += "\n\nStructure patterns (word counts per line):\n"
        pattern_text += "\n".join([f"- {s}" for s in patterns["structures"]])

    hook_templates = get_hook_templates()
    pattern_text += "\n\nHook templates:\n"
    pattern_text += "\n".join([f"- {h}" for h in hook_templates])

    if pattern_text.strip():
        pattern_text = "Learn and adapt these patterns:\n" + pattern_text
    url = "http://localhost:1234/v1/chat/completions"

    system_prompt = f"""
You are a TOP 0.1% viral YouTube Shorts script writer.

Your scripts must feel like something people WANT to replay.

INPUT TOPIC:
{prompt}

STYLE:
{style}

{pattern_text}

----------------------------

STRICT RULES:

1. HOOK = FIRST 5 WORDS MUST BREAK EXPECTATION
   - must create confusion, contradiction, or curiosity
   - NO generic starts like "Did you know"

2. WRITE LIKE A HUMAN THOUGHT
   - not explanation
   - not educational
   - feels like realization

3. BUILD PSYCHOLOGICAL TENSION
   - make viewer feel they misunderstood something
   - use subtle emotional triggers

4. REVEAL SOMETHING SPECIFIC
   - not general facts
   - must feel like insider knowledge

5. LOOP ENDING
   - last sentence must connect back to hook
   - should make viewer rethink first line

----------------------------

STYLE CONTROL:

- Flirty → teasing, attractive, slightly playful
- Shy → soft, indirect, subtle tension
- Psychological → mind-focused, observational
- Bold → direct, confident, slightly shocking

----------------------------

STRUCTURE:

- Keep it short and punchy.
- natural spoken flow
- no bullet points
- no labels
- no quotes

----------------------------

BAD EXAMPLE:
"They are polite and avoid conflict"

GOOD EXAMPLE:
"They’ll say ‘maybe later’… and you think it’s a yes. It’s not."

----------------------------

OUTPUT:
Return ONLY the script.
"""
    payload = {
        "model": "mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},

            {"role": "user", "content": prompt}
        ],
        "temperature": 0.9,
        "top_p": 0.9,
        "max_tokens": 120
    }
    log("LLM", f"Prompt: {prompt}")
    log("LLM", f"Style: {style}")
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        log("ERROR", f"LLM API failed: {response.text}")
        return []

    data = response.json()
    if "choices" not in data:
        print("INVALID LLM RESPONSE:", data)
        return []

    text = data["choices"][0]["message"]["content"]
    log("LLM", f"Raw output: {text}")

    # 🔥 CLEAN MODEL GARBAGE
    text = text.split("Hook:")[0]
    text = text.split("Ending:")[0]
    text = text.split("Structure:")[0]
    text = text.split("###")[0]

    text = text.strip().strip('"').strip("'")

    # remove newlines → single flow
    text = " ".join(text.split())

    scripts = []
    temperatures = [0.7, 0.9, 1.1]
    log("LLM", f"Cleaned: {text}")
    for i, temp in enumerate(temperatures):
        payload["temperature"] = temp
        log("LLM", f"Generating variation {i} (temp={temp})")
        # 🔥 inject variation signal
        payload["messages"][1]["content"] = f"{prompt}\n\nVariation: {i}"

        response = requests.post(url, json=payload)
        data = response.json()

        if "choices" not in data:
            continue

        text = data["choices"][0]["message"]["content"]

        # 🔥 CLEAN AGAIN (IMPORTANT)
        text = text.split("Hook:")[0]
        text = text.split("Ending:")[0]
        text = text.split("Structure:")[0]
        text = text.split("###")[0]

        text = text.strip().strip('"').strip("'")
        text = " ".join(text.split())
        scripts.append(text)
        log("LLM", f"Raw V{i}: {text}")

        improved = " ".join(improve_script([text]))
        scripts.append(improved)
        log("LLM", f"Improved V{i}: {improved}")

        mutated = " ".join(mutate_hook([text]))
        scripts.append(mutated)
        log("LLM", f"Mutated V{i}: {mutated}")
        log("LLM", f"Variation {i}: {text}")
          
    return scripts
def is_good_script(s):
    words = len(s.split())

    return (
    8 <= words <= 45 and
    any(p in s for p in [".", "…", "!", "?"])
)
class VideoWorker(threading.Thread):
    def __init__(self, status_callback):
        super().__init__(daemon=True)
        self.status_callback = status_callback

    def run(self):
        try:
            print("\n===== PIPELINE STARTED =====")
            self.status_callback("🎬 Launching pipeline...")

            process = subprocess.Popen(
    [sys.executable, "part4.py"],
    cwd=ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="ignore",
    bufsize=1
)

            for line in process.stdout:
                line = line.rstrip()

                # 🔥 LOG TO CMD (FULL OUTPUT)
                print(line)

                # 🔹 LIGHT GUI UPDATE (not every spammy line)
                if any(k in line.lower() for k in ["error", "failed", "complete", "generating", "rendering"]):
                    self.status_callback(line)

            process.wait()

            if process.returncode == 0:
                print("===== PIPELINE SUCCESS =====\n")
                self.status_callback("✅ Video generation completed!")
            else:
                print("===== PIPELINE FAILED =====\n")
                self.status_callback("❌ Pipeline failed (check CMD)")

        except Exception as e:
            print("===== GUI ERROR =====")
            print(e)
            self.status_callback(f"❌ GUI Error: {e}")

class VideoGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beast Shorts Generator")
        self.setFixedSize(520, 420)

        layout = QVBoxLayout()
        self.mode_select = QComboBox()
        self.mode_select.addItems(["🧠 AI Mode", "✍️ Manual Mode"])
        layout.addWidget(self.mode_select)
        title = QLabel("🎬 Shorts Script Input")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:18px;font-weight:bold;")
        layout.addWidget(title)
        self.script_box = QTextEdit()
        self.script_box.setPlaceholderText(
    "🧠 AI Mode → enter topic (e.g. shy japanese girls)\n"
    "✍️ Manual Mode → enter script (1 line per sentence)"
)
        self.script_box.setStyleSheet("font-size:14px;")
        layout.addWidget(self.script_box)

        self.meta_box = QTextEdit()
        self.meta_box.setPlaceholderText("Title & Description will appear here...")
        self.meta_box.setFixedHeight(100)
        layout.addWidget(self.meta_box)
        self.generate_btn = QPushButton("🚀 Generate Video")
        self.generate_btn.clicked.connect(self.on_generate)
        layout.addWidget(self.generate_btn)

        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color:gray;")
        layout.addWidget(self.status_label)
        
        self.style_select = QComboBox()
        self.style_select.addItems([
            "💖 Flirty",
            "😳 Shy",
            "🧠 Psychological",
            "🔥 Bold"
        ])
        layout.addWidget(self.style_select)

        self.ai_generate_btn = QPushButton("⚡ Generate Script Only")
        self.ai_generate_btn.clicked.connect(self.generate_script_only)
        layout.addWidget(self.ai_generate_btn)
        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: white;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border-radius: 10px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #00c853;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00e676;
            }
            QComboBox {
                background-color: #1e1e1e;
                padding: 6px;
            }
            """)
    def set_status(self, text):
        self.status_label.setText(text)

    def on_generate(self):
        mode = self.mode_select.currentText()
        raw = self.script_box.toPlainText().strip()

        if not raw:
            QMessageBox.warning(self, "Empty", "Enter something")
            return

        if "AI" in mode:
            if not raw:
                raw = random.choice(generate_topics())
            elif len(raw.split()) <= 4:
                raw = f"hidden behavior about {raw} that people misunderstand"
            style = self.style_select.currentText()
            scripts = generate_scripts(raw)
            

            if not scripts:
                log("ERROR", "No valid scripts passed filter")

                QMessageBox.warning(self, "Weak Script", "Try again")
                return

            best_script = scripts[0]
            log("SELECTED", best_script)
            meta = generate_metadata([best_script])
          
            self.meta_box.setText(meta)
            display = []

            display.append("🔥 SCRIPT:\n" + best_script)

            self.script_box.setText("\n".join(display))
            import re
            lines = [s.strip() for s in re.split(r'[.!?]', best_script) if s.strip()]
        else:
            # manual mode
            raw_lines = raw.split("\n")
            lines = [s.strip() for s in raw_lines if s.strip()]

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(SCRIPT_PATH, "w", encoding="utf-8") as f:
            json.dump({"script": lines}, f)
        self.set_status("⚙️ Starting pipeline…")
        self.generate_btn.setEnabled(False)

        self.worker = VideoWorker(self.on_worker_update)
        self.worker.start()

    def on_worker_update(self, text):
        self.set_status(text)
        if "completed" in text.lower() or "failed" in text.lower():
            self.generate_btn.setEnabled(True)
    def generate_script_only(self):
        prompt = self.script_box.toPlainText().strip()

        if not prompt:
            QMessageBox.warning(self, "Empty", "Enter topic")
            return

        style = self.style_select.currentText()
        scripts = generate_scripts(prompt)

        if not scripts:
            QMessageBox.warning(self, "Error", "No script generated")
            return

        display = []
        for i, s in enumerate(scripts):
            display.append(f"--- Script {i+1} ---\n{s}")

        self.script_box.setText("\n\n".join(display))
        self.set_status("✅ Pick best script")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VideoGUI()
    gui.show()
    sys.exit(app.exec_())
