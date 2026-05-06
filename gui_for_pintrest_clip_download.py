import tkinter as tk
from tkinter import messagebox
import subprocess
import threading

DOWNLOAD_PATH = r"C:\Users\punya\Project\yt bot v3\clips"

def start_download(event=None):
    url = entry.get().strip()

    if not url:
        return

    status_label.config(text="Downloading...", fg="orange")

    def run_download():
        try:
            cmd = [
                "yt-dlp",
                "-o", f"{DOWNLOAD_PATH}\\%(id)s_%(title)s.%(ext)s",
                url
            ]

            subprocess.run(cmd, check=True)

            # ✅ Update UI safely from main thread
            root.after(0, on_complete)

        except Exception as e:
            root.after(0, lambda: status_label.config(text="Download Failed ❌", fg="red"))
            print(e)

    threading.Thread(target=run_download, daemon=True).start()


def on_complete():
    status_label.config(text="Download Complete ✅", fg="green")

    # 🔥 Select entire text
    entry.select_range(0, tk.END)
    entry.focus()


# GUI
root = tk.Tk()
root.title("YT-DLP Downloader")
root.geometry("420x200")

entry = tk.Entry(root, width=55)
entry.pack(pady=20)

# 🔥 ENTER key triggers download
entry.bind("<Return>", start_download)

# 🔥 PASTE (Ctrl+V) triggers download
def on_paste(event):
    root.after(100, start_download)  # wait for paste to complete

entry.bind("<Control-v>", on_paste)

btn = tk.Button(root, text="Download", command=start_download)
btn.pack(pady=10)

status_label = tk.Label(root, text="")
status_label.pack(pady=10)

root.mainloop()