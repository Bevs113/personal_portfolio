#!/usr/bin/env python3
"""
scanner.py - One-file police scanner transcriber.

No venv, no manual pip install. Just run it:
    python scanner.py "https://your-stream-url"
    python scanner.py "https://your-stream-url" small   (bigger model, more accurate, slower)

First run installs faster-whisper automatically (one-time). ffmpeg still
needs to be installed as a system tool -- this script will tell you how if
it's missing. Everything else runs locally; audio never leaves your machine.
"""

import glob
import importlib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


def ensure_package(pip_name: str, import_name: str = None):
    import_name = import_name or pip_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[setup] Installing '{pip_name}' (first run only, may take a minute)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name])


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH. Install it, then re-run this script:")
        print("  Windows : winget install Gyan.FFmpeg")
        print("  macOS   : brew install ffmpeg")
        print("  Linux   : sudo apt install ffmpeg")
        sys.exit(1)


ensure_package("faster-whisper", "faster_whisper")
check_ffmpeg()

from faster_whisper import WhisperModel  # noqa: E402


def start_ffmpeg_capture(stream_url: str, chunk_dir: str, chunk_seconds: int) -> subprocess.Popen:
    os.makedirs(chunk_dir, exist_ok=True)
    for leftover in glob.glob(os.path.join(chunk_dir, "chunk_*.wav")):
        try:
            os.remove(leftover)
        except OSError:
            pass
    pattern = os.path.join(chunk_dir, "chunk_%06d.wav")
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "-referer", "https://www.broadcastify.com/",
        "-i", stream_url,
        "-ac", "1", "-ar", "16000",
        "-f", "segment", "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1",
        pattern,
    ]
    return subprocess.Popen(cmd)


def pending_chunks(chunk_dir: str):
    files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.wav")))
    return files[:-1] if len(files) > 1 else []


def load_model(model_size: str):
    try:
        return WhisperModel(model_size, device="cuda", compute_type="float16"), "cuda"
    except Exception:
        return WhisperModel(model_size, device="cpu", compute_type="int8"), "cpu"


def main():
    stream_url = sys.argv[1] if len(sys.argv) > 1 else input("Scanner stream URL: ").strip()
    model_size = sys.argv[2] if len(sys.argv) > 2 else "tiny"
    chunk_seconds = 20
    chunk_dir = "chunks"
    log_path = "transcript.log"

    model, device = load_model(model_size)
    print(f"[setup] Whisper '{model_size}' loaded on {device}.")

    ffmpeg_proc = start_ffmpeg_capture(stream_url, chunk_dir, chunk_seconds)
    print(f"Listening... transcript prints below and saves to {log_path}. Ctrl+C to stop.\n")

    seen = set()
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            while True:
                for chunk_path in pending_chunks(chunk_dir):
                    if chunk_path in seen:
                        continue
                    seen.add(chunk_path)

                    size = os.path.getsize(chunk_path)
                    print(f"[chunk] {os.path.basename(chunk_path)} ({size} bytes)")

                    if size == 0:
                        print("[chunk]   -> 0 bytes, skipping")
                        try:
                            os.remove(chunk_path)
                        except OSError:
                            pass
                        continue

                    try:
                        segments, _ = model.transcribe(
                            chunk_path,
                            language="en",
                            vad_filter=True,
                            vad_parameters=dict(threshold=0.3, min_silence_duration_ms=300),
                        )
                        segments = list(segments)
                    except Exception as exc:
                        print(f"[chunk]   -> couldn't decode this chunk, skipping ({exc})")
                        try:
                            os.remove(chunk_path)
                        except OSError:
                            pass
                        continue

                    print(f"[chunk]   -> {len(segments)} speech segment(s) detected")

                    text = " ".join(seg.text.strip() for seg in segments).strip()

                    if text:
                        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}"
                        print(line)
                        log_file.write(line + "\n")
                        log_file.flush()
                    else:
                        print("[chunk]   -> no text produced")

                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass

                time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        ffmpeg_proc.terminate()
        try:
            ffmpeg_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffmpeg_proc.kill()


if __name__ == "__main__":
    main()