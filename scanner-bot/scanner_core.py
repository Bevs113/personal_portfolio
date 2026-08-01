"""
scanner_core.py - Shared logic for the police scanner transcriber.

Imported by both scanner_cli.py (terminal version) and
scanner_streamlit.py (web UI version). Not meant to be run directly.
"""

import glob
import importlib
import os
import shutil
import subprocess
import sys
from datetime import datetime


# Preset scanner streams shared by both front-ends. Add more entries here
# as "Name": "stream URL" to expand the list everywhere at once.
STREAM_PRESETS = {
    "Grand Junction, CO": "https://audio.junctionnow.com:8000/radio.mp3",
    "Colorado Springs, CO": "https://countypolicescanner.com/wp-content/uploads/2026/01/Coffee-County-Police-Scanner.mp3",
}


def ensure_package(pip_name: str, import_name: str = None):
    """Install a pip package on first use if it isn't already importable."""
    import_name = import_name or pip_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[setup] Installing '{pip_name}' (first run only, may take a minute)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name])


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def require_ffmpeg():
    """CLI-style hard check: print install instructions and exit if missing."""
    if not check_ffmpeg():
        print("ffmpeg not found on PATH. Install it, then re-run this script:")
        print("  Windows : winget install Gyan.FFmpeg")
        print("  macOS   : brew install ffmpeg")
        print("  Linux   : sudo apt install ffmpeg")
        sys.exit(1)


def load_model(model_size: str):
    """Load a faster-whisper model, preferring GPU and falling back to CPU."""
    from faster_whisper import WhisperModel  # imported here so callers control install timing

    try:
        return WhisperModel(model_size, device="cuda", compute_type="float16"), "cuda"
    except Exception:
        return WhisperModel(model_size, device="cpu", compute_type="int8"), "cpu"


def start_ffmpeg_capture(stream_url: str, chunk_dir: str, chunk_seconds: int) -> subprocess.Popen:
    """Launch ffmpeg to split a live stream into sequential WAV chunks."""
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
    """Return finished chunk files (the newest one is still being written, so skip it)."""
    files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.wav")))
    return files[:-1] if len(files) > 1 else []


def stop_ffmpeg_capture(ffmpeg_proc: subprocess.Popen):
    """Terminate an ffmpeg capture process cleanly."""
    ffmpeg_proc.terminate()
    try:
        ffmpeg_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        ffmpeg_proc.kill()


def chunk_index_from_path(chunk_path: str) -> int:
    """Extract the numeric sequence index from a chunk filename like chunk_000004.wav."""
    base = os.path.basename(chunk_path)
    digits = "".join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else 0


def segment_lines(segments, stream_start_time: float, chunk_path: str, chunk_seconds: int):
    """
    Turn one chunk's faster-whisper segments into individual timestamped lines,
    one per segment, instead of one merged line per chunk.

    Each segment carries its own start offset (in seconds) within the chunk;
    combined with the chunk's position in the stream, this gives each line a
    timestamp close to when that speech actually happened, rather than
    stamping an entire 20-second chunk (which may contain several separate
    transmissions) with a single time.

    Returns a list of (datetime, text) tuples, oldest first.
    """
    chunk_index = chunk_index_from_path(chunk_path)
    chunk_offset = chunk_index * chunk_seconds
    lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        absolute_time = stream_start_time + chunk_offset + seg.start
        lines.append((datetime.fromtimestamp(absolute_time), text))
    return lines