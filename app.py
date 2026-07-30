import glob
import importlib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
import streamlit as st

# Set page layout and title
st.set_page_config(
    page_title="Live Scanner Transcriber",
    page_icon="🎙️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Dependency Checks
# -----------------------------------------------------------------------------
def check_ffmpeg():
    return shutil.which("ffmpeg") is not None

def ensure_package(pip_name: str, import_name: str = None):
    import_name = import_name or pip_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name])

ensure_package("faster-whisper", "faster_whisper")
from faster_whisper import WhisperModel

# -----------------------------------------------------------------------------
# Core Helper Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model(model_size: str):
    """Caches the Whisper model in memory so it doesn't reload on every UI refresh."""
    try:
        return WhisperModel(model_size, device="cuda", compute_type="float16"), "CUDA (GPU)"
    except Exception:
        return WhisperModel(model_size, device="cpu", compute_type="int8"), "CPU"

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

# -----------------------------------------------------------------------------
# UI Layout & App Logic
# -----------------------------------------------------------------------------
def main():
    st.title("🎙️ Real-Time Scanner Transcriber")
    st.markdown("---")

    # Check FFmpeg dependency
    if not check_ffmpeg():
        st.error("⚠️ **FFmpeg not found!** Please install FFmpeg on your system before continuing.")
        st.info("Windows: `winget install Gyan.FFmpeg` | macOS: `brew install ffmpeg` | Linux: `sudo apt install ffmpeg`")
        return

    # Sidebar Options
    st.sidebar.header("Configuration")
    stream_url = st.sidebar.text_input("Stream URL", placeholder="https://your-stream-url").strip().rstrip(".")
    model_size = st.sidebar.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large"], index=0)
    chunk_seconds = st.sidebar.slider("Chunk Duration (seconds)", min_value=5, max_value=60, value=20)

    # Session State Initialization
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "transcripts" not in st.session_state:
        st.session_state.transcripts = []

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Controls")
        
        # Start/Stop Buttons
        if not st.session_state.is_running:
            if st.button("▶️ Start Listening", type="primary"):
                if not stream_url:
                    st.warning("Please enter a stream URL first.")
                else:
                    st.session_state.is_running = True
                    st.rerun()
        else:
            if st.button("⏹️ Stop Listening", type="secondary"):
                st.session_state.is_running = False
                st.rerun()

        # Status Display
        if st.session_state.is_running:
            st.success("🟢 Scanner Active")
        else:
            st.warning("🔴 Scanner Stopped")

        # Clear Logs Button
        if st.button("🗑️ Clear Transcript View"):
            st.session_state.transcripts = []
            st.rerun()

    with col2:
        st.subheader("Live Transcript Feed")
        transcript_container = st.container(height=400)

    # Audio Processing Loop
    if st.session_state.is_running:
        with st.spinner("Loading Whisper Model..."):
            model, compute_device = load_model(model_size)

        st.sidebar.caption(f"Engine running on: **{compute_device}**")

        chunk_dir = "chunks"
        log_path = "transcript.log"
        seen = set()

        # Start background FFmpeg process
        ffmpeg_proc = start_ffmpeg_capture(stream_url, chunk_dir, chunk_seconds)

        try:
            with open(log_path, "a", encoding="utf-8") as log_file:
                while st.session_state.is_running:
                    for chunk_path in pending_chunks(chunk_dir):
                        if chunk_path in seen:
                            continue
                        seen.add(chunk_path)

                        size = os.path.getsize(chunk_path)
                        if size == 0:
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
                            try:
                                os.remove(chunk_path)
                            except OSError:
                                pass
                            continue

                        text = " ".join(seg.text.strip() for seg in segments).strip()

                        if text:
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            line = f"[{timestamp}] {text}"

                            st.session_state.transcripts.append(line)
                            log_file.write(line + "\n")
                            log_file.flush()

                            # Update transcript view in real-time
                            with transcript_container:
                                st.write("\n".join(st.session_state.transcripts[::-1]))

                        try:
                            os.remove(chunk_path)
                        except OSError:
                            pass

                    time.sleep(2)
        finally:
            # Cleanup process on stop or exit
            ffmpeg_proc.terminate()
            try:
                ffmpeg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ffmpeg_proc.kill()
    else:
        # Display saved transcript history when stopped
        with transcript_container:
            if st.session_state.transcripts:
                st.write("\n".join(st.session_state.transcripts[::-1]))
            else:
                st.info("No audio transcribed yet. Click 'Start Listening' to begin.")

if __name__ == "__main__":
    main()