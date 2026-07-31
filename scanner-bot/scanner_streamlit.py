"""
scanner_streamlit.py - Web UI front-end for the police scanner transcriber.
"""

import queue
import threading
import time
import streamlit as st
import scanner_core as core

st.set_page_config(
    page_title="Live Scanner Transcriber",
    page_icon="🎙️",
    layout="wide"
)

CUSTOM_OPTION = "Custom URL..."
FEED_HEIGHT = 420

core.ensure_package("faster-whisper", "faster_whisper")


@st.cache_resource
def load_model_cached(model_size: str):
    """Caches the Whisper model in memory across reruns."""
    model, device = core.load_model(model_size)
    label = "CUDA (GPU)" if device == "cuda" else "CPU"
    return model, label


def render_feed_html(lines):
    import html as html_lib
    if lines:
        rows = "".join(
            '<div style="padding:4px 0;border-bottom:1px solid #333;color:#e0e0e0;">'
            f'<span style="color:#888;">[{html_lib.escape(ts)}]</span> '
            f'{html_lib.escape(text)}</div>'
            for ts, text in lines
        )
    else:
        rows = '<div style="color:#888;">No audio transcribed yet. Click \'Start Listening\' to begin.</div>'

    return f'<div style="height:{FEED_HEIGHT}px;overflow-y:auto;">{rows}</div>'

def render_live_audio_player(stream_url: str):
    """Renders a custom HTML5 audio player that forces playback when triggered."""
    return f"""
    <div style="background: #1e1e1e; padding: 12px; border-radius: 8px; margin-bottom: 15px;">
        <audio id="scanner-audio-player" controls style="width: 100%; height: 40px;">
            <source src="{stream_url}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
    </div>
    <script>
        var player = document.getElementById('scanner-audio-player');
        if (player) {{
            player.play().catch(function(error) {{
                console.log("Autoplay prevented by browser, manual play required:", error);
            }});
        }}
    </script>
    """

def background_transcriber(model, stream_url, chunk_seconds, result_queue, stop_event):
    """Worker thread: runs FFmpeg and Whisper without blocking the Streamlit UI."""
    import os
    chunk_dir = "chunks"
    log_path = "transcript.log"
    seen = set()

    ffmpeg_proc = core.start_ffmpeg_capture(stream_url, chunk_dir, chunk_seconds)
    stream_start_time = time.time()

    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            while not stop_event.is_set():
                for chunk_path in core.pending_chunks(chunk_dir):
                    if chunk_path in seen:
                        continue
                    seen.add(chunk_path)

                    if os.path.getsize(chunk_path) == 0:
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
                    except Exception:
                        try:
                            os.remove(chunk_path)
                        except OSError:
                            pass
                        continue

                    lines = core.segment_lines(segments, stream_start_time, chunk_path, chunk_seconds)
                    for ts, text in lines:
                        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                        result_queue.put((ts_str, text))
                        log_file.write(f"[{ts_str}] {text}\n")
                    
                    if lines:
                        log_file.flush()

                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass

                time.sleep(1)
    finally:
        core.stop_ffmpeg_capture(ffmpeg_proc)


def main():
    st.title("🎙️ Real-Time Scanner Transcriber")
    st.markdown("---")

    if not core.check_ffmpeg():
        st.error("⚠️ **FFmpeg not found!** Please install FFmpeg on your system before continuing.")
        return

    # Session state
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "transcripts" not in st.session_state:
        st.session_state.transcripts = []
    if "transcript_queue" not in st.session_state:
        st.session_state.transcript_queue = queue.Queue()
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()

    # Sidebar Controls
    st.sidebar.header("Configuration")
    dropdown_options = list(core.STREAM_PRESETS.keys()) + [CUSTOM_OPTION]
    selected_preset = st.sidebar.selectbox("Scanner Stream", dropdown_options, index=0)

    if selected_preset == CUSTOM_OPTION:
        stream_url = st.sidebar.text_input("Stream URL", placeholder="https://your-stream-url").strip().rstrip(".")
    else:
        stream_url = core.STREAM_PRESETS[selected_preset]
        st.sidebar.caption(f"Using: {stream_url}")

    model_size = st.sidebar.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large"], index=0)
    chunk_seconds = st.sidebar.slider("Chunk Duration (seconds)", min_value=5, max_value=60, value=20)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Controls")

        if not st.session_state.is_running:
            if st.button("▶️ Start Listening", type="primary"):
                if not stream_url:
                    st.warning("Please enter a stream URL first.")
                else:
                    st.session_state.stop_event.clear()
                    st.session_state.is_running = True
                    # Launch background worker
                    model, _ = load_model_cached(model_size)
                    t = threading.Thread(
                        target=background_transcriber,
                        args=(
                            model,
                            stream_url,
                            chunk_seconds,
                            st.session_state.transcript_queue,
                            st.session_state.stop_event,
                        ),
                        daemon=True
                    )
                    t.start()
                    st.rerun()
        else:
            if st.button("⏹️ Stop Listening", type="secondary"):
                st.session_state.stop_event.set()
                st.session_state.is_running = False
                st.rerun()

        if st.session_state.is_running:
            st.success("🟢 Scanner Active")
        else:
            st.warning("🔴 Scanner Stopped")

        if st.button("🗑️ Clear Transcript View"):
            st.session_state.transcripts = []
            st.rerun()

    # Pull items from queue into session state
    while not st.session_state.transcript_queue.empty():
        item = st.session_state.transcript_queue.get()
        st.session_state.transcripts.append(item)

    with col2:
            # Add Live Audio Stream Player
            if stream_url:
                st.subheader("Live Audio Stream")
                st.audio(stream_url, format="audio/mp3")

            st.subheader("Live Transcript Feed")
            transcript_placeholder = st.empty()
            with transcript_placeholder.container():
                st.html(render_feed_html(st.session_state.transcripts))

    # Auto-refresh UI every 2 seconds if running to show incoming lines
    if st.session_state.is_running:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()