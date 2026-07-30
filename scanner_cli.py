#!/usr/bin/env python3
"""
scanner_cli.py - Terminal front-end for the police scanner transcriber.

Shared logic lives in scanner_core.py (keep both files together).

No venv, no manual pip install. Just run it:
    python scanner_cli.py
    python scanner_cli.py "" small   (bigger model, more accurate, slower)
    python scanner_cli.py "https://your-stream-url"   (skip the menu, use a direct URL)

First run installs faster-whisper automatically (one-time). ffmpeg still
needs to be installed as a system tool -- this script will tell you how if
it's missing. Everything else runs locally; audio never leaves your machine.
"""

import os
import sys
import time

import scanner_core as core

core.ensure_package("faster-whisper", "faster_whisper")
core.require_ffmpeg()


def choose_stream_url() -> str:
    """Show a menu of preset streams, or use argv[1] if given."""
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()

    presets = list(core.STREAM_PRESETS.items())

    print("Select a scanner stream:")
    for i, (name, _url) in enumerate(presets, start=1):
        print(f"  {i}. {name}")
    custom_choice = len(presets) + 1
    print(f"  {custom_choice}. Enter a custom URL")

    while True:
        raw = input(f"Choice [1-{custom_choice}]: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(presets):
                name, url = presets[idx - 1]
                print(f"[setup] Using {name}: {url}")
                return url
            if idx == custom_choice:
                return input("Stream URL: ").strip()
        print(f"Please enter a number between 1 and {custom_choice}.")


def main():
    stream_url = choose_stream_url()
    model_size = sys.argv[2] if len(sys.argv) > 2 else "tiny"
    chunk_seconds = 20
    chunk_dir = "chunks"
    log_path = "transcript.log"

    model, device = core.load_model(model_size)
    print(f"[setup] Whisper '{model_size}' loaded on {device}.")

    ffmpeg_proc = core.start_ffmpeg_capture(stream_url, chunk_dir, chunk_seconds)
    stream_start_time = time.time()
    print(f"Listening... transcript prints below and saves to {log_path}. Ctrl+C to stop.\n")

    seen = set()
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            while True:
                for chunk_path in core.pending_chunks(chunk_dir):
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

                    lines = core.segment_lines(segments, stream_start_time, chunk_path, chunk_seconds)
                    if lines:
                        for ts, text in lines:
                            line = f"[{ts.strftime('%Y-%m-%d %H:%M:%S')}] {text}"
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
        core.stop_ffmpeg_capture(ffmpeg_proc)


if __name__ == "__main__":
    main()