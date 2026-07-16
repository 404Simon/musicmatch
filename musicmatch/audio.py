import subprocess

import numpy as np

from musicmatch.config import SAMPLE_RATE, CHUNK_SAMPLES, CHUNK_SECONDS


def load_audio(filepath: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-i", filepath,
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(sr),
        "-hide_banner",
        "-loglevel", "error",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if len(audio) == 0:
        raise RuntimeError(f"ffmpeg produced no audio data")
    return audio


def chunk_audio(audio: np.ndarray) -> list[tuple[int, float, np.ndarray]]:
    chunks: list[tuple[int, float, np.ndarray]] = []
    num_chunks = (len(audio) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    for i in range(num_chunks):
        start = i * CHUNK_SAMPLES
        end = start + CHUNK_SAMPLES
        chunk = audio[start:end]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        start_time = i * CHUNK_SECONDS
        chunks.append((i, float(start_time), chunk))
    return chunks


def load_and_chunk(filepath: str) -> list[tuple[int, float, np.ndarray]]:
    audio = load_audio(filepath)
    return chunk_audio(audio)
