import numpy as np
import librosa
from musicmatch.config import SAMPLE_RATE, CHUNK_SAMPLES, CHUNK_SECONDS


def load_and_chunk(filepath: str) -> list[tuple[int, float, np.ndarray]]:
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
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
