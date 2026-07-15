import os

SAMPLE_RATE = 48000
CHUNK_SECONDS = 10
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
EMBEDDING_DIM = 512
DB_PATH = os.environ.get("MUSICMATCH_DB_PATH", "music_vectors.db")
MODEL_NAME = "laion/clap-htsat-unfused"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".opus"}
TOP_K = int(os.environ.get("MUSICMATCH_TOP_K", "5"))
MAX_DURATION_MINUTES = int(os.environ.get("MUSICMATCH_MAX_DURATION_MINUTES", "12"))
MPD_MUSIC_DIR = os.environ.get("MUSICMATCH_MPD_MUSIC_DIR", "~/Music")
