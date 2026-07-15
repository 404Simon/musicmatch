import os
import torch
import numpy as np
import transformers
import huggingface_hub
from transformers import AutoProcessor, ClapAudioModelWithProjection, ClapTextModelWithProjection
from musicmatch.config import MODEL_NAME, SAMPLE_RATE
from musicmatch.debug import debug, rss

transformers.logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_error()

torch.set_num_threads(os.cpu_count() or 8)

_audio_model: ClapAudioModelWithProjection | None = None
_text_model: ClapTextModelWithProjection | None = None
_processor: AutoProcessor | None = None


def _load(load_text: bool = False):
    global _audio_model, _text_model, _processor
    if _audio_model is not None:
        return
    debug(f"Loading audio model {MODEL_NAME}... (RSS: {rss()})", tag="model")
    _audio_model = ClapAudioModelWithProjection.from_pretrained(MODEL_NAME)
    _audio_model.eval()
    debug(f"Audio model loaded. (RSS: {rss()})", tag="model")
    if load_text:
        debug(f"Loading text model {MODEL_NAME}... (RSS: {rss()})", tag="model")
        _text_model = ClapTextModelWithProjection.from_pretrained(MODEL_NAME)
        _text_model.eval()
        debug(f"Text model loaded. (RSS: {rss()})", tag="model")
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    debug(f"Processor loaded. (RSS: {rss()})", tag="model")


def get_text_embedding(text: str) -> np.ndarray:
    _load(load_text=True)
    inputs = _processor(text=text, return_tensors="pt")
    with torch.inference_mode():
        outputs = _text_model(**inputs)
    return outputs.text_embeds.squeeze().numpy()


def get_audio_embeddings(audios: list[np.ndarray]) -> np.ndarray:
    _load(load_text=False)
    inputs = _processor(
        audio=audios, return_tensors="pt", sampling_rate=SAMPLE_RATE
    )
    with torch.inference_mode():
        outputs = _audio_model(**inputs)
    return outputs.audio_embeds.numpy()
