import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained("whisper_model")
model = WhisperForConditionalGeneration.from_pretrained("whisper_model").to(device)

predicted_ids = model.generate(input_features,
                               do_sample=True,
                               temperature=0.7,
                               num_beams=5,
                               no_repeat_ngram_size=2,
                               max_new_tokens=256)
