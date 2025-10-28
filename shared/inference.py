import torch
import torchaudio
from scipy.special import softmax
import numpy as np
from pathlib import Path
from typing import List, Dict
from .utils import apply_random_segment_extraction


class AntiSpoofingModel:
    def __init__(self, model_path: str = "weights/torch_script_weigths.pt", device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.target_sample_rate = 16000

    def _load_model(self, model_path: str) -> torch.jit.ScriptModule:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model

    def predict(self, audio_path: str) -> List[Dict]:
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        results = []

        for channel_idx in range(waveform.shape[0]):
            channel_audio = waveform[channel_idx:channel_idx + 1]
            processed_audio = apply_random_segment_extraction(channel_audio)
            processed_audio = processed_audio.unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(processed_audio)
                logits_np = logits.cpu().numpy().flatten()

            probabilities = softmax(logits_np)
            prediction_class = np.argmax(probabilities)
            confidence = float(probabilities[prediction_class])

            results.append({
                'channel': channel_idx,
                'prediction': 'REAL' if prediction_class == 1 else 'FAKE',
                'confidence': confidence,
                'fake_prob': float(probabilities[0]),
                'real_prob': float(probabilities[1]),
                'logits': logits_np.tolist()
            })

        return results
