import torch
import torchaudio
from scipy.special import softmax
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
import librosa

warnings.filterwarnings('ignore')

from utils.audio_utils import apply_random_segment_extraction


class AntiSpoofingModel:
    def __init__(self, model_path: str = "weights/torch_script_weigths.pt", device: str = None):
        if device:
            self.device = device
        else:
            try:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                self.device = 'cpu'

        print(f"🖥️  Используется устройство: {self.device.upper()}")

        self.model = self._load_model(model_path)
        self.target_sample_rate = 16000

    def _load_model(self, model_path: str) -> torch.jit.ScriptModule:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model

    def load_audio_file(self, file_path: str) -> Tuple[torch.Tensor, int]:
        try:
            waveform, sample_rate = librosa.load(
                file_path,
                sr=self.target_sample_rate,
                mono=False
            )

            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
                original_channels = 1
            else:
                original_channels = waveform.shape[0]

            waveform = torch.from_numpy(waveform).float()

            return waveform, original_channels

        except Exception as e:
            print(f"⚠️  Librosa failed, trying torchaudio: {e}")
            try:
                waveform, sample_rate = torchaudio.load(file_path)
                original_channels = waveform.size(0)

                if sample_rate != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=self.target_sample_rate
                    )
                    waveform = resampler(waveform)

                return waveform, original_channels

            except Exception as e2:
                raise RuntimeError(f"Error processing file {file_path}: {str(e2)}")

    def preprocess_channel(self, channel_waveform: torch.Tensor) -> torch.Tensor:
        processed = torchaudio.functional.preemphasis(channel_waveform.unsqueeze(0))
        processed = apply_random_segment_extraction(processed)
        return processed

    def predict_channel(self, channel_waveform: torch.Tensor) -> Dict[str, float]:
        audio = self.preprocess_channel(channel_waveform).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(audio)

        logits_np = logits.cpu().numpy().ravel()
        probs = softmax(logits_np)

        return {
            'logits': logits_np.tolist(),
            'probabilities': probs.tolist(),
            'fake_prob': float(probs[0]),
            'real_prob': float(probs[1]),
            'prediction': 'REAL' if probs[1] > probs[0] else 'FAKE',
            'confidence': float(max(probs))
        }

    def predict(self, file_path: str) -> List[Dict]:
        waveform, num_channels = self.load_audio_file(file_path)
        results = []

        for channel_idx in range(num_channels):
            channel_waveform = waveform[channel_idx:channel_idx + 1, :]
            channel_result = self.predict_channel(channel_waveform)
            channel_result['channel'] = channel_idx
            results.append(channel_result)

        return results

def predict_audio(file_path: str, model_path: str = "weights/torch_script_weigths.pt") -> List[Dict]:
    model = AntiSpoofingModel(model_path, device='cpu')
    return model.predict(file_path)
