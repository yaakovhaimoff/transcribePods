import torch
import numpy as np
from pyannote.audio import Model
from typing import Dict, Any


class EmbeddingExtractor:
    """Load a pyannote embedding model and extract averaged speaker embeddings.

    Responsibilities:
    - load embedding model
    - extract embeddings per speaker (average over segments)
    """

    def __init__(self, hf_token: str = None, device: str = "cpu", sample_rate: int = 16000):
        self.hf_token = hf_token
        self.device = device
        self.sample_rate = sample_rate
        self.model = None

    def load_model(self):
        self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=self.hf_token)
        self.model.to(self.device)
        self.model.eval()
        return self.model

    def _ensure_model(self):
        if self.model is None:
            self.load_model()
        return self.model

    def extract(self, diarize_segments, audio) -> Dict[str, torch.Tensor]:
        """Return mapping speaker_label -> averaged embedding tensor."""
        model = self._ensure_model()
        embeddings = {}

        # diarize_segments is expected to support `.unique()` and row iteration like a pandas DataFrame
        for speaker in diarize_segments["speaker"].unique():
            speaker_rows = diarize_segments[diarize_segments["speaker"] == speaker]
            speaker_embs = []

            for _, row in speaker_rows.iterrows():
                start = int(row["start"] * self.sample_rate)
                end = int(row["end"] * self.sample_rate)
                segment_audio = audio[start:end]

                if len(segment_audio) < self.sample_rate * 0.5:
                    continue

                segment_tensor = torch.tensor(segment_audio, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    emb = model(segment_tensor)
                speaker_embs.append(emb.squeeze())

            if speaker_embs:
                embeddings[speaker] = torch.stack(speaker_embs).mean(dim=0)

        return embeddings
