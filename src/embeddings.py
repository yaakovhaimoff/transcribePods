import torch
import numpy as np
from pyannote.audio import Model
from typing import Dict, Any
from pathlib import Path


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
		# check disk cache first
		cached = self._load_cache()
		if cached is not None:
			return cached

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

		# save to disk for future runs
		try:
			self._save_cache(embeddings)
		except Exception:
			pass

		return embeddings

	def _cache_path(self) -> Path:
		return Path.cwd() / "cache" / "embeddings.npy"

	def _load_cache(self) -> Dict[str, torch.Tensor] | None:
		p = self._cache_path()
		if not p.exists():
			return None
		with np.load(p, allow_pickle=True) as data:
			emb = {}
			for key in data.files:
				arr = data[key]
				emb[key] = torch.from_numpy(arr)
			return emb

	def _save_cache(self, embeddings: Dict[str, torch.Tensor]) -> None:
		p = self._cache_path()
		p.parent.mkdir(parents=True, exist_ok=True)
		# convert tensors to numpy arrays for saving
		save_dict = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)) for k, v in embeddings.items()}
		# use savez to store a mapping of speaker -> array
		np.savez(p, **save_dict)

