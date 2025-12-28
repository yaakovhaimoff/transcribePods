from whisperx.diarize import DiarizationPipeline
from typing import Any
from pathlib import Path
import pandas as pd


class Diarizer:
	"""Wraps the whisperx DiarizationPipeline.

	Responsibilities:
	- instantiate the DiarizationPipeline
	- run diarization on raw audio
	- manage disk caching to cache/diarization.csv
	"""

	def __init__(self, hf_token: str = None, device: str = "cpu"):
		self.hf_token = hf_token
		self.device = device
		self._pipeline = None

	def _ensure_pipeline(self):
		if self._pipeline is None:
			self._pipeline = DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
		return self._pipeline

	def diarize(self, audio) -> Any:
		"""Run diarization with disk caching to cache/diarization.csv.

		If a CSV cache exists, returns a pandas DataFrame. Otherwise runs the
		pipeline, attempts to save the result as CSV, and returns the pipeline output.
		"""
		cached = self._load_cache()
		if cached is not None:
			return cached

		pipeline = self._ensure_pipeline()
		result = pipeline(audio)

		try:
			# attempt to save the returned object (DataFrame-like or convertible)
			self._save_cache(result)
		except Exception:
			pass

		return result

	def _cache_path(self) -> Path:
		return Path.cwd() / "cache" / "diarization.csv"

	def _load_cache(self) -> Any | None:
		p = self._cache_path()
		if not p.exists():
			return None
		# Minimal: file exists -> load and return DataFrame
		return pd.read_csv(p)

	def _save_cache(self, df: Any) -> None:
		p = self._cache_path()
		p.parent.mkdir(parents=True, exist_ok=True)
		# Minimal: write DataFrame (assume df is DataFrame-like)
		if hasattr(df, "to_csv"):
			df.to_csv(p, index=False)
		else:
			pd.DataFrame(df).to_csv(p, index=False)
