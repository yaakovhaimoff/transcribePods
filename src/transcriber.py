import json
from pathlib import Path
from typing import Dict, Any
import whisperx


class Transcriber:
	"""Wraps WhisperX transcription and alignment steps.

	Responsibilities:
	- load the whisperx model
	- load audio
	- transcribe audio
	- load alignment model and align words
	- assign word speakers after diarization
	"""

	def __init__(self, model_size: str, device: str, compute_type: str = None, batch_size: int = 16):
		self.model_size = model_size
		self.device = device
		self.compute_type = compute_type
		self.batch_size = batch_size
		self.model = None

	def load_model(self):
		self.model = whisperx.load_model(self.model_size, self.device, compute_type=self.compute_type)
		return self.model

	@staticmethod
	def load_audio(path: str):
		return whisperx.load_audio(path)

	def transcribe(self, audio) -> Dict[str, Any]:
		"""Transcribe audio with disk caching.

		Flow:
		- Check cache/transcription.json exists
		  - if yes: load JSON and return it
		  - if no: run the model.transcribe(...) call, save JSON to disk, return result

		Note: This method keeps the model logic unchanged and only wraps it with
		explicit disk I/O. No in-memory/global caches are used.
		"""
		# 1) try load cache
		cached = self._load_cache()
		if cached is not None:
			return cached

		# 2) compute (model unchanged)
		if self.model is None:
			self.load_model()

		result = self.model.transcribe(audio, batch_size=self.batch_size)

		# 3) save to disk and return
		try:
			self._save_cache(result)
		except Exception:
			pass

		return result

	@staticmethod
	def load_align_model(language_code: str, device: str):
		return whisperx.load_align_model(language_code=language_code, device=device)

	def align(self, segments, align_model, metadata, audio, device, return_char_alignments: bool = False):
		"""Align words with disk caching to cache/alignment.json.

		The method keeps the underlying whisperx.align call unchanged. It first
		checks for a cached JSON result and returns it if present. Otherwise it
		runs the align call, saves JSON, and returns the result.
		"""
		# try load cache
		cached = self._load_alignment_cache()
		if cached is not None:
			return cached

		# compute
		result = whisperx.align(segments, align_model, metadata, audio, device, return_char_alignments=return_char_alignments)

		# save if possible
		try:
			self._save_alignment_cache(result)
		except Exception:
			pass

		return result

	@staticmethod
	def assign_word_speakers(diarize_segments, result):
		return whisperx.assign_word_speakers(diarize_segments, result)

	def _cache_path(self) -> Path:
		"""Return Path to transcription cache file: cache/transcription.json"""
		return Path.cwd() / "cache" / "transcription.json"

	def _load_cache(self) -> Dict[str, Any] | None:
		p = self._cache_path()
		if not p.exists():
			return None
		with p.open("r", encoding="utf-8") as fh:
			return json.load(fh)

	def _save_cache(self, data: Dict[str, Any]) -> None:
		p = self._cache_path()
		p.parent.mkdir(parents=True, exist_ok=True)
		with p.open("w", encoding="utf-8") as fh:
			json.dump(data, fh, ensure_ascii=False, indent=2)

	def _alignment_cache_path(self) -> Path:
		return Path.cwd() / "cache" / "alignment.json"

	def _load_alignment_cache(self) -> Dict[str, Any] | None:
		p = self._alignment_cache_path()
		if not p.exists():
			return None
		with p.open("r", encoding="utf-8") as fh:
			return json.load(fh)

	def _save_alignment_cache(self, data: Dict[str, Any]) -> None:
		p = self._alignment_cache_path()
		p.parent.mkdir(parents=True, exist_ok=True)
		with p.open("w", encoding="utf-8") as fh:
			json.dump(data, fh, ensure_ascii=False, indent=2)
