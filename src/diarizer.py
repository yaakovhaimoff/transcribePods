from whisperx.diarize import DiarizationPipeline
from typing import Any


class Diarizer:
    """Wraps the whisperx DiarizationPipeline.

    Responsibilities:
    - instantiate the DiarizationPipeline
    - run diarization on raw audio
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
        pipeline = self._ensure_pipeline()
        return pipeline(audio)
