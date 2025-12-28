import whisperx
from typing import Dict, Any


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
        if self.model is None:
            self.load_model()
        return self.model.transcribe(audio, batch_size=self.batch_size)

    @staticmethod
    def load_align_model(language_code: str, device: str):
        return whisperx.load_align_model(language_code=language_code, device=device)

    @staticmethod
    def align(segments, align_model, metadata, audio, device, return_char_alignments: bool = False):
        return whisperx.align(segments, align_model, metadata, audio, device, return_char_alignments=return_char_alignments)

    @staticmethod
    def assign_word_speakers(diarize_segments, result):
        return whisperx.assign_word_speakers(diarize_segments, result)
