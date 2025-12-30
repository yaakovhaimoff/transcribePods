import src.config
from src.transcriber import Transcriber
from src.diarizer import Diarizer
from src.embeddings import EmbeddingExtractor
from src.speaker_db import SpeakerDB
from src.postprocess import PostProcessor


def main():
	transcriber = Transcriber(model_size=src.config.MODEL_SIZE, device=src.config.DEVICE, compute_type=src.config.COMPUTE_TYPE, batch_size=src.config.BATCH_SIZE)
	diarizer = Diarizer(hf_token=src.config.HF_TOKEN, device=src.config.DEVICE)
	extractor = EmbeddingExtractor(hf_token=src.config.HF_TOKEN, device=src.config.DEVICE, sample_rate=src.config.SAMPLE_RATE)
	db = SpeakerDB(src.config.DB_CONFIG, similarity_threshold=src.config.SIMILARITY_THRESHOLD)

	print("Loading audio...")
	try:
		audio = transcriber.load_audio(src.config.AUDIO_FILE)
	except Exception as e:
		print(f"Error loading audio: {e}")
		return

	print("Transcribing audio...")
	result = transcriber.transcribe(audio)

	print("Loading alignment model and aligning words...")
	align_model, metadata = transcriber.load_align_model(language_code=result["language"], device=src.config.DEVICE)
	result = transcriber.align(result["segments"], align_model, metadata, audio, src.config.DEVICE, return_char_alignments=False)

	print("Running diarization pipeline...")
	diarize_segments = diarizer.diarize(audio)

	print("Assigning word speakers...")
	result = transcriber.assign_word_speakers(diarize_segments, result)

	print("Extracting speaker embeddings...")
	speaker_embeddings = extractor.extract(diarize_segments, audio)

	print("Resolving speaker identities...")
	SPEAKER_NAME_MAP = {}
	for speaker_label, embedding in speaker_embeddings.items():
		resolved = db.resolve_speaker_identity(embedding)
		# resolved is (id, name) or possibly None
		if resolved:
			speaker_id, name = resolved
			SPEAKER_NAME_MAP[speaker_label] = name if isinstance(name, str) else f"Speaker {speaker_id}"
		else:
			SPEAKER_NAME_MAP[speaker_label] = speaker_label

	print("Merging transcript...")
	merged_turns = PostProcessor.merge_segments_by_speaker(result["segments"])

	PostProcessor.print_transcript(merged_turns, SPEAKER_NAME_MAP)
	PostProcessor.save_transcript_to_file(merged_turns, SPEAKER_NAME_MAP, filename=src.config.TRANSCRIPT_FILE)


if __name__ == "__main__":
	main()
