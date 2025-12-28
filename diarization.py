import os
import numpy as np
import torch
import mysql.connector
import whisperx
from whisperx.diarize import DiarizationPipeline
from pyannote.audio import Model
from dotenv import load_dotenv


AUDIO_FILE = "DIALOGUE.ogg"
DEVICE = "cpu"
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"
BATCH_SIZE = 16
HF_TOKEN = os.environ.get("HF_TOKEN")
SIMILARITY_THRESHOLD = 0.75


def merge_segments_by_speaker(segments):
	"""
	Merge consecutive segments spoken by the same speaker
	into a single readable paragraph.
	"""
	merged = []
	current_speaker = None
	current_text = []

	for seg in segments:
		speaker = seg.get("speaker", "UNKNOWN")
		text = seg["text"].strip()

		if not text:
			continue

		if speaker != current_speaker:
			if current_text:
				merged.append({
					"speaker": current_speaker,
					"text": " ".join(current_text)
				})
			current_speaker = speaker
			current_text = [text]
		else:
			current_text.append(text)

	if current_text:
		merged.append({
			"speaker": current_speaker,
			"text": " ".join(current_text)
		})

	return merged


def clean_text(text):
	"""
	Light text cleanup for readability
	"""
	return (
		text.replace(" ,", ",")
			.replace(" .", ".")
			.replace(" ?", "?")
			.replace(" !", "!")
			.strip()
	)


def print_transcript(merged_turns, SPEAKER_NAME_MAP):
	"""
	Print a readable, speaker-labeled transcript to stdout.

	merged_turns: list of dicts with keys `speaker` and `text`.
	SPEAKER_NAME_MAP: mapping from speaker labels to human-friendly names.
	"""
	print("\n==================== FINAL TRANSCRIPT ====================\n")
	for turn in merged_turns:
		speaker = SPEAKER_NAME_MAP.get(turn["speaker"], turn["speaker"])
		text = clean_text(turn["text"])
		print(f"{speaker}: {text}\n")
	print("==========================================================\n")


def save_transcript_to_file(merged_turns, SPEAKER_NAME_MAP, filename="transcript.txt"):
	"""
	Save the merged transcript to a plain text file.

	Parameters:
	- merged_turns: list of dicts with keys `speaker` and `text`.
	- SPEAKER_NAME_MAP: mapping from speaker labels to human-friendly names.
	- filename: output filename (defaults to `transcript.txt`).
	"""
	with open(filename, "w", encoding="utf-8") as f:
		for turn in merged_turns:
			speaker = SPEAKER_NAME_MAP.get(turn["speaker"], turn["speaker"])
			text = clean_text(turn["text"])
			f.write(f"{speaker}: {text}\n\n")

	print("Transcript saved to transcript.txt")


def extract_speaker_embeddings(diarize_segments, audio, embedding_model, sample_rate=16000):
	"""
	Extract a single averaged embedding per diarized speaker.

	This slices `audio` by the diarization segments, runs the `embedding_model`
	on each segment, and averages embeddings per speaker. Very short segments
	(< 0.5s) are skipped.

	Returns a dict mapping speaker label -> torch.Tensor embedding.
	"""
	embeddings = {}

	for speaker in diarize_segments["speaker"].unique():
		speaker_rows = diarize_segments[diarize_segments["speaker"] == speaker]

		speaker_embs = []

		for _, row in speaker_rows.iterrows():
			start = int(row["start"] * sample_rate)
			end = int(row["end"] * sample_rate)

			segment_audio = audio[start:end]

			# Skip very short segments (important!)
			if len(segment_audio) < sample_rate * 0.5:
				continue

			segment_tensor = torch.tensor(segment_audio, dtype=torch.float32)
			segment_tensor = segment_tensor.unsqueeze(0)  # [1, T]

			with torch.no_grad():
				emb = embedding_model(segment_tensor)

			speaker_embs.append(emb.squeeze())

		if speaker_embs:
			embeddings[speaker] = torch.stack(speaker_embs).mean(dim=0)

	return embeddings


def resolve_speaker_identity(embedding, cursor, db, threshold=0.75):
	"""
	Resolve speaker identity using embeddings.
	Returns (speaker_id, speaker_name).
	"""
	emb_np = embedding.detach().cpu().numpy().astype(np.float32)

	cursor.execute("""
		SELECT se.speaker_id, s.name, se.embedding
		FROM speaker_embeddings se
		JOIN speakers s ON se.speaker_id = s.id
	""")
	rows = cursor.fetchall()

	best_match = None
	best_score = -1

	for speaker_id, name, blob in rows:
		db_emb = np.frombuffer(blob, dtype=np.float32)

		score = np.dot(emb_np, db_emb) / (
			np.linalg.norm(emb_np) * np.linalg.norm(db_emb)
		)

		if score > best_score:
			best_score = score
			best_match = (speaker_id, name)

	# Known speaker
	if best_score > threshold:
		return best_match

	# New speaker → ask for name
	name = input("New speaker detected. Enter speaker name: ").strip()

	cursor.execute(
		"INSERT INTO speakers (name) VALUES (%s)",
		(name,)
	)
	speaker_id = cursor.lastrowid

	cursor.execute(
		"INSERT INTO speaker_embeddings (speaker_id, embedding) VALUES (%s, %s)",
		(speaker_id, emb_np.tobytes())
	)

	db.commit()

	return speaker_id, name


def main():
	load_dotenv()
	print("Connecting to DB...")
	DB_CONFIG = {
		"host": os.environ.get("DB_HOST"),
		"user": os.environ.get("DB_USER"),
		"password": os.environ.get("DB_PASSWORD"),
		"database": os.environ.get("DB_DATABASE"),
	}
	db = mysql.connector.connect(**DB_CONFIG)
	cursor = db.cursor()

	print("Loading WhisperX model...")
	model = whisperx.load_model(
		MODEL_SIZE,
		DEVICE,
		compute_type=COMPUTE_TYPE
	)

	print("Loading audio...")
	audio = whisperx.load_audio(AUDIO_FILE)

	print("Transcribing...")
	result = model.transcribe(audio, batch_size=BATCH_SIZE)

	print("Aligning words...")
	align_model, metadata = whisperx.load_align_model(
		language_code=result["language"],
		device=DEVICE
	)

	result = whisperx.align(
		result["segments"],
		align_model,
		metadata,
		audio,
		DEVICE,
		return_char_alignments=False
	)

	print("Running speaker diarization...")
	diarize_model = DiarizationPipeline(
		use_auth_token=HF_TOKEN,
		device=DEVICE
	)

	diarize_segments = diarize_model(audio)

	print("Assigning speakers...")
	result = whisperx.assign_word_speakers(diarize_segments, result)

	print("Loading embedding model...")
	embedding_model = Model.from_pretrained(
		"pyannote/embedding",
		use_auth_token=HF_TOKEN
	)
	embedding_model.to(DEVICE)
	embedding_model.eval()

	print("Extracting speaker embeddings...")
	speaker_embeddings = extract_speaker_embeddings(
		diarize_segments,
		audio,
		embedding_model
	)

	print("Resolving identities...")
	SPEAKER_NAME_MAP = {}

	for speaker_label, embedding in speaker_embeddings.items():
		speaker_id = resolve_speaker_identity(embedding, cursor, db)
		SPEAKER_NAME_MAP[speaker_label] = f"Speaker {speaker_id}"

	print("Merging transcript...")
	merged_turns = merge_segments_by_speaker(result["segments"])

	print_transcript(merged_turns, SPEAKER_NAME_MAP)

	save_transcript_to_file(merged_turns, SPEAKER_NAME_MAP)


if __name__ == "__main__":
	main()
