from typing import List, Dict


class PostProcessor:
	"""Text post-processing and I/O helpers.

	Responsibilities:
	- merge consecutive segments by speaker
	- clean text
	- print and save final transcript
	"""

	@staticmethod
	def merge_segments_by_speaker(segments: List[Dict]) -> List[Dict]:
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
					merged.append({"speaker": current_speaker, "text": " ".join(current_text)})
				current_speaker = speaker
				current_text = [text]
			else:
				current_text.append(text)

		if current_text:
			merged.append({"speaker": current_speaker, "text": " ".join(current_text)})

		return merged

	@staticmethod
	def clean_text(text: str) -> str:
		return (
			text.replace(" ,", ",")
				.replace(" .", ".")
				.replace(" ?", "?")
				.replace(" !", "!")
				.strip()
		)

	@staticmethod
	def print_transcript(merged_turns: List[Dict], speaker_name_map: Dict[str, str]):
		print("\n==================== FINAL TRANSCRIPT ====================\n")
		for turn in merged_turns:
			speaker = speaker_name_map.get(turn["speaker"], turn["speaker"])
			text = PostProcessor.clean_text(turn["text"])
			print(f"{speaker}: {text}\n")
		print("==========================================================\n")

	@staticmethod
	def save_transcript_to_file(merged_turns: List[Dict], speaker_name_map: Dict[str, str], filename: str = "transcript.txt"):
		with open(filename, "w", encoding="utf-8") as f:
			for turn in merged_turns:
				speaker = speaker_name_map.get(turn["speaker"], turn["speaker"])
				text = PostProcessor.clean_text(turn["text"])
				f.write(f"{speaker}: {text}\n\n")
		print(f"Transcript saved to {filename}")
