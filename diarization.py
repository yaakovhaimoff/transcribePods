import whisperx
from whisperx.diarize import DiarizationPipeline
import os


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
    print("\n==================== FINAL TRANSCRIPT ====================\n")
    for turn in merged_turns:
        speaker = SPEAKER_NAME_MAP.get(turn["speaker"], turn["speaker"])
        text = clean_text(turn["text"])
        print(f"{speaker}: {text}\n")
    print("==========================================================\n")


def save_transcript_to_file(merged_turns, SPEAKER_NAME_MAP, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for turn in merged_turns:
            speaker = SPEAKER_NAME_MAP.get(turn["speaker"], turn["speaker"])
            text = clean_text(turn["text"])
            f.write(f"{speaker}: {text}\n\n")

    print("Transcript saved to transcript.txt")


def main():
    # CONFIG
    AUDIO_FILE = "DIALOGUE.ogg"
    DEVICE = "cpu"          # MUST be "cpu" on Mac M1
    MODEL_SIZE = "medium"    # small / medium
    COMPUTE_TYPE = "int8"   # best for CPU
    BATCH_SIZE = 16

    HF_TOKEN = os.environ.get("HF_TOKEN")
    SPEAKER_NAME_MAP = {
        "SPEAKER_00": "First Speaker",
        "SPEAKER_01": "Second Speaker",
    }
    
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
    result = whisperx.assign_word_speakers(
        diarize_segments,
        result
    )

    print("Merging speaker turns...")
    merged_turns = merge_segments_by_speaker(result["segments"])

    print_transcript(merged_turns, SPEAKER_NAME_MAP)

    save_transcript_to_file(merged_turns, SPEAKER_NAME_MAP)


if __name__ == "__main__":
    main()
