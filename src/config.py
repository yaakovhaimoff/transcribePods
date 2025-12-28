import os
from dotenv import load_dotenv
load_dotenv()

# Basic runtime and model configuration
AUDIO_FILE = os.getenv("AUDIO_FILE", "DIALOGUE.ogg")
DEVICE = os.getenv("DEVICE", "cpu")
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
HF_TOKEN = os.getenv("HF_TOKEN")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
TRANSCRIPT_FILE = os.getenv("TRANSCRIPT_FILE", "transcript.txt")

# Database config read at runtime from env
DB_CONFIG = {
	"host": os.getenv("DB_HOST"),
	"user": os.getenv("DB_USER"),
	"password": os.getenv("DB_PASSWORD"),
	"database": os.getenv("DB_DATABASE"),
}
