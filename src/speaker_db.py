import mysql.connector
import numpy as np
from typing import Tuple


class SpeakerDB:
	"""Simple wrapper around a MySQL speaker-identity table.

	Responsibilities:
	- open DB connection
	- resolve or register speaker embeddings
	"""

	def __init__(self, db_config: dict, similarity_threshold: float = 0.75):
		self.db_config = db_config
		self.similarity_threshold = similarity_threshold
		self.db = None
		self.cursor = None

	def connect(self):
		if self.db is None:
			self.db = mysql.connector.connect(**self.db_config)
			self.cursor = self.db.cursor()
		return self.db, self.cursor

	def resolve_speaker_identity(self, embedding, prompt_for_name: bool = True) -> Tuple[int, str]:
		"""Return (speaker_id, speaker_name). If unknown, optionally prompt and register."""
		db, cursor = self.connect()

		emb_np = embedding.detach().cpu().numpy().astype(np.float32)

		cursor.execute("""
			SELECT se.speaker_id, s.name, se.embedding
			FROM speaker_embeddings se
			JOIN speakers s ON se.speaker_id = s.id
		""")
		rows = cursor.fetchall()

		best_match = None
		best_score = -1.0

		for speaker_id, name, blob in rows:
			db_emb = np.frombuffer(blob, dtype=np.float32)
			score = np.dot(emb_np, db_emb) / (np.linalg.norm(emb_np) * np.linalg.norm(db_emb))
			if score > best_score:
				best_score = score
				best_match = (speaker_id, name)

		if best_score > self.similarity_threshold:
			return best_match

		if not prompt_for_name:
			return None

		name = input("New speaker detected. Enter speaker name: ").strip()
		cursor.execute("INSERT INTO speakers (name) VALUES (%s)", (name,))
		speaker_id = cursor.lastrowid
		cursor.execute("INSERT INTO speaker_embeddings (speaker_id, embedding) VALUES (%s, %s)", (speaker_id, emb_np.tobytes()))
		db.commit()
		return speaker_id, name

	def close(self):
		if self.cursor:
			self.cursor.close()
		if self.db:
			self.db.close()
