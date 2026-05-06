from typing import Final

DATABASE_URL_KEY: Final[str] = "DATABASE_URL"
MATCHING_DISTANCE_THRESHOLD: Final[float] = 0.15 # Lower = stricter
CLINICAL_BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MEDICAL_TRANSCRIPTIONS_TABLE_NAME = "medical_transcriptions"