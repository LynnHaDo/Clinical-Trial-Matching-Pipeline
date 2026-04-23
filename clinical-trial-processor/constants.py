from typing import Final
from enum import Enum

NCBI_DATASET_NAME: Final[str] = "ncbi_disease"
LOCAL_NCBI_DATASET_DISK_PATH: Final[str] = './datasets'

class NCBI_DATASET_VOCAB_KEYS(Enum):
    PADDING = "<PAD>",
    UNKNOWN = "<UNK>"

NCBI_DATASET_TAG_TO_IX: Final[dict] = {    
    "O": 0,
    "B-Disease": 1,
    "I-Disease": 2,
    "B-Chemical": 3,
    "I-Chemical": 4,
    "B-Neg-Disease": 5,
    "I-Neg-Disease": 6,
    "<START_TAG>": 7, # Required by CRF to know where a sequence begins
    "<STOP_TAG>": 8   # Required by CRF to know where a sequence ends
}

class NCBI_DATASET_DATA_FIELDS(Enum):
    NER_TAGS = 'ner_tags'
    TOKENS = 'tokens'

NCBI_DATASET_NEGATION_TRIGGERS: Final[set] = {"no", "not", "without", "absence", "deny", "denies", "denied", "negative", "rules out"}
NCBI_DATASET_NEGATION_WINDOW_SIZE = 3 # Consider up to 3 words backwards

DATABASE_URL_KEY: Final[str] = "DATABASE_URL"

class MODEL_PARAMS(Enum):
    LR = 0.001
    EPOCHS = 10