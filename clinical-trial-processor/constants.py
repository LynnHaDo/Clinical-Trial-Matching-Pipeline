from typing import Final
from enum import Enum

DATASET_DISK_PATH: Final[str] = './datasets'
"""
    This dataset contains the disease name and concept annotations of the NCBI disease corpus, 
    a collection of 793 PubMed abstracts fully annotated at the mention and 
    concept level to serve as a research resource for the biomedical natural language processing community.
"""
NCBI_DATASET_NAME: Final[str] = "ncbi_disease"
BC5CDR_DATASET_NAME: Final[str] = "tner/bc5cdr"
DEFAULT_DATASET: Final[str] = BC5CDR_DATASET_NAME

class DATASET_VOCAB_KEYS(Enum):
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

BC5CDR_DATASET_NATIVE_TAG_TO_IX: Final[dict] = {   
    0: "O",
    1: "B-Chemical",
    2: "B-Disease",
    3: "I-Disease",
    4: "I-Chemical"
}

BC5CDR_DATASET_TAG_TO_IX: Final[dict] = {    
    "O": 0,
    "B-Disease": 1,
    "I-Disease": 2,
    "B-Chemical": 3,
    "I-Chemical": 4,
    "B-Neg-Disease": 5,
    "I-Neg-Disease": 6,
    "B-Neg-Chemical": 7,
    "I-Neg-Chemical": 8,
    "<START_TAG>": 9, # Required by CRF to know where a sequence begins
    "<STOP_TAG>": 10   # Required by CRF to know where a sequence ends
}

class NCBI_DATASET_DATA_FIELDS(Enum):
    NER_TAGS = 'ner_tags'
    TOKENS = 'tokens'

class BC5CDR_DATASET_DATA_FIELDS(Enum):
    NER_TAGS = 'tags'
    TOKENS = 'tokens'

DATASET_NEGATION_TRIGGERS: Final[set] = {"no", "not", "without", "absence", "deny", "denies", "denied", "negative", "rules out"}
DATASET_NEGATION_WINDOW_SIZE = 3 # Consider up to 3 words backwards

DATABASE_URL_KEY: Final[str] = "DATABASE_URL"

AACT_DB_NULL_VALUES: Final[set] = {'n/a (no limit)', 'n/a', 'none', '[null]', 'null'}
COMMON_DRUG_SUFFIXES: Final[tuple] = ('ine', 'pam', 'lol', 'mab', 'vir', 'cillin')

POSTGRES_SQL_FETCH_SIZE = 100
POSTGRES_SQL_CURSOR_NAME = 'fetch_trials_cursor'
DEFAULT_SPACY_MODEL = 'en_core_sci_sm'

class MODEL_PARAMS(Enum):
    LR = 0.001
    EPOCHS = 20
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    WEIGHTS_SAVE_DIR = './models'
    WEIGHTS_NAME = 'clinical_trial_ner.pt'
    TRAINING_LOSSES_OUTPUT_NAME = 'clinical_trial_ner_losses.pt'

class SEX_AT_BIRTH(Enum):
    FEMALE = 'Female'
    MALE = 'Male'
    BOTH = 'Both'