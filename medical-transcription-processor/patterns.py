CUSTOM_PATTERNS = [
    {"label": "CHEMICAL", "pattern": [{"LOWER": "ortho"}, {"TEXT": {"REGEX": "(?i)^tri-?cyclen$"}}]},
    {"label": "CHEMICAL", "pattern": [{"LOWER": "ortho"}, {"LOWER": "tri"}, {"TEXT": "-"}, {"LOWER": "cyclen"}]},
    {"label": "CHEMICAL", "pattern": [{"LOWER": "allegra"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "mildly"}, {"LOWER": "erythematous"}, {"LOWER": "throat"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "erythematous"}, {"LOWER": "nasal"}, {"LOWER": "mucosa"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "swollen"}, {"LOWER": "nasal"}, {"LOWER": "mucosa"}]},
    {"label": "DISEASE", "pattern": [{"TEXT": "CVA"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "cholesterol"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "pitting"}, {"LOWER": "edema"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "allergic"}, {"LOWER": "rhinitis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "gastroesophageal"}, {"LOWER": "reflux"}, {"LOWER": "disease"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "left"}, {"LOWER": "atrial"}, {"LOWER": "enlargement"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "hysterectomy"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "oophorectomy"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "salpingectomy"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "vaginectomy"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "vulvectomy"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "prostatectomy"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "orchiectomy"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "vasectomy"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "mastectomy"}]},
    
]
RACE_PATTERNS = [
    {"label": "RACE", "pattern": [{"LOWER": {"IN": ["white", "black", "asian", "hispanic", "latino", "latina", "caucasian", "african", "american"]}}]},
    {"label": "RACE", "pattern": [{"LOWER": "african"}, {"TEXT": "-", "OP": "?"}, {"LOWER": "american"}]}
]

GENDER_PATTERNS = [
    {"label": "GENDER", "pattern": [{"LOWER": {"IN": ["male", "female", "man", "woman", "boy", "girl", "lady", "gentleman", "transgender", "trans", "nonbinary", "agender"]}}]},
    {"label": "GENDER", "pattern": [{"LOWER": "non"}, {"TEXT": "-", "OP": "?"}, {"LOWER": "binary"}]}
]

FEMALE_PROCEDURES = {
    "hysterectomy", "oophorectomy", "salpingectomy", "vaginectomy", "vulvectomy"
}

MALE_PROCEDURES = {
    "prostatectomy", "orchiectomy", "vasectomy"
}

BOTH_PROCEDURES = {
    "mastectomy"
}

AGE_PATTERNS = [
    {"label": "AGE", "pattern": [{"LIKE_NUM": True}, {"TEXT": "-", "OP": "?"}, {"LOWER": "year"}, {"TEXT": "-", "OP": "?"}, {"LOWER": "old"}]},
    {"label": "AGE", "pattern": [{"LIKE_NUM": True}, {"TEXT": "-", "OP": "?"}, {"LOWER": "years"}, {"TEXT": "-", "OP": "?"}, {"LOWER": "old"}]},
    {"label": "AGE", "pattern": [{"TEXT": {"REGEX": "(?i)^\d+-years?-old$"}}]}
]

PREGNANCY_PATTERNS = [
    {"label": "PREGNANCY", "pattern": [{"LOWER": {"IN": ["pregnant", "pregnancy", "gravid", "gestation"]}}]},
    {"label": "PREGNANCY", "pattern": [{"LOWER": "intrauterine"}, {"LOWER": "pregnancy"}]},
    {"label": "PREGNANCY", "pattern": [{"TEXT": {"REGEX": "(?i)^iup$"}}]} # Common abbreviation
]

IGNORE_ANATOMY = {"throat", "nose", "ear", "eye", "mouth", "teeth", "afebrile", "alcohol"}

# --- NORMALIZATION DICTIONARIES ---
GENDER_MAP = {
    "male": "male", "man": "male", "boy": "male", "gentleman": "male",
    "female": "female", "woman": "female", "girl": "female", "lady": "female",
    "transgender": "other", "trans": "other", "nonbinary": "other", 
    "non-binary": "other", "non binary": "other", "agender": "other"
}

NEGATION_WORDS = {"no", "not", "denies", "without", "negative", "ruled out"}
# Core anatomy that goes with the word pain
ANATOMY_TERMS = ["knee", "back", "chest", "hip", "foot", "ankle", "leg", "arm", 
    "shoulder", "neck", "head", "joint", "muscle", "throat", "nose", 
    "ear", "eye", "mouth", "spine", "lumbar", "cervical", "thoracic", 
    "sacrum", "coccyx", "pelvis", "groin", "abdomen", "epigastric", 
    "wrist", "elbow", "hand", "finger", "thumb", "toe", "heel", 
    "calf", "thigh", "shin", "forearm", "jaw", "temple", "occipital", 
    "rib", "sternum", "scapula", "ligament", "tendon", "nerve", 
    "pelvic floor", "sciatic nerve", "teeth", "tooth"]
FILLER_WORDS = [
    "in", "on", "the", "their", "his", "her", "my", "a", 
    "left", "right", "bilateral", "lower", "upper", "and", "of"
]
DYNAMIC_PAIN_PATTERNS = [
    # Catches: "pain in their knee", "pain in the left ankle", "pain of the lower back"
    {
        "label": "DISEASE", 
        "pattern": [
            {"LOWER": "pain"}, 
            {"LOWER": {"IN": FILLER_WORDS}, "OP": "*"}, # The "*" means zero or more of these filler words
            {"LOWER": {"IN": ANATOMY_TERMS}}
        ]
    },
    # Catches: "knee and ankle pain", "left hip pain", "lower back pain"
    {
        "label": "DISEASE", 
        "pattern": [
            {"LOWER": {"IN": ANATOMY_TERMS}}, 
            {"LOWER": {"IN": FILLER_WORDS}, "OP": "*"}, 
            {"LOWER": "pain"}
        ]
    }
]

# Combine them for easy import
ALL_PATTERNS = AGE_PATTERNS + GENDER_PATTERNS + RACE_PATTERNS + CUSTOM_PATTERNS + PREGNANCY_PATTERNS + DYNAMIC_PAIN_PATTERNS