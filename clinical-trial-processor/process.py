import json
import torch
import os
import psycopg2
import spacy

from constants import DATABASE_URL_KEY, MODEL_PARAMS, DEFAULT_DATASET, DEFAULT_SPACY_MODEL, POSTGRES_SQL_FETCH_SIZE, POSTGRES_MAX_PROCESSING_SIZE, SCISPACY_LINKER_NAME
from encoder import ClinicalTrialEncoder
from utils import classify_gender_description, normalize_age, process_entities_from_text_chunks, split_criteria, clean_lines, clean_dataset_name

from dotenv import load_dotenv

# ==========================================
# Load model
# ==========================================

# Load the model
model_path = os.path.abspath(os.path.join(MODEL_PARAMS.WEIGHTS_SAVE_DIR.value, clean_dataset_name(DEFAULT_DATASET), MODEL_PARAMS.WEIGHTS_NAME.value))
content = torch.load(model_path, map_location=torch.device('cpu'))

# Extract the critical dictionaries
word_to_ix = content['word_to_ix']
tag_to_ix = content['tag_to_ix']
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

model = ClinicalTrialEncoder(
    vocab_size=len(word_to_ix),
    tagset_size=len(tag_to_ix),
    embedding_dim=MODEL_PARAMS.EMBEDDING_DIM.value,
    hidden_dim=MODEL_PARAMS.HIDDEN_DIM.value
)

# Inject the trained weights into the architecture
model.load_state_dict(content['model_state_dict'])

# Lock the model for inference
model.eval()
print("Model loaded and locked for inference.")

# Load Spacy model
print("Load sciSpacy model...")
nlp = spacy.load(DEFAULT_SPACY_MODEL)

# Add the UMLS Linker to the pipeline
nlp.add_pipe(SCISPACY_LINKER_NAME, config={"resolve_abbreviations": True, "linker_name": "umls"})
print(f"Done loading sciSpacy {DEFAULT_SPACY_MODEL} and UMLS")

# ==========================================
# Connect to db and set up schema
# ==========================================

load_dotenv()

# Connect to db
db_url = os.environ.get(DATABASE_URL_KEY)
conn = psycopg2.connect(db_url)
cur = conn.cursor()

print("Connected to PostgreSQL. Checking schema...")

cur.execute("""
            ALTER TABLE ctgov.eligibilities
            ADD COLUMN IF NOT EXISTS extracted_graph JSONB,
            ADD COLUMN IF NOT EXISTS min_age_months FLOAT,
            ADD COLUMN IF NOT EXISTS max_age_months FLOAT;
            """)
# Create index for faster range look up
cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trial_age_ranges
            ON ctgov.eligibilities (min_age_months, max_age_months);
            """)
conn.commit()
print("Schema ready!")

# ==========================================
# Execution loop
# ==========================================

BATCH_SIZE = POSTGRES_SQL_FETCH_SIZE
count = 0
print("Starting chunked processing...")

while count < POSTGRES_MAX_PROCESSING_SIZE:
    cur.execute(f"""
        SELECT id, minimum_age, maximum_age, gender_description, gender_based, criteria FROM ctgov.eligibilities WHERE extracted_graph IS NOT NULL LIMIT {BATCH_SIZE};
    """)
    
    trials = cur.fetchall()
    
    # If the database returns 0 rows, we are done.
    if not trials:
        print("No more unprocessed trials found. Pipeline complete!")
        break
        
    print(f"--- Processing new batch of {len(trials)} trials ---")
    for trial in trials:
        record_id, min_age, max_age, gender_desc, is_gender_based, criteria_text = trial
        
        trial_graph = {
            "nodes": [],
            "edges": []
        }
        
        min_age_months = normalize_age(min_age) if min_age else None
        max_age_months = normalize_age(max_age) if max_age else None
        
        # Add the bool columns as structured edges
        if is_gender_based:             
            if gender_desc is not None:
                demographics = classify_gender_description(gender_desc)
                
                trial_graph["edges"].append({"type": "REQUIRES_PREGNANCY", "target": demographics["requires_pregnancy"]})
                    
                if demographics["target_sex_at_birth"]:
                    trial_graph["edges"].append({"type": "REQUIRES_BIOLOGICAL_SEX", "target": demographics["target_sex_at_birth"]})
                    
                for procedure in demographics["requires_procedures"]:
                    trial_graph["edges"].append({"type": "REQUIRES_PRIOR_PROCEDURE", "target": procedure})
        
        # Process inclusions & exc
        inc_text, exc_text = split_criteria(criteria_text)
        cleaned_inc_text, cleaned_exc_text = clean_lines(inc_text), clean_lines(exc_text)
        
        process_entities_from_text_chunks(cleaned_inc_text, nlp, word_to_ix, model, ix_to_tag, trial_graph)
        process_entities_from_text_chunks(cleaned_exc_text, nlp, word_to_ix, model, ix_to_tag, trial_graph, False)

        # Save the graph back to Postgres
        cur.execute("""
            UPDATE ctgov.eligibilities 
            SET 
                extracted_graph = %s,
                min_age_months = %s,
                max_age_months = %s
            WHERE id = %s;
            """, (json.dumps(trial_graph), min_age_months, max_age_months, record_id)
        )

    # Commit changes to the db
    conn.commit()
    count += BATCH_SIZE
    
cur.close()
conn.close()
print("Processing complete.")