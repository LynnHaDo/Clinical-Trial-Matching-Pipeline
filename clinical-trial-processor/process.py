import json
import torch
import os
import psycopg2
import spacy

from constants import DATABASE_URL_KEY, MODEL_PARAMS, DEFAULT_DATASET, DEFAULT_SPACY_MODEL, POSTGRES_SQL_CURSOR_NAME, POSTGRES_SQL_FETCH_SIZE
from encoder import ClinicalTrialEncoder
from utils import normalize_age, process_entities_from_text_chunks, split_criteria, clean_lines, clean_dataset_name, is_valid_medical_term

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
print(f"Done loading sciSpacy {DEFAULT_SPACY_MODEL}")

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
            ADD COLUMN IF NOT EXISTS extracted_graph JSONB;
            """)
conn.commit()
print("Schema ready!")

# ==========================================
# Execution loop
# ==========================================

BATCH_SIZE = POSTGRES_SQL_FETCH_SIZE
print("Starting chunked processing...")

while True:
    cur.execute(f"""
        SELECT id, minimum_age, maximum_age, healthy_volunteers, gender_description, gender_based, criteria FROM ctgov.eligibilities WHERE criteria IS NOT NULL AND extracted_graph IS NULL LIMIT {BATCH_SIZE};
    """)
    
    trials = cur.fetchall()
    
    # If the database returns 0 rows, we are done.
    if not trials:
        print("No more unprocessed trials found. Pipeline complete!")
        break
        
    print(f"--- Processing new batch of {len(trials)} trials ---")
    for trial in trials:
        record_id, min_age, max_age, is_healthy, gender_desc, is_gender_based, criteria_text = trial
        
        trial_graph = {
            "nodes": [],
            "edges": []
        }
        
        if min_age: 
            age = normalize_age(min_age)
            if age:
                trial_graph["edges"].append({"type": "HAS_MIN_AGE", "target": age})
            
        if max_age:
            age = normalize_age(max_age) 
            if age:
                trial_graph["edges"].append({"type": "HAS_MAX_AGE", "target": age})
        
        # Add the bool columns as structured edges
        if is_healthy is not None: trial_graph["edges"].append({"type": "REQUIRES_HEALTHY_PATIENTS", "target": is_healthy})
        if is_gender_based is not None: trial_graph["edges"].append({"type": "IS_GENDER_BASED", "target": is_gender_based})
        if gender_desc is not None:
            pass # TODO
        
        # Process inclusions & exc
        inc_text, exc_text = split_criteria(criteria_text)
        cleaned_inc_text, cleaned_exc_text = clean_lines(inc_text), clean_lines(exc_text)
        
        process_entities_from_text_chunks(cleaned_inc_text, nlp, word_to_ix, model, ix_to_tag, trial_graph)
        process_entities_from_text_chunks(cleaned_exc_text, nlp, word_to_ix, model, ix_to_tag, trial_graph, False)

        # Save the graph back to Postgres
        cur.execute(
            "UPDATE ctgov.eligibilities SET extracted_graph = %s WHERE id = %s;",
            (json.dumps(trial_graph), record_id)
        )

    # Commit changes to the db
    conn.commit()
    
cur.close()
conn.close()
print("Processing complete.")