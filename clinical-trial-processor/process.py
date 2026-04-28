import json
import torch
import os
import psycopg2

from constants import DATABASE_URL_KEY, MODEL_PARAMS, DEFAULT_DATASET
from encoder import ClinicalTrialEncoder
from utils import normalize_age, prepare_sequence, extract_entities, split_criteria, clean_lines, clean_dataset_name

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

# ==========================================
# Connect to db and set up schema
# ==========================================

load_dotenv()

# Connect to db
db_url = os.environ.get(DATABASE_URL_KEY)
conn = psycopg2.connect(db_url)
cur = conn.cursor()

print("Connected to PostgreSQL. Checking schema...")

# cur.execute("""
#             ALTER TABLE ctgov.eligibilities
#             ADD COLUMN IF NOT EXISTS extracted_graph JSONB
#             """)
# conn.commit()
print("Schema ready!")

# ==========================================
# Execution loop
# ==========================================

print("Fetching trials...")
# Fetch the trials where the population text hasn't been processed yet
# cur.execute("""
#             SELECT 
#                 id, nct_id, minimum_age, maximum_age, 
#                 criteria, adult, child, older_adult 
#             FROM ctgov.eligibilities 
#             WHERE (population IS NOT NULL OR criteria IS NOT NULL)
#             AND extracted_graph IS NULL
#             LIMIT 100;
#             """)

cur.execute("""
            SELECT 
                id, nct_id, minimum_age, maximum_age, healthy_volunteers, 
                criteria, adult, child, older_adult 
            FROM ctgov.eligibilities 
            WHERE (population IS NOT NULL OR criteria IS NOT NULL)
            LIMIT 5;
            """)

# cur.execute("""
#             SELECT 
#                 id, nct_id, minimum_age, maximum_age, 
#                 criteria, adult, child, older_adult 
#             FROM ctgov.eligibilities 
#             WHERE (population IS NOT NULL OR criteria IS NOT NULL)
#             AND extracted_graph IS NULL
#             LIMIT 5;
#             """)
trials = cur.fetchall()
print(f"Found {len(trials)} trials to process.")

for trial in trials:
    record_id, nct_id, min_age, max_age, is_healthy, criteria_text, adult, child, older_adult = trial
    
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
            trial_graph["edges"].append({"type": "HAS_MAX_AGE", "target": max_age})
    
    # Add the bool columns as structured edges
    if is_healthy is not None: trial_graph["edges"].append({"type": "REQUIRES_HEALTHY_PATIENTS", "target": is_healthy})
    if adult is not None: trial_graph["edges"].append({"type": "ACCEPTS_ADULTS", "target": adult})
    if child is not None: trial_graph["edges"].append({"type": "ACCEPTS_CHILDREN", "target": child})
    if older_adult is not None: trial_graph["edges"].append({"type": "ACCEPTS_OLDER_ADULTS", "target": older_adult})
    
    # Process inclusions & exc
    inc_text, exc_text = split_criteria(criteria_text)
    cleaned_inc_text, cleaned_exc_text = clean_lines(inc_text), clean_lines(exc_text)
    
    for text_chunk in cleaned_inc_text:
        if len(text_chunk.split()) < 2: continue
        
        inputs = prepare_sequence(text_chunk, word_to_ix)
        mask = torch.ones(1, len(inputs), dtype=torch.bool)
        with torch.no_grad():
            predicted_tag_ids = model.decode(inputs.unsqueeze(0), mask)[0]
        
        predicted_tags = [ix_to_tag[tag_id] for tag_id in predicted_tag_ids]
        extracted_entities = extract_entities(text_chunk, predicted_tags)
        
        for entity in extracted_entities:
            if entity['tag'] == 'Disease':
                trial_graph["edges"].append({"type": "REQUIRES_CONDITION", "target": entity['text']})
            elif entity['tag'] == 'Neg-Disease':
                trial_graph["edges"].append({"type": "EXCLUDES_CONDITION", "target": entity['text']})
            elif entity['tag'] == 'Chemical':
                trial_graph["edges"].append({"type": "REQUIRES_CHEMICAL", "target": entity['text']})
    
    for text_chunk in cleaned_exc_text:
        if len(text_chunk.split()) < 2: continue
        
        inputs = prepare_sequence(text_chunk, word_to_ix)
        mask = torch.ones(1, len(inputs), dtype=torch.bool)
        with torch.no_grad():
            predicted_tag_ids = model.decode(inputs.unsqueeze(0), mask)[0]
        
        predicted_tags = [ix_to_tag[tag_id] for tag_id in predicted_tag_ids]
        extracted_entities = extract_entities(text_chunk, predicted_tags)
        
        for entity in extracted_entities:
            if entity['tag'] == 'Disease':
                trial_graph["edges"].append({"type": "EXCLUDES_CONDITION", "target": entity['text']}) # any disease found here are excluded
            elif entity['tag'] == 'Chemical':
                trial_graph["edges"].append({"type": "EXCLUDES_CHEMICAL", "target": entity['text']})

    # Save the fully enriched graph back to Postgres
    # cur.execute(
    #     "UPDATE ctgov.eligibilities SET extracted_graph = %s WHERE id = %s",
    #     (json.dumps(trial_graph), record_id)
    # )
    print(f"Trial id: {record_id}: {json.dumps(trial_graph)}\n")

# Commit changes to the db
conn.commit()
cur.close()
conn.close()
print("Processing complete.")