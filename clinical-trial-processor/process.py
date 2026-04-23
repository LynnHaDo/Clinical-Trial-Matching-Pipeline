import json
import torch
import os
import psycopg2

from constants import DATABASE_URL_KEY, MODEL_PARAMS
from encoder import ClinicalTrialEncoder
from utils import prepare_sequence, extract_entities

# ==========================================
# Load model
# ==========================================

# Load the model
model_path = os.path.abspath(os.path.join(MODEL_PARAMS.WEIGHTS_SAVE_DIR.value, MODEL_PARAMS.WEIGHTS_NAME.value))
content = torch.load(model_path, map_location=torch.device('cpu'))

# Extract the critical dictionaries
word_to_ix = content['word_to_ix']
tag_to_ix = content['tag_to_ix']
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

model = ClinicalTrialEncoder(
    vocab_size=len(word_to_ix),
    tag_to_ix_size=len(tag_to_ix),
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

# Connect to db
db_url = os.environ.get(DATABASE_URL_KEY)
conn = psycopg2.connect(db_url)
cur = conn.cursor()

print("Connected to PostgreSQL. Checking schema...")

cur.execute("""
            ALTER TABLE ctgov.eligibilities
            ADD COLUMN IF NOT EXISTS extracted_graph JSONB
            """)
conn.commit()
print("Schema ready!")

# ==========================================
# Execution loop
# ==========================================

print("Fetching trials...")
# Fetch the trials where the population text hasn't been processed yet
cur.execute("""
            SELECT 
                id, nct_id, minimum_age, maximum_age, population, 
                criteria, adult, child, older_adult 
            FROM ctgov.eligibilities 
            WHERE (population IS NOT NULL OR criteria IS NOT NULL)
            AND extracted_graph IS NULL
            LIMIT 100;
            """)
trials = cur.fetchall()
print(f"Found {len(trials)} trials to process.")

for trial in trials:
    record_id, nct_id, min_age, max_age, population_text, criteria_text, adult, child, older_adult = trial
    
    trial_graph = {
        "nodes": [],
        "edges": []
    }
    
    # Add the bool columns as structured edges
    if min_age: trial_graph["edges"].append({"type": "HAS_MIN_AGE", "target": min_age})
    if max_age: trial_graph["edges"].append({"type": "HAS_MAX_AGE", "target": max_age})
    if adult is not None: trial_graph["edges"].append({"type": "ACCEPTS_ADULTS", "target": adult})
    if child is not None: trial_graph["edges"].append({"type": "ACCEPTS_CHILDREN", "target": child})
    if older_adult is not None: trial_graph["edges"].append({"type": "ACCEPTS_OLDER_ADULTS", "target": older_adult})
    
    # Combine the unstructured text
    # We use 'or ""' to handle None types safely
    raw_unstructured_texts = []
    if population_text:
        raw_unstructured_texts.append(population_text)
    if criteria_text:
        # Split criteria by newlines
        criteria_lines = [line.strip() for line in criteria_text.split('\n') if line.strip()]
        raw_unstructured_texts.extend(criteria_lines)
    
    # Process the text chunk by chunk
    for text_chunk in raw_unstructured_texts:
        # Skip weirdly short artifacts
        if len(text_chunk.split()) < 2:
            continue
            
        inputs = prepare_sequence(text_chunk, word_to_ix)
        mask = torch.ones(1, len(inputs), dtype=torch.uint8)
        
        with torch.no_grad():
            predicted_tag_ids = model.decode(inputs.unsqueeze(0), mask)[0]
        
        predicted_tags = [ix_to_tag[tag_id] for tag_id in predicted_tag_ids]
        extracted_entities = extract_entities(text_chunk, predicted_tags)
        
        # Add the dynamically extracted NLP edges
        for entity in extracted_entities:
            if entity['tag'] == 'Disease':
                trial_graph["edges"].append({"type": "REQUIRES_CONDITION", "target": entity['text']})
            elif entity['tag'] == 'Neg-Disease':
                trial_graph["edges"].append({"type": "EXCLUDES_CONDITION", "target": entity['text']})

    # Save the fully enriched graph back to Postgres
    cur.execute(
        "UPDATE ctgov.eligibilities SET extracted_graph = %s WHERE id = %s",
        (json.dumps(trial_graph), record_id)
    )

# Commit changes to the db
conn.commit()
cur.close()
conn.close()
print("Processing complete.")