import json
import torch
import os
import psycopg2

# Connect to db
db_url = os.environ.get("DATABASE_URL")
conn = psycopg2.connect(db_url)
cur = conn.cursor()

# Fetch the trials where the population text hasn't been processed yet
cur.execute("SELECT trial_id, minimum_age, maximum_age, population FROM trials WHERE population IS NOT NULL;")
trials = cur.fetchall()

for trial in trials:
    trial_id, min_age, max_age, population_text = trial
    
    # 2. Prepare the Trial Subgraph (Gt)
    trial_graph = {
        "nodes": [],
        "edges": []
    }
    
    # Add structured relational data as edges immediately
    if min_age:
        trial_graph["edges"].append({"type": "HAS_MIN_AGE", "target": min_age})
    
    # 3. Process the unstructured 'population' text
    # (Assume 'prepare_sequence' converts text to token IDs)
    inputs = prepare_sequence(population_text, word_to_ix)
    mask = torch.ones(1, len(inputs), dtype=torch.uint8)
    
    # Run the BiLSTM-CRF model!
    with torch.no_grad():
        predicted_tag_ids = model.decode(inputs.unsqueeze(0), mask)[0]
    
    # Convert IDs back to strings (e.g., [0, 1, 2, 0]) -> ['O', 'B-Disease', 'I-Disease', 'O']
    predicted_tags = [ix_to_tag[tag_id] for tag_id in predicted_tag_ids]
    
    # 4. Parse the tags into Graph Edges
    # (Assume 'extract_entities' groups B and I tags into full words)
    extracted_entities = extract_entities(population_text, predicted_tags)
    
    for entity in extracted_entities:
        if entity['tag'] == 'Disease':
            trial_graph["edges"].append({"type": "REQUIRES_CONDITION", "target": entity['text']})
        elif entity['tag'] == 'Neg-Disease':
            trial_graph["edges"].append({"type": "EXCLUDES_CONDITION", "target": entity['text']})

    # 5. Save the generated graph back into PostgreSQL as JSON
    cur.execute(
        "UPDATE trials SET extracted_graph = %s WHERE trial_id = %s",
        (json.dumps(trial_graph), trial_id)
    )

# Commit changes to the database
conn.commit()
cur.close()
conn.close()