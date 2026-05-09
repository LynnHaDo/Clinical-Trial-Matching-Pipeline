import pandas as pd
import spacy
import json
from sqlalchemy import create_engine, text
from tqdm import tqdm

DATABASE_URL = "postgresql://username:password@host:port/database_name"
engine = create_engine(DATABASE_URL)

print("Connecting to the database...")
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS processed_data;"))
    conn.commit()

# Load the Medical Model
print("Loading the scispaCy medical NLP model...")
nlp = spacy.load("en_core_sci_sm")

print("Fetching raw patient notes...")
query = "SELECT * FROM patients_raw.notes;" 
df_raw = pd.read_sql(query, engine)
df_raw = df_raw.head(10)
# --- NLP Extraction with Progress Bar ---
print("Running NLP extraction...")
extracted_records = []
predicted_tuples = [] # Simplified for baseline evaluation

for index, row in tqdm(df_raw.iterrows(), total=df_raw.shape[0], desc="Processing Notes"):
    note_text = str(row['transcription']) 
    doc = nlp(note_text)
    
    for ent in doc.ents:
        target_text = ent.text.lower()
        
        extracted_records.append({
            "note_index": str(index), 
            "type": "ENTITY",
            "target": target_text
        })
        
        # Append tuple for baseline evaluation: (note_id, target)
        predicted_tuples.append((str(index), target_text))

# Save to Database
df_entities = pd.DataFrame(extracted_records)
df_entities.to_sql(name='patient_entities', con=engine, schema='processed_data', if_exists='replace', index=False)
print("Transformation and Database load complete!")

# --- EVALUATION SECTION ---

def load_baseline_ground_truth(json_filepath):
    """
    Loads JSON but ignores negations, demographics, and doc-level flags.
    Returns a list of (note_id, target_text) tuples.
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)
        
    gold_standard = []
    # Ignore these types entirely for the baseline evaluation
    ignored_types = {"DOES_NOT_HAVE_CONDITION", "IS_HEALTHY", "AGE", "HAS_GENDER", "HAS_RACE"}
    
    for note_id, attributes in data.items():
        for attr in attributes:
            if attr["type"] not in ignored_types:
                gold_standard.append((str(note_id), str(attr["target"]).lower()))
                
    return gold_standard

def run_evaluation(predictions, ground_truth):
    """Calculates baseline NER metrics using set operations."""
    pred_set = set(predictions)
    gold_set = set(ground_truth)
    
    tp = len(pred_set.intersection(gold_set))  
    fp = len(pred_set - gold_set)             
    fn = len(gold_set - pred_set)             
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n--- Baseline Model Evaluation Results ---")
    print(f"Total Extracted Entities: {len(pred_set)}")
    print(f"Total Valid Ground Truth Entities: {len(gold_set)}")
    print("-----------------------------------------")
    print(f"True Positives (Correct): {tp}")
    print(f"False Positives (Noise):   {fp}")
    print(f"False Negatives (Missed):  {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

# Execute Evaluation
print("\nLoading Ground Truth and Running Evaluation...")
try:
    gold_tuples = load_baseline_ground_truth('truth_table_02.json')
    run_evaluation(predicted_tuples, gold_tuples)
except FileNotFoundError:
    print("Error: 'truth_table_02.json' not found. Ensure it is in the same directory as the script.")