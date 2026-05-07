from tqdm import tqdm
import pandas as pd
import spacy
from sqlalchemy import create_engine, text
import re
import json
import medspacy
from patterns import ALL_PATTERNS, GENDER_MAP, IGNORE_ANATOMY, FEMALE_PROCEDURES, MALE_PROCEDURES, BOTH_PROCEDURES

DATABASE_URL = "postgresql://username:password@host:port/database_name"
engine = create_engine(DATABASE_URL)

print("Connecting to the database...")
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS processed_data;"))
    conn.commit()

print("Loading the scispaCy medical NLP model...")
nlp = medspacy.load("en_ner_bc5cdr_md", exclude=["parser"]) # Load scispaCy through Medspacy

ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(ALL_PATTERNS)

print("Fetching raw patient notes...")
df_raw = pd.read_sql("SELECT * FROM patients_raw.notes;", engine)

print("Running NLP extraction...")
extracted_records = []
data_tuples = list(zip(df_raw['transcription'].astype(str), df_raw.index))

for doc, note_id in nlp.pipe(tqdm(data_tuples, desc="Processing patient notes"), as_tuples=True, batch_size=50):  
    
    for ent in doc.ents:
        # 1. Skip Family History and Surgical Risks (Hypotheticals)
        if ent._.section_category in ["family_history", "patient_education"] or ent._.is_hypothetical:
            continue

        # 2. Skip historical pregnancy mentions (e.g., "History of two prior pregnancies")
        if ent.label_ == "PREGNANCY" and ent._.is_historical:
            continue

        # 3. Standardize extraction data
        numeric_value = None
        is_negated = ent._.is_negated # MedspaCy automatically handles "denies pregnancy"
        is_historical = ent._.is_historical # Capture past history status
        
        clean_entity_text = re.sub(r'[.,\s]*(PLAN|SUBJECTIVE|OBJECTIVE|ASSESSMENT)[:,\s\d]*$', '', ent.text, flags=re.IGNORECASE).strip()

        # Filter out anatomical noise
        if clean_entity_text.lower() in IGNORE_ANATOMY and ent.label_ == "DISEASE":
            continue
            
        if ent.label_ == "AGE":
            numbers = re.findall(r'\d+', ent.text)
            if numbers:
                numeric_value = float(numbers[0]) 

        extracted_records.append({
            "note_index": note_id, 
            "extracted_entity": clean_entity_text, 
            "medical_label": ent.label_,
            "numeric_value": numeric_value,
            "is_negated": is_negated,
            "is_historical": is_historical 
        })

df_entities = pd.DataFrame(extracted_records)
print("Formatting entities into graph edges...")

formatted_records = []

for note_id, group in df_entities.groupby('note_index'):
    edges = []
    seen_edges = set()
    active_conditions_count = 0 # Track active conditions for health status
    
    for _, row in group.iterrows():
        edge_data = (row['medical_label'], str(row['extracted_entity']).lower(), row.get('is_negated', False), row.get('is_historical', False))
        if edge_data in seen_edges:
            continue
        seen_edges.add(edge_data)
        
        if row['medical_label'] == 'AGE' and pd.notna(row['numeric_value']):
            edges.append({"type": "AGE", "target": float(row['numeric_value'] * 12)})

        elif row['medical_label'] == 'GENDER':
            standardized_gender = GENDER_MAP.get(str(row['extracted_entity']).lower(), "other")
            edges.append({"type": "HAS_GENDER", "target": standardized_gender})

        elif row['medical_label'] == 'RACE':
            edges.append({"type": "HAS_RACE", "target": str(row['extracted_entity']).lower()})
            
        elif row['medical_label'] == 'DISEASE':
            target_val = str(row['extracted_entity']).lower()
            if row.get('is_historical', False):
                edge_type = "HAD_PAST_CONDITION"
            elif row.get('is_negated', False):
                edge_type = "DOES_NOT_HAVE_CONDITION"
            else:
                edge_type = "HAS_CONDITION"
                active_conditions_count += 1 # Only count un-negated, active diseases
            edges.append({"type": edge_type, "target": target_val})
                
        elif row['medical_label'] == 'CHEMICAL':
            target_val = str(row['extracted_entity']).lower()
            if row.get('is_historical', False):
                edge_type = "TOOK_PAST_MEDICATION"
            else:
                edge_type = "TAKES_MEDICATION"
            edges.append({"type": edge_type, "target": target_val})
            
        elif row['medical_label'] == 'PREGNANCY':
            if row.get('is_negated', False):
                edges.append({"type": "IS_PREGNANT", "target": False})
            else:
                edges.append({"type": "IS_PREGNANT", "target": True})

    # Flag as healthy if we found zero active conditions
    if active_conditions_count == 0:
        edges.append({"type": "IS_HEALTHY", "target": True})
    else:
        edges.append({"type": "IS_HEALTHY", "target": False})

    formatted_records.append({
        "note_index": int(note_id),
        "clinical_graph": json.dumps({"edges": edges, "nodes": []})
    })

df_structured = pd.DataFrame(formatted_records)
print(f"Formatted {len(df_structured)} patient graphs. Pushing to database...")

df_structured.to_sql('patient_graphs', engine, schema='processed_data', if_exists='replace', index=False)
print("Transformation complete! The patients' data is securely stored.")