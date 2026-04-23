import pandas as pd
import spacy
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://username:password@host:port/database_name"
engine = create_engine(DATABASE_URL)

print("Connecting to the database...")
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS processed_data;"))
    conn.commit()
    print("Schema 'processed_data' is ready.")

# Load the Medical Model
print("Loading the scispaCy medical NLP model...")
nlp = spacy.load("en_core_sci_sm")

# Read the Unstructured Notes
print("Fetching raw patient notes...")
query = "SELECT * FROM patients_raw.notes;" 
df_raw = pd.read_sql(query, engine)

# NLP Extraction
print("Running NLP extraction...")
extracted_records = []

for index, row in df_raw.iterrows():
    note_text = str(row['transcription']) 
    doc = nlp(note_text)
    
    for ent in doc.ents:
        extracted_records.append({
            "note_index": index, 
            "extracted_entity": ent.text,
            "medical_label": ent.label_
        })

# Update Docker
df_entities = pd.DataFrame(extracted_records)
print(f"Extracted {len(df_entities)} individual medical entities. Pushing to database...")

df_entities.to_sql(
    name='patient_entities',
    con=engine,
    schema='processed_data',
    if_exists='replace',
    index=False
)

print("Transformation complete! The clean data is securely stored in 'processed_data.patient_entities'.")