import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://username:password@host:port/database_name"
engine = create_engine(DATABASE_URL)

print("Connecting to the database...")
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS patients_raw;"))
    conn.commit()
    print("Schema 'patients_raw' is ready.")

print("Loading the MTSamples data...")
df_patients = pd.read_csv('/datasets/mtsamples.csv')
df_patients = df_patients.dropna(subset=['transcription'])

print("Pushing data into the Docker container...")
df_patients.to_sql(
    name='notes', 
    con=engine, 
    schema='patients_raw', 
    if_exists='replace', 
    index=False
)

print("Success! Patient notes are officially merged and ready for the scispaCy NLP layer.")