import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

from dotenv import load_dotenv
from constants import DATABASE_URL_KEY, POSTGRES_SQL_PROCESSING_SIZE

# ==========================================
# Connect to db and set up schema
# ==========================================

load_dotenv()

# Connect to db
db_url = os.environ.get(DATABASE_URL_KEY)
engine = create_engine(db_url)

def load_mimic_to_postgres():
    # Configuration
    csv_file_path = os.path.abspath(os.path.join('./datasets', "discharge.csv"))
    schema_name = "ctgov"
    table_name = "discharge"
    chunk_size = POSTGRES_SQL_PROCESSING_SIZE
    
    print(f"Starting import of {csv_file_path} to {schema_name}.{table_name}...")

    # Read and insert in chunks
    # Pandas automatically handles the multi-line "text" column enclosed in double quotes
    try:
        for i, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunk_size)):
            print(f"Processing chunk {i + 1} (Rows {i * chunk_size} to {(i + 1) * chunk_size})...")
            
            # Convert datetime columns to native datetime objects before insertion
            chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
            chunk['storetime'] = pd.to_datetime(chunk['storetime'], errors='coerce')

            # Write the chunk to the PostgreSQL database
            chunk.to_sql(
                name=table_name,
                schema=schema_name,
                con=engine,
                if_exists="replace" if i == 0 else "append", # Replace table on first chunk, append after
                index=False
            )
            
        print("Successfully completed data import!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    load_mimic_to_postgres()
    print("Processing complete.")