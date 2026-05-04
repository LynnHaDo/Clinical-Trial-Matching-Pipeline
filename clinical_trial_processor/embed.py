import os
import json
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from transformers import AutoModel, AutoTokenizer

from clinical_trial_processor.constants import DATABASE_URL_KEY, CLINICAL_TRIALS_SEMANTIC_CRITERIA_TABLE_NAME, POSTGRES_SQL_FETCH_SIZE, POSTGRES_MAX_PROCESSING_SIZE, CRITERIA_BOOLEAN_EDGE_TYPES
from constants import CLINICAL_BERT_MODEL_NAME
from utils import get_embedding

load_dotenv()

# Connect to db
db_url = os.environ.get(DATABASE_URL_KEY)
conn = psycopg2.connect(db_url)
cur = conn.cursor()

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained(CLINICAL_BERT_MODEL_NAME)
model = AutoModel.from_pretrained(CLINICAL_BERT_MODEL_NAME)

def set_up_db(cur, conn):
    """Sets up DB for trial matching"""
    print("Connected to PostgreSQL. Checking schema...")

    # Ensure pgvector is enabled
    cur.execute("DROP TABLE IF EXISTS trial_semantic_criteria; CREATE EXTENSION IF NOT EXISTS vector;")
    # Create a new table for the semantic nodes
    cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {CLINICAL_TRIALS_SEMANTIC_CRITERIA_TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                trial_id INTEGER REFERENCES ctgov.eligibilities(id),
                edge_type VARCHAR(50),
                edge_text TEXT,
                edge_embedding vector(768)
                );
                """)
    # Indexing for faster vector searching
    cur.execute(f"CREATE INDEX ON {CLINICAL_TRIALS_SEMANTIC_CRITERIA_TABLE_NAME} USING hnsw (edge_embedding vector_cosine_ops);")
    conn.commit()
    print("Schema ready!")

def embed_trials(cur, conn):
    count = 0
    while count < POSTGRES_MAX_PROCESSING_SIZE:
        # Find trials that have a graph but haven't been embedded yet
        cur.execute(f"""
            SELECT e.id, e.extracted_graph 
            FROM ctgov.eligibilities e 
            LEFT JOIN {CLINICAL_TRIALS_SEMANTIC_CRITERIA_TABLE_NAME} c ON e.id = c.trial_id 
            WHERE e.extracted_graph IS NOT NULL 
            AND c.trial_id IS NULL 
            LIMIT {POSTGRES_SQL_FETCH_SIZE};
        """)
        
        trials = cur.fetchall()
        if not trials:
            print("All trials are fully embedded!")
            return

        print(f"Processing {len(trials)} trials...")
        
        records_to_insert = []
        
        # Iterate through the trials and extract the semantic edges
        for trial_id, graph_json in trials:
            # Check if the graph is stored as a string or a dict
            graph = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
            
            if "edges" not in graph:
                continue
                
            for edge in graph["edges"]:
                edge_type = edge.get("type", "")
                edge_text = edge.get("target", "")
                
                # Skip boolean or demographic edges (they belong in standard columns)
                if edge_type in CRITERIA_BOOLEAN_EDGE_TYPES:
                    continue
                    
                # If it's a valid semantic concept, embed it
                if edge_text and isinstance(edge_text, str):
                    vector = get_embedding(edge_text, tokenizer, model)
                    
                    # Format required by psycopg2 for batch insertion
                    records_to_insert.append((
                        trial_id, 
                        edge_type, 
                        edge_text, 
                        vector.tolist() # pgvector needs standard python lists
                    ))

        # Batch insert into the database
        if records_to_insert:
            print(f"Inserting {len(records_to_insert)} semantic criteria vectors...")
            
            insert_query = f"""
                INSERT INTO {CLINICAL_TRIALS_SEMANTIC_CRITERIA_TABLE_NAME}
                (trial_id, edge_type, edge_text, edge_embedding) 
                VALUES %s
            """
            
            execute_values(cur, insert_query, records_to_insert)
            conn.commit()
            print("Batch successful!")
        
        count += POSTGRES_SQL_FETCH_SIZE

if __name__ == "__main__":
    set_up_db(cur, conn)
    embed_trials(cur, conn)
    cur.close()
    conn.close()