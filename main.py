import argparse
import json
import os
from rich.console import Console
from rich.table import Table
import psycopg2
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

from clinical_trial_processor.constants import CLINICAL_TRIALS_SEMANTIC_CRITERIA_TABLE_NAME
from constants import CLINICAL_BERT_MODEL_NAME, DATABASE_URL_KEY, MATCHING_DISTANCE_THRESHOLD, MEDICAL_TRANSCRIPTIONS_TABLE_NAME
from constructs.patient import Patient
from utils import get_embedding

# ==========================================
# Set up arguments
# ==========================================
parser = argparse.ArgumentParser(description="This script sets up the database to embed clinical trials extracted into vectors.")
parser.add_argument("input_patient_notes", help="Path to input patient notes (list of ids) in .txt")
parser.add_argument("--k", help="Number of trials suggested for each patient")

args = parser.parse_args()

# ==========================================
# Connect to db and set up schema
# ==========================================

load_dotenv()

# Connect to db
db_url = os.environ.get(DATABASE_URL_KEY)
conn = psycopg2.connect(db_url)
cur = conn.cursor()

# ==========================================
# Trials fetching
# ==========================================

def get_eligible_trials(cur, trial_ids, patient):
    # Parsing biological sex
    op_biological_sex_edge = json.dumps([{
        "type": "IGNORE"
    }])
    if patient.biological_sex == "Female":
        op_biological_sex_edge = json.dumps([{
            "type": "REQUIRES_BIOLOGICAL_SEX",
            "target": "Male"
    }])
    elif patient.biological_sex == "Male":
        op_biological_sex_edge = json.dumps([{
            "type": "REQUIRES_BIOLOGICAL_SEX",
            "target": "Female"
    }])
    
    # Parsing pregnancy status
    op_pregnancy_status_edge = json.dumps([{
            "type": "IGNORE"
    }])
    if patient.is_pregnant is True:
        op_pregnancy_status_edge = json.dumps([{
            "type": "REQUIRES_PREGNANCY",
            "target": False
        }])
    elif patient.is_pregnant is False:
        op_pregnancy_status_edge = json.dumps([{
            "type": "REQUIRES_PREGNANCY",
            "target": True
        }])
    
    query = f"""
        SELECT
            e.id,
            e.nct_id,
            c.edge_type,
            c.edge_text,
            c.edge_embedding::text
        FROM ctgov.eligibilities e
        LEFT JOIN {CLINICAL_TRIALS_SEMANTIC_CRITERIA_TABLE_NAME} c ON e.id = c.trial_id
        WHERE
            e.nct_id = ANY(%s) 
            --- Strict demographic & Boolean filters
            AND %s >= COALESCE(e.min_age_months, 0)
            AND %s <= COALESCE(e.max_age_months, 99999)
            AND (e.healthy_volunteers IS NULL OR e.healthy_volunteers = %s)
            AND (e.gender = 'ALL' OR e.gender = %s)
            --- JSON Graph filters (Pregnancy/Biological Sex)
            AND NOT (e.extracted_graph->'edges' @> CAST(%s AS jsonb))
            AND NOT (e.extracted_graph->'edges' @> CAST(%s AS jsonb))
    """
    
    cur.execute(query, (
        trial_ids,
        patient.age_months,
        patient.age_months,
        patient.is_healthy,
        patient.gender.upper() if patient.gender is not None else 'ALL',
        op_biological_sex_edge,
        op_pregnancy_status_edge
    ))
    
    return cur.fetchall()

def get_trials_from_sample(filepath='trials.txt'):
    try:
        with open(filepath, 'r') as f:
            target_ids = [line.strip() for line in f if line.strip()]
            return target_ids
    except FileNotFoundError:
        print(f'Error: The file {filepath} was not found.')
        return []

# ==========================================
# Matching calculations
# ==========================================

def evaluate_criteria(category, patient_condition_vectors, c_vec, c_type, trial_data):
    if category in c_type:
        for p_vec in patient_condition_vectors:
            distance = cosine(c_vec, p_vec)
                    
            if distance < MATCHING_DISTANCE_THRESHOLD:
                if c_type == f'EXCLUDES_{category}':
                    trial_data["vetoed"] = True # VETO! Patient has a banned condition
                    break
                elif c_type == f'REQUIRES_{category}':
                    trial_data["score"] += 1 # REWARD! Patient has a required condition

def score_trials(eligible_trials, patient_condition_vectors, patient_med_vectors, k):
    trials = {}
    
    for row in eligible_trials:
        t_id, nct_id, edge_type, edge_text, vector_str = row
        
        if t_id not in trials:
            trials[t_id] = {
                'nct_id': nct_id,
                'score': 0,
                'vetoed': False,
                'criteria': []
            }
        
        if edge_type:
            vector = [float(x) for x in vector_str.strip('[]').split(',')]
            trials[t_id]['criteria'].append({
                "type": edge_type,
                "text": edge_text,
                "vector": vector
            })
    
    valid_trials = []
    
    for t_id, trial_data in trials.items():
        for criteria in trial_data['criteria']:
            if trial_data['vetoed']: break
            
            c_type = criteria['type']
            c_vec = criteria['vector']

            # Evaluate criteria
            evaluate_criteria('CONDITION', patient_condition_vectors, c_vec, c_type, trial_data)
            evaluate_criteria('CHEMICAL', patient_med_vectors, c_vec, c_type, trial_data)
            # evaluate_criteria('PROCEDURE', patient_prior_procedures_vectors, c_vec, c_type, trial_data)
            # evaluate_criteria('OBSERVATION', patient_condition_vectors, c_vec, c_type, trial_data)
            # evaluate_criteria('DEVICE', patient_condition_vectors, c_vec, c_type, trial_data)
        
        if not trial_data['vetoed']:
            valid_trials.append(trial_data)
    
    valid_trials.sort(key=lambda x: x['score'], reverse=True)
    top_k = valid_trials[:k]
    return [(trial['nct_id'], trial['score']) for trial in top_k]

# ==========================================
# Load model for input patient notes
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(CLINICAL_BERT_MODEL_NAME)
model = AutoModel.from_pretrained(CLINICAL_BERT_MODEL_NAME)

# ==========================================
# Retrieve patient's extracted graph
# ==========================================
def get_patient_graph(patient_id, cur):
    cur.execute(
        f"SELECT extracted_graph FROM {MEDICAL_TRANSCRIPTIONS_TABLE_NAME} WHERE id = %s;",
        (patient_id,)
    )
        
    result = cur.fetchone()
        
    if result:
        return result[0]
    else:
        print(f"No record found for ID: {patient_id}")
        return None
    
# ==========================================
# Prepare args
# ==========================================

k = int(args.k) if args.k else 3
trial_ids_sample = get_trials_from_sample()

# ==========================================
# Initialize rich console for table formatting
# ==========================================
console = Console()
table = Table(title="Clinical Trial Matching Results", show_header=True, header_style="bold magenta")
table.add_column("Patient ID", style="cyan", width=12, justify="center")
table.add_column("Matching Trials (ID, Score)", style="green")

with console.status("[bold blue]Running patient matching pipeline...", spinner="dots"):
    with open(args.input_patient_notes, 'r') as file:
        for line in file:
            patient_id = int(line.strip())
            patient_graph = get_patient_graph(patient_id, cur)
            if not patient_graph:
                table.add_row(str(patient_id), "[red]Graph not found in DB[/red]")
            patient_edges = patient_graph['edges']
            patient = Patient(patient_edges)

            # Embed the Patient's Semantic Data
            patient_condition_vectors = [get_embedding(cond, tokenizer, model) for cond in patient.conditions]
            patient_med_vectors = [get_embedding(med, tokenizer, model) for med in patient.medications]

            # Load eligible trials
            eligible_trials = get_eligible_trials(cur, trial_ids_sample, patient)
            valid_trials = score_trials(eligible_trials, patient_condition_vectors, patient_med_vectors, k)
            if valid_trials:
                trials_str = "\n".join([f"{nct} (Score: {score})" for nct, score in valid_trials])
            else:
                trials_str = "[yellow]No eligible trials found[/yellow]"
            
            table.add_row(str(patient_id), trials_str)

# ==========================================
# Print result
# ==========================================
console.print('\n')
console.print(table)

cur.close()
conn.close()
