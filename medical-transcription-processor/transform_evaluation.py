import json
import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:rootroot@127.0.0.1:5433/aact"

def load_ground_truth(filepath="truth_table_02.json"):
    with open(filepath, "r") as f:
        return json.load(f)

def get_pipeline_extractions(engine, note_indices):
    """Fetches the generated graphs from the database for our specific test notes."""
    indices_str = ",".join(note_indices)
    query = f"SELECT note_index, clinical_graph FROM processed_data.patient_graphs WHERE note_index IN ({indices_str});"
    df = pd.read_sql(query, engine)
    
    pipeline_results = {}
    for _, row in df.iterrows():
        graph_data = json.loads(row['clinical_graph'])
        pipeline_results[str(row['note_index'])] = graph_data.get("edges", [])
    
    return pipeline_results

def calculate_metrics(ground_truth, pipeline_results):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for note_id, truth_edges in ground_truth.items():
        predicted_edges = pipeline_results.get(note_id, [])
        
        # Keep the set generation to ensure we don't have exact duplicates
        truth_set = set((d['type'], str(d['target']).lower()) for d in truth_edges)
        pred_set = set((d['type'], str(d['target']).lower()) for d in predicted_edges)

        matched_truth = set()
        matched_pred = set()

        for t_type, t_target in truth_set:
            for p_type, p_target in pred_set:
                
                # Check if types match AND if one target string is inside the other
                if t_type == p_type and (t_target in p_target or p_target in t_target):
                    
                    # Ensure we don't map multiple truths to the exact same prediction
                    if (p_type, p_target) not in matched_pred:
                        matched_truth.add((t_type, t_target))
                        matched_pred.add((p_type, p_target))
                        break # Match found, move to the next truth entity

        # Calculate metrics for this specific note
        tp = len(matched_truth)
        fp = len(pred_set) - len(matched_pred) # Predictions that never matched anything
        fn = len(truth_set) - len(matched_truth) # Truths that were never found

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Prevent division by zero errors
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Pipeline Evaluation Metrics ---")
    print(f"True Positives (Correct): {total_tp}")
    print(f"False Positives (Noise): {total_fp}")
    print(f"False Negatives (Missed): {total_fn}")
    print("-" * 30)
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}\n")

if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)
    truth_data = load_ground_truth()
    
    # Get the string IDs of the notes we want to test
    test_indices = list(truth_data.keys()) 
    
    predictions = get_pipeline_extractions(engine, test_indices)
    calculate_metrics(truth_data, predictions)