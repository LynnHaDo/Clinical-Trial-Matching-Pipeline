import json
import math

def calculate_precision_at_k(predictions, ground_truth, k):
    total_precision = 0.0
    num_patients = len(ground_truth)
    
    for patient_id, true_trials in ground_truth.items():
        # Handle cases where the model didn't output predictions for a patient
        if patient_id not in predictions:
            continue
            
        # Get the top K predictions 
        top_k_predictions = predictions[patient_id][:k]
        
        # Count how many of the top K predictions are actually in the ground truth
        hits = 0
        for trial in top_k_predictions:
            if trial in true_trials:
                hits += 1
                
        # Calculate precision for this specific patient
        patient_precision = hits / k
        total_precision += patient_precision
        
    # Return the average precision across all patients
    return total_precision / num_patients
def calculate_ndcg_at_k(predictions, ground_truth, k):
    total_ndcg = 0.0
    num_patients = len(ground_truth)
    
    for patient_id, true_trials in ground_truth.items():
        if patient_id not in predictions:
            continue
            
        top_k_predictions = predictions[patient_id][:k]
        
        # 1. Calculate actual DCG
        dcg = 0.0
        for i, trial in enumerate(top_k_predictions):
            if trial in true_trials:
                rank = i + 1 # Ranks are 1-indexed (1, 2, 3...)
                # Binary relevance means the numerator is always 1
                dcg += 1.0 / math.log2(rank + 1)
                
        # 2. Calculate Ideal DCG (IDCG)
        idcg = 0.0
        num_ideal_hits = min(len(true_trials), k)
        for i in range(num_ideal_hits):
            rank = i + 1
            idcg += 1.0 / math.log2(rank + 1)
            
        # 3. Calculate Normalized DCG for this patient
        if idcg > 0:
            patient_ndcg = dcg / idcg
        else:
            # Edge case: If the ground truth has 0 valid trials for a patient
            patient_ndcg = 0.0 
            
        total_ndcg += patient_ndcg
        
    # Return the average nDCG across all patients
    return total_ndcg / num_patients

if __name__ == "__main__":
    # 1. Load the predictions generated 
    try:
        with open("model_predictions.json", "r") as f:
            model_predictions = json.load(f)
    except FileNotFoundError:
        print("Error: model_predictions.json not found. Run your matching pipeline first.")
        exit()

    # 2. Load Gold Label
    try:
        with open("ground_truth.json", "r") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print("Error: ground_truth.json not found. Please create your labeled dataset file.")
        exit()

    # 3. Calculate and Print Metrics
    p_at_5 = calculate_precision_at_k(model_predictions, ground_truth, k=5)
    p_at_10 = calculate_precision_at_k(model_predictions, ground_truth, k=10)
    ndcg_at_5  = calculate_ndcg_at_k(model_predictions, ground_truth, k=5)
    ndcg_at_10 = calculate_ndcg_at_k(model_predictions, ground_truth, k=10)

    print(f"\n{'='*40}")
    print(f"EVALUATION RESULTS (N={len(ground_truth)} Patients)")
    print(f"{'='*40}")
    print(f"Precision@5:  {p_at_5:.4f}")
    print(f"Precision@10: {p_at_10:.4f}")
    print(f"nDCG@5:       {ndcg_at_5:.4f}")
    print(f"nDCG@10:      {ndcg_at_10:.4f}")
    print(f"{'='*40}\n")