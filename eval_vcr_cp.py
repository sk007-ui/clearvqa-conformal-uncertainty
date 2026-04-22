import os
import pickle
import json
import numpy as np
from scipy.special import softmax

def load_data(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def compute_lac_scores(probs, labels):
    n = len(probs)
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = 1.0 - probs[i, labels[i]]
    return scores

def compute_aps_scores(probs, labels):
    n = len(probs)
    scores = np.zeros(n)
    for i in range(n):
        p = probs[i]
        sort_idx = np.argsort(p)[::-1]
        rank = np.where(sort_idx == labels[i])[0][0]
        scores[i] = np.sum(p[sort_idx[:rank+1]])
    return scores

def get_q_hat(scores, alpha):
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))

def evaluate_lac(probs, labels, q_hat):
    sets = [np.where(1.0 - p <= q_hat)[0] for p in probs]
    sizes = [len(s) for s in sets]
    coverage = np.mean([labels[i] in sets[i] for i in range(len(probs))])
    return float(np.mean(sizes)), float(coverage)

def evaluate_aps(probs, labels, q_hat):
    sizes = []
    coverage = []
    for i in range(len(probs)):
        p = probs[i]
        sort_idx = np.argsort(p)[::-1]
        cumsum = np.cumsum(p[sort_idx])
        idx = np.where(cumsum >= q_hat)[0]
        if len(idx) == 0:
            k = len(p)
        else:
            k = idx[0] + 1
            
        pred_set = sort_idx[:k]
        sizes.append(k)
        coverage.append(labels[i] in pred_set)
    return float(np.mean(sizes)), float(np.mean(coverage))

def main():
    filepath = os.path.join("outputs_vcr", "vcr_final_logits.pkl")
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        return
        
    data = load_data(filepath)
    print(f"Loaded {len(data)} samples from VCR.")
    
    labels = np.array([ex["answer_idx"] for ex in data])
    clear_logits = np.array([ex["logits_clear"] for ex in data])
    ambig_logits = np.array([ex["logits_ambig"] for ex in data])
    
    clear_probs = softmax(clear_logits, axis=1)
    ambig_probs = softmax(ambig_logits, axis=1)
    
    # Split into cal and test (50/50)
    CAL_RATIO = 0.5
    n_cal = int(CAL_RATIO * len(data))
    
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    cal_idx, test_idx = indices[:n_cal], indices[n_cal:]
    print(f"Calibration set: {len(cal_idx)} samples")
    print(f"Test set: {len(test_idx)} samples")
    
    res = {}
    alpha = 0.1  # 90% coverage target
    
    for split_name, probs in [("clear", clear_probs), ("ambig", ambig_probs)]:
        cal_probs, test_probs = probs[cal_idx], probs[test_idx]
        cal_labels, test_labels = labels[cal_idx], labels[test_idx]
        
        # LAC
        lac_scores = compute_lac_scores(cal_probs, cal_labels)
        lac_q = get_q_hat(lac_scores, alpha)
        lac_avg_size, lac_cov = evaluate_lac(test_probs, test_labels, lac_q)
        
        # APS
        aps_scores = compute_aps_scores(cal_probs, cal_labels)
        aps_q = get_q_hat(aps_scores, alpha)
        aps_avg_size, aps_cov = evaluate_aps(test_probs, test_labels, aps_q)
        
        res[split_name] = {
            "LAC": {
                "avg_set_size": lac_avg_size, 
                "coverage": lac_cov, 
                "threshold": lac_q
            },
            "APS": {
                "avg_set_size": aps_avg_size, 
                "coverage": aps_cov, 
                "threshold": aps_q
            }
        }
        
    print("\nResults:")
    print(json.dumps(res, indent=4))
    
    out_file = "cp_vcr_results.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=4)
    print(f"\nSaved results to {out_file}")

if __name__ == "__main__":
    main()
