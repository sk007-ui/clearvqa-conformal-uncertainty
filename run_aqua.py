"""
run_aqua.py
-----------
Converted from '2B second file.ipynb'.
Runs Qwen2-VL-2B-Instruct on AQuA (samples 0-17000),
per-level conformal prediction (LAC + APS), and saves results.

Supports checkpoint resuming so long-running jobs can safely restart.
"""

import json
import os
import pickle
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.special import softmax
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR  = "outputs_qwen2vl"
MAX_SAMPLES = 17000          # samples 0..17000
CAL_RATIO   = 0.5
ALPHA       = 0.1

OPTIONS = ["A", "B", "C", "D", "E", "F"]
LABELS  = ["A", "B", "C", "D", "E", "F"]

# ── Step 1: Load & convert AQuA dataset ────────────────────────────────────────
def convert_aqua_to_mcq(split="train", max_samples=None):
    """Load jihyoung/AQuA and convert to 6-choice MCQ format."""
    ds = load_dataset("jihyoung/AQuA", split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    result = []
    labels = ["A", "B", "C", "D"]
    all_answers = [ex["answer"] for ex in ds]

    for idx, example in enumerate(ds):
        correct = example["answer"]

        wrong = []
        for ans in all_answers:
            if ans != correct and len(wrong) < 3:
                wrong.append(ans)
        while len(wrong) < 3:
            wrong.append("other")

        options = [correct] + wrong[:3]
        np.random.seed(idx)          # deterministic shuffle per example
        np.random.shuffle(options)

        choices = {labels[i]: options[i] for i in range(4)}
        choices["E"] = "I don't know"
        choices["F"] = "None of the above"

        correct_label = next(k for k, v in choices.items() if v == correct)

        result.append({
            "source":      "AQUA",
            "task":        "QA",
            "question_id": idx,
            "question":    example["question"],
            "choices":     choices,
            "answer":      correct_label,
            "id":          idx,
            "image":       example["image"],   # PIL image from HF dataset
            "level":       example["level"],
        })

    return result


# ── Step 2: Build model prompt ──────────────────────────────────────────────────
def build_prompt(example):
    prompt = "Question: " + example["question"] + "\nChoices:\n"
    for k, v in example["choices"].items():
        prompt += k + ". " + str(v) + "\n"
    prompt += "Answer with only the letter (A, B, C, D, E, or F)."
    return prompt


# ── Step 3: Run Qwen2-VL inference with checkpointing ──────────────────────────
def get_qwen_logits(data, output_dir, processor, model, option_ids,
                    tag="aqua"):
    """
    Run inference over `data` (list of MCQ dicts).
    Saves a checkpoint every 100 samples so the job can resume.
    Returns path to the final .pkl file.
    """
    os.makedirs(output_dir, exist_ok=True)

    save_file    = os.path.join(output_dir, f"qwen_{tag}_{len(data)}.pkl")
    ckpt_file    = save_file + ".ckpt.pkl"   # incremental checkpoint

    # Load existing checkpoint if present
    if os.path.exists(ckpt_file):
        with open(ckpt_file, "rb") as f:
            all_outputs = pickle.load(f)
        start_idx = len(all_outputs)
        print(f"Resuming from checkpoint: {start_idx}/{len(data)} done")
    else:
        all_outputs = []
        start_idx   = 0

    errors = 0
    CKPT_EVERY = 100

    for i, example in enumerate(tqdm(data[start_idx:], initial=start_idx,
                                     total=len(data), desc=f"Inference [{tag}]")):
        try:
            # AQuA images are PIL objects embedded in the HF dataset
            if isinstance(example["image"], str):
                image = Image.open(example["image"]).convert("RGB")
            else:
                image = example["image"].convert("RGB")

            prompt_text = build_prompt(example)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text":  prompt_text},
                    ],
                }
            ]

            text   = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            logits         = outputs.logits[:, -1, :].squeeze(0)
            logits_options = logits[option_ids].float().cpu().numpy()

            all_outputs.append({
                "id":             example["id"],
                "logits_options": logits_options,
                "answer":         example["answer"],
                "level":          example.get("level", 0),
            })

        except Exception as e:
            errors += 1
            print(f"\n[ERROR] id={example['id']}: {e}")
            all_outputs.append({
                "id":             example["id"],
                "logits_options": np.zeros(6),
                "answer":         example["answer"],
                "level":          example.get("level", 0),
            })

        # Save checkpoint periodically
        if (start_idx + i + 1) % CKPT_EVERY == 0:
            with open(ckpt_file, "wb") as f:
                pickle.dump(all_outputs, f)

    # Final save
    with open(save_file, "wb") as f:
        pickle.dump(all_outputs, f)

    # Remove checkpoint once fully saved
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)

    print(f"\nSaved {len(all_outputs)} results to {save_file} | Errors: {errors}")
    return save_file


# ── Step 4: Conformal Prediction (LAC + APS) ───────────────────────────────────
def compute_corrected_quantile(scores, alpha):
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(scores, q_level, method="higher")


def compute_aps_scores(probs, labels):
    scores = []
    for i in range(len(probs)):
        p = probs[i]
        y = labels[i]
        sorted_indices = np.argsort(-p)
        sorted_probs   = p[sorted_indices]
        cumsum         = np.cumsum(sorted_probs)
        true_rank      = np.where(sorted_indices == y)[0][0]
        scores.append(cumsum[true_rank])
    return np.array(scores)


def get_aps_prediction_set(p, q_hat):
    sorted_indices = np.argsort(-p)
    sorted_probs   = p[sorted_indices]
    cumsum         = np.cumsum(sorted_probs)
    k = np.searchsorted(cumsum, q_hat, side="left")
    return sorted_indices[:k + 1]


def apply_conformal_prediction(logits_data, json_data, cal_ratio=0.5, alpha=0.1,
                                label="dataset"):
    """
    `logits_data` – list of dicts with logits_options / answer
    `json_data`   – same list (used for ground truth labels)
    """
    n         = len(logits_data)
    json_data = json_data[:n]

    label_map  = {l: i for i, l in enumerate(LABELS)}
    all_probs  = np.array([softmax(x["logits_options"]) for x in logits_data])
    all_labels = np.array([label_map.get(d["answer"], 0) for d in json_data])

    n_cal      = int(n * cal_ratio)
    cal_probs,  cal_labels  = all_probs[:n_cal],  all_labels[:n_cal]
    test_probs, test_labels = all_probs[n_cal:],  all_labels[n_cal:]

    results = {}
    for method in ["LAC", "APS"]:
        if method == "LAC":
            cal_scores = 1 - cal_probs[np.arange(n_cal), cal_labels]
        else:
            cal_scores = compute_aps_scores(cal_probs, cal_labels)

        q_hat = compute_corrected_quantile(cal_scores, alpha)
        print(f"  {method} Threshold ({label}): {q_hat:.4f}")

        pred_sets = []
        for p in test_probs:
            if method == "LAC":
                pred_set = [i for i in range(6) if (1 - p[i]) <= q_hat]
            else:
                pred_set = list(get_aps_prediction_set(p, q_hat))
            pred_sets.append(pred_set)

        accuracy  = np.mean(np.argmax(test_probs, 1) == test_labels) * 100
        set_sizes = [len(s) for s in pred_sets]
        avg_ss    = np.mean(set_sizes)
        coverage  = np.mean([test_labels[i] in pred_sets[i]
                              for i in range(len(test_labels))]) * 100

        results[method] = {
            "accuracy":             round(accuracy, 2),
            "avg_set_size":         round(avg_ss, 3),
            "coverage_rate":        round(coverage, 2),
            "threshold":            round(float(q_hat), 4),
            "set_size_distribution": np.bincount(set_sizes, minlength=7).tolist(),
        }

    return results


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load AQuA dataset
    print(f"Loading AQuA ({MAX_SAMPLES} samples)...")
    aqua_data = convert_aqua_to_mcq(split="train", max_samples=MAX_SAMPLES)
    print(f"Loaded {len(aqua_data)} examples")

    # 2. Split by ambiguity level
    aqua_by_level = {}
    for ex in aqua_data:
        lvl = ex.get("level", 0)
        aqua_by_level.setdefault(lvl, []).append(ex)

    print("\nLevel breakdown:")
    for lvl, items in sorted(aqua_by_level.items()):
        print(f"  Level {lvl}: {len(items)} samples")

    # 3. Check for existing full logits pkl
    full_pkl_path = os.path.join(OUTPUT_DIR, f"qwen_aqua_{len(aqua_data)}.pkl")

    if os.path.exists(full_pkl_path):
        print(f"\nFound existing logits: {full_pkl_path}. Skipping full inference.")
        with open(full_pkl_path, "rb") as f:
            all_outputs_full = pickle.load(f)
    else:
        # Load model once
        print("\nLoading Qwen2-VL-2B-Instruct...")
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        option_ids = [
            processor.tokenizer.encode(" " + o, add_special_tokens=False)[-1]
            for o in OPTIONS
        ]
        print(f"Option token IDs: {dict(zip(OPTIONS, option_ids))}")

        # Run inference on full dataset (with checkpointing)
        full_pkl_path = get_qwen_logits(
            aqua_data, OUTPUT_DIR, processor, model, option_ids,
            tag=f"aqua"
        )
        with open(full_pkl_path, "rb") as f:
            all_outputs_full = pickle.load(f)

    # 4. Group logits by level for per-level conformal prediction
    # Build a lookup from id -> output record
    id_to_output = {rec["id"]: rec for rec in all_outputs_full}

    level_results = {}
    all_results   = {}

    for lvl, data_lvl in sorted(aqua_by_level.items()):
        if len(data_lvl) < 50:
            print(f"\nSkipping Level {lvl} (only {len(data_lvl)} samples, need >=50)")
            continue

        print(f"\n{'='*70}")
        print(f"Running Conformal Prediction — Level {lvl} ({len(data_lvl)} samples)")
        print(f"{'='*70}")

        # Gather logits records for this level
        logits_lvl = [id_to_output[ex["id"]] for ex in data_lvl
                      if ex["id"] in id_to_output]

        if len(logits_lvl) < 50:
            print(f"  Not enough logits records for Level {lvl}, skipping.")
            continue

        res = apply_conformal_prediction(
            logits_lvl, data_lvl, cal_ratio=CAL_RATIO, alpha=ALPHA,
            label=f"Level {lvl}"
        )

        level_results[str(lvl)] = res

        for method in ["LAC", "APS"]:
            print(f"\n  {'='*60}")
            print(f"  METHOD: {method} | Alpha={ALPHA} | Level={lvl}")
            print(f"  {'='*60}")
            print(f"  Accuracy (%):      {res[method]['accuracy']:.2f}")
            print(f"  Avg Set Size:      {res[method]['avg_set_size']:.3f}")
            print(f"  Coverage Rate (%): {res[method]['coverage_rate']:.2f}")
            print(f"  Threshold:         {res[method]['threshold']:.4f}")
            print(f"  Set Size Dist:     {res[method]['set_size_distribution']}")

    # 5. Also run conformal on the FULL dataset
    print(f"\n{'='*70}")
    print(f"Running Conformal Prediction — ALL LEVELS COMBINED ({len(all_outputs_full)} samples)")
    all_results = apply_conformal_prediction(
        all_outputs_full, aqua_data, cal_ratio=CAL_RATIO, alpha=ALPHA,
        label="All Levels"
    )
    for method in ["LAC", "APS"]:
        print(f"\n  METHOD: {method}")
        print(f"  Accuracy (%):      {all_results[method]['accuracy']:.2f}")
        print(f"  Avg Set Size:      {all_results[method]['avg_set_size']:.3f}")
        print(f"  Coverage Rate (%): {all_results[method]['coverage_rate']:.2f}")
        print(f"  Threshold:         {all_results[method]['threshold']:.4f}")

    # 6. Save results
    output = {
        "model":       MODEL_NAME,
        "dataset":     "AQuA",
        "n_samples":   len(aqua_data),
        "alpha":       ALPHA,
        "cal_ratio":   CAL_RATIO,
        "all_levels":  all_results,
        "per_level":   level_results,
    }

    results_path = "results_qwen2vl_aqua_17k.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to {results_path}")
