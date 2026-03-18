import json
import os
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.special import softmax
from collections import Counter
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
import zipfile
from huggingface_hub import hf_hub_download

# ── Configuration ──────────────────────────────────────────
MODEL_NAME    = "llava-hf/llava-1.5-7b-hf"
IMAGE_FOLDER  = "clearvqa_images/images"
OUTPUT_DIR    = "outputs_llava"
MAX_SAMPLES   = None   # set to 500 for quick test, None for full run
CAL_RATIO     = 0.5
ALPHA         = 0.1
OPTIONS       = ["Answer: A", "Answer: B", "Answer: C",
                 "Answer: D", "Answer: E", "Answer: F"]
LABELS        = ["A", "B", "C", "D", "E", "F"]

# ── Step 1: Download images ─────────────────────────────────
def download_images():
    if os.path.exists(IMAGE_FOLDER):
        print(f"Images already extracted: {len(os.listdir(IMAGE_FOLDER))} files")
        return
    print("Downloading images.zip...")
    zip_path = hf_hub_download(
        repo_id="jian0418/ClearVQA",
        filename="images.zip",
        repo_type="dataset"
    )
    print("Extracting images...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("clearvqa_images")
    print(f"Extracted {len(os.listdir(IMAGE_FOLDER))} images")

# ── Step 2: Load and convert ClearVQA dataset ──────────────
def convert_to_mcq(df, question_col):
    result = []
    for idx, row in df.iterrows():
        answer_counts = Counter(row["answers"])
        top_answers = [a for a, _ in answer_counts.most_common(4)]
        if row["gold_answer"] not in top_answers:
            top_answers = top_answers[:3] + [row["gold_answer"]]
        while len(top_answers) < 4:
            top_answers.append("other")
        choices = {l: top_answers[i] for i, l in enumerate(["A","B","C","D"])}
        choices["E"] = "I don\'t know"
        choices["F"] = "None of the above"
        correct = next((l for l, a in choices.items()
                        if a == row["gold_answer"]), None)
        result.append({
            "source": "ClearVQA",
            "task": "QA",
            "question_id": row["question_id"],
            "question": row[question_col],
            "choices": choices,
            "answer": correct,
            "id": idx,
            "image": row["image"]
        })
    return result

def prepare_data():
    print("Loading ClearVQA dataset...")
    ds = load_dataset(
        "jian0418/ClearVQA",
        data_files={"train": "train_annotated.jsonl"},
        split="train"
    )
    df = ds.to_pandas()
    clear_mcq     = convert_to_mcq(df, "question")
    ambiguous_mcq = convert_to_mcq(df, "blurred_question")
    with open("clearvqa_clear.json", "w") as f:
        json.dump(clear_mcq, f, indent=2)
    with open("clearvqa_ambiguous.json", "w") as f:
        json.dump(ambiguous_mcq, f, indent=2)
    print(f"Saved {len(clear_mcq)} clear and {len(ambiguous_mcq)} ambiguous examples")

# ── Step 3: Generate LLaVA logits ──────────────────────────
def build_prompt(example):
    prompt = "USER: <image>\nQuestion: " + example["question"] + "\nChoices:\n"
    for k, v in example["choices"].items():
        prompt += k + ". " + str(v) + "\n"
    prompt += "Answer with only the letter.\nASSISTANT:"
    return prompt

def get_llava_logits(data_file, output_dir, processor, model,
                     option_ids, image_folder, max_samples=None):
    with open(data_file) as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    print(f"Running on {len(data)} examples")
    all_outputs = []
    errors = 0
    for example in tqdm(data):
        try:
            image = Image.open(
                os.path.join(image_folder, example["image"])
            ).convert("RGB")
            prompt = build_prompt(example)
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze(0)
            logits_options = logits[option_ids].float().cpu().numpy()
            all_outputs.append({
                "id": example["id"],
                "logits_options": logits_options,
                "answer": example["answer"]
            })
        except Exception as e:
            errors += 1
            all_outputs.append({
                "id": example["id"],
                "logits_options": np.zeros(6),
                "answer": example["answer"]
            })
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.basename(data_file).replace(".json", "")
    save_file = os.path.join(output_dir, f"llava_{fname}.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(all_outputs, f)
    print(f"Saved to {save_file} | Errors: {errors}")
    return save_file

# ── Step 4: Conformal Prediction ───────────────────────────
def apply_conformal_prediction(logits_file, json_file,
                                cal_ratio=0.5, alpha=0.1):
    with open(logits_file, "rb") as f:
        logits_data = pickle.load(f)
    with open(json_file) as f:
        json_data = json.load(f)
    n = len(logits_data)
    json_data = json_data[:n]
    label_map = {l: i for i, l in enumerate(LABELS)}
    all_probs  = np.array([softmax(x["logits_options"]) for x in logits_data])
    all_labels = np.array([
        label_map.get(d["answer"], 0) for d in json_data
    ])
    n_cal      = int(n * cal_ratio)
    cal_probs, cal_labels   = all_probs[:n_cal],  all_labels[:n_cal]
    test_probs, test_labels = all_probs[n_cal:],  all_labels[n_cal:]
    cal_scores  = 1 - cal_probs[np.arange(n_cal), cal_labels]
    threshold   = np.quantile(cal_scores, 1 - alpha)
    pred_sets   = [[i for i in range(6) if (1-p[i]) <= threshold]
                   for p in test_probs]
    accuracy    = np.mean(np.argmax(test_probs,1) == test_labels) * 100
    set_sizes   = [len(s) for s in pred_sets]
    avg_ss      = np.mean(set_sizes)
    coverage    = np.mean([test_labels[i] in pred_sets[i]
                           for i in range(len(test_labels))]) * 100
    return {
        "accuracy":             round(accuracy, 2),
        "avg_set_size":         round(avg_ss, 3),
        "coverage_rate":        round(coverage, 2),
        "set_size_distribution": np.bincount(set_sizes, minlength=7).tolist()
    }

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Images
    download_images()

    # 2. Data
    prepare_data()

    # 3. Model
    print("Loading LLaVA...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    option_ids = [processor.tokenizer.encode(o)[-1] for o in OPTIONS]
    print("LLaVA ready.")

    # 4. Logits
    clear_pkl = get_llava_logits(
        "clearvqa_clear.json", OUTPUT_DIR,
        processor, model, option_ids, IMAGE_FOLDER, MAX_SAMPLES)
    ambiguous_pkl = get_llava_logits(
        "clearvqa_ambiguous.json", OUTPUT_DIR,
        processor, model, option_ids, IMAGE_FOLDER, MAX_SAMPLES)

    # 5. Conformal prediction
    cr = apply_conformal_prediction(clear_pkl,    "clearvqa_clear.json",    CAL_RATIO, ALPHA)
    ar = apply_conformal_prediction(ambiguous_pkl,"clearvqa_ambiguous.json", CAL_RATIO, ALPHA)

    # 6. Print results
    print("\n" + "="*55)
    print("RESULTS: LLaVA-1.5-7B on ClearVQA")
    print("="*55)
    print(f"{'Metric':<25} {'Clear':>10} {'Ambiguous':>12}")
    print("-"*55)
    print(f"{'Accuracy (%)':<25} {cr['accuracy']:>10.2f} {ar['accuracy']:>12.2f}")
    print(f"{'Avg Set Size (SS)':<25} {cr['avg_set_size']:>10.2f} {ar['avg_set_size']:>12.2f}")
    print(f"{'Coverage Rate (%)':<25} {cr['coverage_rate']:>10.2f} {ar['coverage_rate']:>12.2f}")
    print("\nSet Size Distribution:")
    print(f"  Clear:     {cr['set_size_distribution']}")
    print(f"  Ambiguous: {ar['set_size_distribution']}")
    ss_diff  = ar['avg_set_size'] - cr['avg_set_size']
    acc_diff = cr['accuracy']     - ar['accuracy']
    print("\nKEY FINDING:")
    print(f"  Set size increase : +{ss_diff:.3f}")
    print(f"  Accuracy drop     : -{acc_diff:.2f}%")
    print("="*55)

    # 7. Save results
    results = {"model": MODEL_NAME, "dataset": "ClearVQA",
               "alpha": ALPHA, "cal_ratio": CAL_RATIO,
               "clear": cr, "ambiguous": ar,
               "finding": {"ss_increase": ss_diff, "acc_drop": acc_diff}}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results.json")
