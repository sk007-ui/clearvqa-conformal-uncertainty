"""
run_clearvqa.py
---------------
Runs Qwen2-VL-2B-Instruct locally on ClearVQA.
Tests Clear vs. Ambiguous questions on the SAME image.
"""

import os
import json
import pickle
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.special import softmax
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR      = "outputs_clearvqa"
# UPDATE THIS PATH TO YOUR UNZIPPED IMAGES FOLDER IN ANTIGRAVITY
LOCAL_IMAGE_DIR = r"C:\Users\sathw\Desktop\images\images"
MAX_SAMPLES     = 17332
CAL_RATIO       = 0.5
ALPHA           = 0.1

OPTIONS = ["A", "B", "C", "D", "E", "F"]

# ── Step 1: Load Dataset & Build MCQs ──────────────────────────────────────────
def convert_clearvqa(max_samples=None):
    # Download raw JSONL directly to bypass schema mismatch in the dataset card
    jsonl_path = hf_hub_download(
        repo_id="jian0418/ClearVQA",
        filename="train_annotated.jsonl",
        repo_type="dataset",
    )
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    result = []
    for idx, ex in enumerate(records):
        correct = ex["gold_answer"]
        crowd_answers = ex["answers"] if isinstance(ex["answers"], list) else json.loads(ex["answers"])
        
        # Build "Strong" Distractors from crowd answers
        wrong = list(set([ans for ans in crowd_answers if ans.lower() != correct.lower()]))
        
        # Pad if we don't have enough distractors
        fallback = ["I don't know", "Not visible", "Uncertain", "None of the above", "Other"]
        while len(wrong) < 5:
            wrong.append(fallback.pop(0))
            
        options = [correct] + wrong[:5]
        np.random.seed(idx)
        np.random.shuffle(options)
        
        choices = {OPTIONS[i]: options[i] for i in range(6)}
        correct_label = next(k for k, v in choices.items() if v == correct)

        result.append({
            "id": ex["question_id"],
            "image_file": ex["image"],
            "question_clear": ex["question"],
            "question_ambig": ex["blurred_question"],
            "choices": choices,
            "answer": correct_label
        })
    return result

def build_prompt(question, choices):
    prompt = f"Question: {question}\nChoices:\n"
    for k, v in choices.items():
        prompt += f"{k}. {v}\n"
    prompt += "Answer with only the letter (A, B, C, D, E, or F)."
    return prompt

# ── Step 2: Inference Loop ──────────────────────────────────────────────────────
def run_inference(data, processor, model, option_ids):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_file = os.path.join(OUTPUT_DIR, "clearvqa_logits.ckpt.pkl")
    
    if os.path.exists(ckpt_file):
        with open(ckpt_file, "rb") as f:
            all_outputs = pickle.load(f)
        start_idx = len(all_outputs)
        print(f"Resuming from sample {start_idx}...")
    else:
        all_outputs = []
        start_idx = 0

    errors = 0
    missing_images = 0

    for i, ex in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
        img_path = os.path.join(LOCAL_IMAGE_DIR, ex["image_file"])
        
        # EXPLICIT IMAGE VERIFICATION (For Prof. Babu)
        if not os.path.exists(img_path):
            missing_images += 1
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            
            # --- START FIX: Resize giant images to prevent Out-Of-Memory ---
            # Max width/height of 1024 to keep VRAM usage safe
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            # --- END FIX ---
            
            # Helper to run model on a specific prompt
            def get_logits(prompt_text):
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                return outputs.logits[:, -1, :].squeeze(0)[option_ids].float().cpu().numpy()

            # Run CLEAR
            logits_clear = get_logits(build_prompt(ex["question_clear"], ex["choices"]))
            # Run AMBIGUOUS
            logits_ambig = get_logits(build_prompt(ex["question_ambig"], ex["choices"]))

            all_outputs.append({
                "id": ex["id"],
                "logits_clear": logits_clear,
                "logits_ambig": logits_ambig,
                "answer": ex["answer"]
            })

        except Exception as e:
            if errors < 3:
                import traceback
                traceback.print_exc()
            errors += 1

        if (start_idx + i + 1) % 100 == 0:
            with open(ckpt_file, "wb") as f:
                pickle.dump(all_outputs, f)

    print(f"Finished. Missing Images: {missing_images}, Errors: {errors}")
    return all_outputs

# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("1. Loading ClearVQA and generating strong distractors...")
    data = convert_clearvqa(MAX_SAMPLES)
    
    print("2. Loading Model (FP16 for RTX 5060)...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    option_ids = [processor.tokenizer.encode(" " + o, add_special_tokens=False)[-1] for o in OPTIONS]

    print("3. Starting Inference...")
    outputs = run_inference(data, processor, model, option_ids)
    
    # Save final results
    final_file = os.path.join(OUTPUT_DIR, "clearvqa_final_logits.pkl")
    with open(final_file, "wb") as f:
        pickle.dump(outputs, f)
    
    print(f"Successfully saved {len(outputs)} results to {final_file}")
    print("\nNext step: Run conformal prediction math on these logits!")