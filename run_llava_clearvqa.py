"""
run_llava_clearvqa.py
----------------------
Runs llava-hf/llava-1.5-7b-hf (4-bit quantized) on the first 1000 samples
of the ClearVQA dataset, testing Clear vs. Ambiguous (blurred question) prompts.
Matches the exact same samples used in the Qwen2-VL ClearVQA run.
"""

import os
import json
import pickle
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME      = "llava-hf/llava-1.5-7b-hf"
LOCAL_IMAGE_DIR = "images/"
OUTPUT_DIR      = "outputs_llava"
MAX_SAMPLES     = 1000

OPTIONS = ["A", "B", "C", "D", "E", "F"]

# ── Step 1: Load ClearVQA data (same logic as clearvqa_results.py) ─────────────
def convert_clearvqa(max_samples):
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

    records = records[:max_samples]

    result = []
    for idx, ex in enumerate(records):
        correct      = ex["gold_answer"]
        crowd_answers = ex["answers"] if isinstance(ex["answers"], list) else json.loads(ex["answers"])

        wrong = list(set([a for a in crowd_answers if a.lower() != correct.lower()]))
        fallback = ["I don't know", "Not visible", "Uncertain", "None of the above", "Other"]
        while len(wrong) < 5:
            wrong.append(fallback.pop(0))

        options = [correct] + wrong[:5]
        np.random.seed(idx)
        np.random.shuffle(options)

        choices = {OPTIONS[i]: options[i] for i in range(6)}
        correct_label = next(k for k, v in choices.items() if v == correct)

        result.append({
            "id":             ex["question_id"],
            "image_file":     ex["image"],
            "question_clear": ex["question"],
            "question_ambig": ex["blurred_question"],   # text-level ambiguity
            "choices":        choices,
            "answer":         correct_label,
        })
    return result

# ── Step 2: Build LLaVA-format prompt ─────────────────────────────────────────
def build_prompt(question, choices):
    choice_str = ""
    for k, v in choices.items():
        choice_str += f"{k}. {v}\n"
    # LLaVA requires the exact USER/ASSISTANT template
    return (
        f"USER: <image>\n"
        f"{question}\n"
        f"Choices:\n{choice_str}"
        f"Answer with only the letter (A, B, C, D, E, or F).\n"
        f"ASSISTANT:"
    )

# ── Step 3: Inference Loop ─────────────────────────────────────────────────────
def run_inference():
    print("1. Loading ClearVQA data (first 1000 samples)...")
    data = convert_clearvqa(MAX_SAMPLES)

    print("2. Loading LLaVA-1.5-7B in 4-bit (BitsAndBytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Token IDs for option letters (with leading space to match generation context)
    option_ids = [
        processor.tokenizer.encode(" " + o, add_special_tokens=False)[-1]
        for o in OPTIONS
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_file  = os.path.join(OUTPUT_DIR, "llava_clearvqa_logits.ckpt.pkl")
    final_file = os.path.join(OUTPUT_DIR, "llava_clearvqa_final_logits.pkl")

    if os.path.exists(ckpt_file):
        with open(ckpt_file, "rb") as f:
            all_outputs = pickle.load(f)
        start_idx = len(all_outputs)
        print(f"   Resuming from sample {start_idx} (checkpoint found)...")
    else:
        all_outputs = []
        start_idx = 0

    errors         = 0
    missing_images = 0

    print("3. Starting inference...")
    for i, ex in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
        img_path = os.path.join(LOCAL_IMAGE_DIR, ex["image_file"])

        if not os.path.exists(img_path):
            missing_images += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")

            # Standard anti-OOM resize: cap longest side at 1024px
            max_size = 1024
            if max(image.size) > max_size:
                ratio    = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image    = image.resize(new_size, Image.Resampling.LANCZOS)

            def get_logits(prompt_text):
                inputs = processor(
                    text=prompt_text,
                    images=image,
                    return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Logits at the last generated position for our 6 option tokens
                return outputs.logits[:, -1, :].squeeze(0)[option_ids].float().cpu().numpy()

            # Clear run — original question
            logits_clear = get_logits(build_prompt(ex["question_clear"], ex["choices"]))
            # Ambiguous run — blurred/rephrased question (ClearVQA text-level ambiguity)
            logits_ambig = get_logits(build_prompt(ex["question_ambig"], ex["choices"]))

            all_outputs.append({
                "id":           ex["id"],
                "logits_clear": logits_clear,
                "logits_ambig": logits_ambig,
                "answer":       ex["answer"],
            })

        except Exception as e:
            if errors < 3:
                import traceback
                traceback.print_exc()
            errors += 1

        # Checkpoint every 100 samples
        if (start_idx + i + 1) % 100 == 0:
            with open(ckpt_file, "wb") as f:
                pickle.dump(all_outputs, f)

    # Final save
    with open(final_file, "wb") as f:
        pickle.dump(all_outputs, f)

    print(f"\nFinished.  Missing images: {missing_images} | Errors: {errors}")
    print(f"Saved {len(all_outputs)} result pairs to {final_file}")

if __name__ == "__main__":
    run_inference()
