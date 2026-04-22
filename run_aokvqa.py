"""
run_aokvqa.py
--------------
Runs Qwen2-VL-2B-Instruct on the A-OKVQA validation split.
Tests Clear vs. Ambiguous (blurred) images on the SAME question.
"""

import os
import pickle
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR  = "outputs_aokvqa"
MAX_SAMPLES = 1000

OPTIONS = ["A", "B", "C", "D"]

# ── Step 1: Build MCQ Prompt ───────────────────────────────────────────────────
def build_prompt(question, choices):
    prompt = f"Question: {question}\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{OPTIONS[i]}. {choice}\n"
    prompt += "Answer with only the letter (A, B, C, or D)."
    return prompt

# ── Step 2: Inference Loop ─────────────────────────────────────────────────────
def run_inference():
    # Load dataset — streaming=False to allow slicing
    print("1. Loading A-OKVQA validation split (first 1000 samples)...")
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    print(f"   Loaded {len(ds)} samples.")

    print("2. Loading Qwen2-VL-2B-Instruct (FP16)...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda", local_files_only=True
    )
    model.eval()

    # Token IDs for the four option letters
    option_ids = [
        processor.tokenizer.encode(" " + o, add_special_tokens=False)[-1]
        for o in OPTIONS
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_file  = os.path.join(OUTPUT_DIR, "aokvqa_logits.ckpt.pkl")
    final_file = os.path.join(OUTPUT_DIR, "aokvqa_final_logits.pkl")

    # Resume from checkpoint if one exists
    if os.path.exists(ckpt_file):
        with open(ckpt_file, "rb") as f:
            all_outputs = pickle.load(f)
        start_idx = len(all_outputs)
        print(f"   Resuming from sample {start_idx} (checkpoint found)...")
    else:
        all_outputs = []
        start_idx = 0

    errors = 0

    print("3. Starting inference...")
    for i, ex in enumerate(tqdm(ds.select(range(start_idx, len(ds))),
                                 initial=start_idx, total=len(ds))):

        try:
            # Image comes directly as a PIL Image from the dataset column
            img_clear = ex["image"].convert("RGB")

            # Standard anti-OOM resize: cap longest side at 1024px
            max_size = 1024
            if max(img_clear.size) > max_size:
                ratio    = max_size / max(img_clear.size)
                new_size = (int(img_clear.size[0] * ratio), int(img_clear.size[1] * ratio))
                img_clear = img_clear.resize(new_size, Image.Resampling.LANCZOS)

            # Ambiguous version: Gaussian blur radius=8
            img_ambig = img_clear.filter(ImageFilter.GaussianBlur(radius=8))

            prompt_text = build_prompt(ex["question"], ex["choices"])

            def get_logits(input_image):
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": input_image},
                        {"type": "text",  "text": prompt_text}
                    ]}
                ]
                text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[input_image], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                return outputs.logits[:, -1, :].squeeze(0)[option_ids].float().cpu().numpy()

            logits_clear = get_logits(img_clear)
            logits_ambig = get_logits(img_ambig)

            all_outputs.append({
                "question_id": ex.get("question_id", start_idx + i),
                "logits_clear": logits_clear,
                "logits_ambig": logits_ambig,
                "answer_idx":   int(ex["correct_choice_idx"]),
                "answer_char":  OPTIONS[int(ex["correct_choice_idx"])]
            })

        except Exception as e:
            if errors < 3:
                import traceback
                traceback.print_exc()
            errors += 1

        # Checkpoint every 100 processed samples
        if (start_idx + i + 1) % 100 == 0:
            with open(ckpt_file, "wb") as f:
                pickle.dump(all_outputs, f)

    # Save final output
    with open(final_file, "wb") as f:
        pickle.dump(all_outputs, f)

    print(f"\nFinished. Errors: {errors}")
    print(f"Successfully saved {len(all_outputs)} result pairs to {final_file}")

if __name__ == "__main__":
    run_inference()
