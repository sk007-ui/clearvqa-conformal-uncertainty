"""
run_llava_aokvqa.py
--------------------
Runs llava-hf/llava-1.5-7b-hf (4-bit quantized) on exactly the 1000 samples
of the A-OKVQA dataset processed by run_aokvqa.py.
Tests Clear image vs. Ambiguous (Gaussian blur, radius=8) image on the SAME question.

Colab-ready:
  - Upload colab_aokvqa_images.zip and run:
      !unzip colab_aokvqa_images.zip -d images/
  - Upload outputs_aokvqa/aokvqa_final_logits.pkl
  - Then: python run_llava_aokvqa.py
"""

import os
import pickle
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME      = "llava-hf/llava-1.5-7b-hf"
LOCAL_IMAGE_DIR = "images/"          # Colab: extracted from colab_aokvqa_images.zip
OUTPUT_DIR      = "outputs_llava"

OPTIONS = ["A", "B", "C", "D"]

# ── Step 1: Build LLaVA MCQ Prompt ────────────────────────────────────────────
def build_prompt(question, choices):
    """
    Constructs the LLaVA USER/ASSISTANT chat template for a 4-choice MCQ.
    `choices` is a list of 4 answer strings.
    """
    choice_str = ""
    for i, choice in enumerate(choices):
        choice_str += f"{OPTIONS[i]}. {choice}\n"
    return (
        f"USER: <image>\n"
        f"Question: {question}\n"
        f"Choices:\n{choice_str}"
        f"Answer with only the letter (A, B, C, or D).\n"
        f"ASSISTANT:"
    )

# ── Step 2: Load exactly the same 1000 samples used in run_aokvqa.py ──────────
def load_target_data():
    """
    Reads aokvqa_final_logits.pkl and rebuilds a lightweight list of
    {question_id, image_file, question, choices, answer_idx, answer_char}
    by loading the A-OKVQA dataset and matching on question_id.
    """
    pkl_path = os.path.join("outputs_aokvqa", "aokvqa_final_logits.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Missing {pkl_path}. Upload it to Colab before running."
        )

    print(f"1. Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        prev_outputs = pickle.load(f)

    # Preserve the original order
    ordered_ids = [out["question_id"] for out in prev_outputs]
    id_to_meta  = {out["question_id"]: out for out in prev_outputs}
    target_id_set = set(ordered_ids)
    print(f"   Found {len(ordered_ids)} entries to replicate.")

    # The images are saved as {question_id}.jpg under LOCAL_IMAGE_DIR
    # Reconstruct dataset items from the pkl metadata + choices from HF dataset
    print("2. Loading A-OKVQA validation split to recover questions & choices...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    except Exception as e:
        raise RuntimeError(
            "Could not load HuggingFaceM4/A-OKVQA dataset. "
            "Ensure datasets is installed and HF is accessible."
        ) from e

    id_to_qa = {}
    for ex in tqdm(ds, desc="Scanning A-OKVQA"):
        qid = ex["question_id"]
        if qid in target_id_set:
            id_to_qa[qid] = {
                "question": ex["question"],
                "choices":  ex["choices"],
            }
            if len(id_to_qa) == len(target_id_set):
                break

    # Build final ordered list
    data = []
    for qid in ordered_ids:
        meta = id_to_meta[qid]
        qa   = id_to_qa.get(qid, {})
        data.append({
            "question_id": qid,
            "image_file":  f"{qid}.jpg",          # saved by zip_aokvqa_images.py
            "question":    qa.get("question", ""),
            "choices":     qa.get("choices", []),
            "answer_idx":  meta["answer_idx"],
            "answer_char": meta["answer_char"],
        })

    print(f"   Reconstructed {len(data)} samples.")
    return data


# ── Step 3: Inference Loop ─────────────────────────────────────────────────────
def run_inference():
    data = load_target_data()

    print("3. Loading LLaVA-1.5-7B in 4-bit (BitsAndBytes NF4)...")
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

    # Token IDs for the four option letters (leading space for generation context)
    option_ids = [
        processor.tokenizer.encode(" " + o, add_special_tokens=False)[-1]
        for o in OPTIONS
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_file  = os.path.join(OUTPUT_DIR, "llava_aokvqa_logits.ckpt.pkl")
    final_file = os.path.join(OUTPUT_DIR, "llava_aokvqa_final_logits.pkl")

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

    print("4. Starting inference  (Clear image vs. Ambiguous Blur r=8)...")
    for i, ex in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
        img_path = os.path.join(LOCAL_IMAGE_DIR, ex["image_file"])

        if not os.path.exists(img_path):
            missing_images += 1
            if missing_images <= 5:
                print(f"   WARNING: Missing image {img_path}")
            continue

        try:
            img_clear = Image.open(img_path).convert("RGB")

            # Standard anti-OOM resize: cap longest side at 1024 px
            max_size = 1024
            if max(img_clear.size) > max_size:
                ratio    = max_size / max(img_clear.size)
                new_size = (int(img_clear.size[0] * ratio), int(img_clear.size[1] * ratio))
                img_clear = img_clear.resize(new_size, Image.Resampling.LANCZOS)

            # Ambiguous version: Gaussian blur radius=8 (image-level ambiguity)
            img_ambig = img_clear.filter(ImageFilter.GaussianBlur(radius=8))

            prompt_text = build_prompt(ex["question"], ex["choices"])

            def get_logits(input_image):
                inputs = processor(
                    text=prompt_text,
                    images=input_image,
                    return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Logit scores at last token position for the 4 option letters
                return outputs.logits[:, -1, :].squeeze(0)[option_ids].float().cpu().numpy()

            logits_clear = get_logits(img_clear)
            logits_ambig = get_logits(img_ambig)

            all_outputs.append({
                "question_id":  ex["question_id"],
                "logits_clear": logits_clear,
                "logits_ambig": logits_ambig,
                "answer_idx":   ex["answer_idx"],
                "answer_char":  ex["answer_char"],
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

    # Final save
    with open(final_file, "wb") as f:
        pickle.dump(all_outputs, f)

    print(f"\nFinished!")
    print(f"  Missing images : {missing_images}")
    print(f"  Errors caught  : {errors}")
    print(f"  Saved {len(all_outputs)} result pairs -> {final_file}")


if __name__ == "__main__":
    run_inference()
