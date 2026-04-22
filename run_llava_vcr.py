"""
run_llava_vcr.py
----------------
Runs llava-hf/llava-1.5-7b-hf (4-bit quantized) on exactly the 1000 samples
of the VCR dataset processed earlier. Captures Clear vs. Ambiguous (blur) scores.
Colab-ready: VCR_IMAGES_DIR points to "images/" relative folder.
"""

import os
import json
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
MODEL_NAME     = "llava-hf/llava-1.5-7b-hf"
# Expected Colab path when extracted
VCR_IMAGES_DIR = "images/"
VCR_ANNOTS_DIR = "vcr1annots"  # Make sure you upload val.jsonl here!
OUTPUT_DIR     = "outputs_llava"

OPTIONS = ["A", "B", "C", "D"]

# ── Step 1: Parsing & Text Translation ──────────────────────────────────────────
def detokenize(text_list, objects_list):
    words = []
    for item in text_list:
        if isinstance(item, list):
            translated = [f"{objects_list[idx]} {idx+1}" for idx in item]
            words.append(" and ".join(translated))
        else:
            words.append(str(item))
            
    sentence = " ".join(words)
    sentence = sentence.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    sentence = sentence.replace(" '", "'").replace(" n't", "n't")
    return sentence

def build_llava_prompt(question_str, choices_str_list):
    """Constructs the LLaVA USER/ASSISTANT MCQ prompt"""
    choice_str = ""
    for i, choice in enumerate(choices_str_list):
        choice_str += f"{OPTIONS[i]}. {choice}\n"
        
    return (
        f"USER: <image>\n"
        f"Question: {question_str}\n"
        f"Choices:\n{choice_str}"
        f"Answer with only the letter (A, B, C, or D).\n"
        f"ASSISTANT:"
    )

def load_vcr_data(filepath, target_ids):
    dataset = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["annot_id"] not in target_ids:
                continue
                
            q_clean = detokenize(record["question"], record["objects"])
            choices_clean = [detokenize(c, record["objects"]) for c in record["answer_choices"]]
            
            prompt = build_llava_prompt(q_clean, choices_clean)
            answer_label_idx = int(record["answer_label"])
            
            dataset.append({
                "annot_id": record["annot_id"],
                "img_fn": record["img_fn"],
                "prompt": prompt,
                "answer_idx": answer_label_idx,
                "answer_char": OPTIONS[answer_label_idx]
            })
    return dataset

# ── Step 2: Inference Loop ─────────────────────────────────────────────────────
def run_inference():
    print("1. Extracting exactly matching IDs from CP evaluation...")
    pkl_file = os.path.join("outputs_vcr", "vcr_final_logits.pkl")
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Missing {pkl_file}. Please ensure it exists on Colab.")
        
    with open(pkl_file, "rb") as f:
        previous_outputs = pickle.load(f)
    target_ids = set([out["annot_id"] for out in previous_outputs])
    
    val_jsonl_path = os.path.join(VCR_ANNOTS_DIR, "val.jsonl")
    data = load_vcr_data(val_jsonl_path, target_ids)
    print(f"   Loaded {len(data)} matching VCR annotation samples.")

    print("2. Loading LLaVA-1.5-7B in 4-bit...")
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

    option_ids = [processor.tokenizer.encode(" " + o, add_special_tokens=False)[-1] for o in OPTIONS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_file = os.path.join(OUTPUT_DIR, "llava_vcr_logits.ckpt.pkl")
    final_file = os.path.join(OUTPUT_DIR, "llava_vcr_final_logits.pkl")
    
    if os.path.exists(ckpt_file):
        with open(ckpt_file, "rb") as f:
            all_outputs = pickle.load(f)
        start_idx = len(all_outputs)
        print(f"   Resuming from sample {start_idx} (checkpoint found)...")
    else:
        all_outputs = []
        start_idx = 0

    errors = 0
    missing_images = 0

    print("3. Starting Inference...")
    for i, ex in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
        img_path = os.path.join(VCR_IMAGES_DIR, ex["img_fn"])
        if not os.path.exists(img_path):
            missing_images += 1
            continue
            
        try:
            img_clear = Image.open(img_path).convert("RGB")
            
            max_size = 1024
            if max(img_clear.size) > max_size:
                ratio = max_size / max(img_clear.size)
                new_size = (int(img_clear.size[0] * ratio), int(img_clear.size[1] * ratio))
                img_clear = img_clear.resize(new_size, Image.Resampling.LANCZOS)
                
            img_ambig = img_clear.filter(ImageFilter.GaussianBlur(radius=8))

            def get_logits(input_image):
                inputs = processor(
                    text=ex["prompt"],
                    images=input_image,
                    return_tensors="pt"
                ).to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                return outputs.logits[:, -1, :].squeeze(0)[option_ids].float().cpu().numpy()

            logits_clear = get_logits(img_clear)
            logits_ambig = get_logits(img_ambig)

            all_outputs.append({
                "annot_id": ex["annot_id"],
                "logits_clear": logits_clear,
                "logits_ambig": logits_ambig,
                "answer_idx": ex["answer_idx"],
                "answer_char": ex["answer_char"]
            })

        except Exception as e:
            if errors < 3:
                import traceback
                traceback.print_exc()
            errors += 1

        if (start_idx + i + 1) % 100 == 0:
            with open(ckpt_file, "wb") as f:
                pickle.dump(all_outputs, f)

    with open(final_file, "wb") as f:
        pickle.dump(all_outputs, f)

    print(f"\nProcessing Complete!")
    print(f"Missing Images: {missing_images}")
    print(f"Errors Caught: {errors}")
    print(f"Saved {len(all_outputs)} result pairs to {final_file}.")

if __name__ == "__main__":
    run_inference()
