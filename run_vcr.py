import os
import json
import pickle
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ── Paths & Configuration ──────────────────────────────────────────────────────────
VCR_ANNOTS_DIR = r"C:\Users\sathw\Desktop\vcr1annots"
# IMPORTANT: Update this path to where your VCR images are actually stored.
VCR_IMAGES_DIR = r"C:\Users\sathw\Downloads\vcr1images\vcr1images"

OUTPUT_DIR = "outputs_vcr"
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
MAX_SAMPLES = 1000

OPTIONS = ["A", "B", "C", "D"]

# ── Step 1: Parsing & Text Translation ───────────────────────────────────────────
def detokenize(text_list, objects_list):
    """
    Translates VCR's mixed text/index lists into plain strings.
    E.g. ['What', 'are', [0,1], '?'] -> "What are person 1 and dog 2 ?"
    """
    words = []
    for item in text_list:
        if isinstance(item, list):
            # Translate each index into "ObjectName Index+1"
            translated = [f"{objects_list[idx]} {idx+1}" for idx in item]
            words.append(" and ".join(translated))
        else:
            words.append(str(item))
            
    # Join into a sentence and perform basic punctuation spacing cleanup
    sentence = " ".join(words)
    sentence = sentence.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    sentence = sentence.replace(" '", "'").replace(" n't", "n't")
    return sentence

def build_prompt(question_str, choices_str_list):
    """Constructs the standard MCQ format for the model."""
    prompt = f"Question: {question_str}\nChoices:\n"
    for i, choice in enumerate(choices_str_list):
        prompt += f"{OPTIONS[i]}. {choice}\n"
    prompt += "Answer with only the letter (A, B, C, or D)."
    return prompt

def load_vcr_data(filepath, max_lines):
    print(f"Loading top {max_lines} lines from {filepath}...")
    dataset = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            record = json.loads(line)
            
            # Detokenize question
            q_clean = detokenize(record["question"], record["objects"])
            
            # Detokenize answer choices
            choices_clean = [
                detokenize(choice, record["objects"]) 
                for choice in record["answer_choices"]
            ]
            
            # Form final prompt
            prompt = build_prompt(q_clean, choices_clean)
            answer_label_idx = int(record["answer_label"])
            
            dataset.append({
                "annot_id": record["annot_id"],
                "img_fn": record["img_fn"],
                "prompt": prompt,
                "answer_idx": answer_label_idx,
                "answer_char": OPTIONS[answer_label_idx]
            })
    return dataset

# ── Step 2: Inference Loop ────────────────────────────────────────────────────────
def run_inference():
    val_jsonl_path = os.path.join(VCR_ANNOTS_DIR, "val.jsonl")
    if not os.path.exists(val_jsonl_path):
        raise FileNotFoundError(f"Annotation file not found: {val_jsonl_path}")
        
    data = load_vcr_data(val_jsonl_path, MAX_SAMPLES)
    
    print("Loading Qwen2-VL-2B-Instruct (FP16)...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda", local_files_only=True
    )
    model.eval()

    # Get the strict generation IDs for options A, B, C, D
    # We add a leading space before encoding to match generic MCQ format likelihoods
    option_ids = [processor.tokenizer.encode(" " + o, add_special_tokens=False)[-1] for o in OPTIONS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_file = os.path.join(OUTPUT_DIR, "vcr_logits.ckpt.pkl")
    final_file = os.path.join(OUTPUT_DIR, "vcr_final_logits.pkl")
    
    if os.path.exists(ckpt_file):
        with open(ckpt_file, "rb") as f:
            all_outputs = pickle.load(f)
        start_idx = len(all_outputs)
        print(f"Resuming from sample {start_idx} (Found existing checkpoint)...")
    else:
        all_outputs = []
        start_idx = 0

    errors = 0
    missing_images = 0

    print("Starting Inference...")
    for i, ex in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
        img_path = os.path.join(VCR_IMAGES_DIR, ex["img_fn"])
        
        if not os.path.exists(img_path):
            missing_images += 1
            continue
            
        try:
            # 1. Load Clear Image
            img_clear = Image.open(img_path).convert("RGB")
            
            # Anti-OOM safety precaution: resize if image is massive
            max_size = 1024
            if max(img_clear.size) > max_size:
                ratio = max_size / max(img_clear.size)
                new_size = (int(img_clear.size[0] * ratio), int(img_clear.size[1] * ratio))
                img_clear = img_clear.resize(new_size, Image.Resampling.LANCZOS)
                
            # 2. Create Dynamically Blurred "Ambiguous" Image
            img_ambig = img_clear.filter(ImageFilter.GaussianBlur(radius=8))
            
            prompt_text = ex["prompt"]

            # Local helper to run inference on a specific image variant
            def get_logits(input_image):
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": input_image}, 
                        {"type": "text", "text": prompt_text}
                    ]}
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[input_image], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                # Sequence logic output over the 4 targeted option IDs
                return outputs.logits[:, -1, :].squeeze(0)[option_ids].float().cpu().numpy()

            # Execute forward passes
            logits_clear = get_logits(img_clear)
            logits_ambig = get_logits(img_ambig)

            # Store the data
            all_outputs.append({
                "annot_id": ex["annot_id"],
                "logits_clear": logits_clear,
                "logits_ambig": logits_ambig,
                "answer_idx": ex["answer_idx"],
                "answer_char": ex["answer_char"]
            })

        except Exception as e:
            if errors < 3:  # Only print first few to prevent spam
                import traceback
                traceback.print_exc()
            errors += 1

        # Checkpoint every 100 steps
        if (start_idx + i + 1) % 100 == 0:
            with open(ckpt_file, "wb") as f:
                pickle.dump(all_outputs, f)

    # Save final results mapping
    with open(final_file, "wb") as f:
        pickle.dump(all_outputs, f)

    print(f"\nProcessing Complete!")
    print(f"Missing Images: {missing_images}")
    print(f"Errors Caught: {errors}")
    print(f"Successfully saved {len(all_outputs)} result pairs to {final_file}.")

if __name__ == "__main__":
    run_inference()
