"""
zip_aokvqa_images.py
---------------------
1. Reads outputs_aokvqa/aokvqa_final_logits.pkl to get the 1000 question_ids.
2. Loads the HuggingFaceM4/A-OKVQA validation split.
3. For each matching sample, saves the embedded PIL image as {question_id}.jpg
   into colab_aokvqa_images/.
4. Compresses the folder into colab_aokvqa_images.zip.

Run this locally (not on Colab).
"""

import os
import pickle
import zipfile
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# ── Configuration ──────────────────────────────────────────────────────────────
PKL_FILE   = os.path.join("outputs_aokvqa", "aokvqa_final_logits.pkl")
OUT_FOLDER = "colab_aokvqa_images"
ZIP_FILE   = "colab_aokvqa_images.zip"

def main():
    # ── Step 1: Load pkl and extract the 1000 question_ids ────────────────────
    print(f"1. Loading {PKL_FILE}...")
    with open(PKL_FILE, "rb") as f:
        outputs = pickle.load(f)

    target_ids = [out["question_id"] for out in outputs]
    target_id_set = set(target_ids)
    print(f"   Found {len(target_ids)} entries  ({len(target_id_set)} unique question_ids).")

    # ── Step 2: Load dataset and extract images ────────────────────────────────
    print("2. Loading A-OKVQA validation split from HuggingFace...")
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    print(f"   Dataset has {len(ds)} total samples.")

    # Build a lookup: question_id -> PIL Image
    print("3. Matching dataset samples to target question_ids...")
    id_to_image = {}
    for ex in tqdm(ds, desc="Scanning dataset"):
        qid = ex["question_id"]
        if qid in target_id_set:
            id_to_image[qid] = ex["image"]
            if len(id_to_image) == len(target_id_set):
                break  # early-exit once all found

    print(f"   Matched {len(id_to_image)}/{len(target_id_set)} images.")

    # ── Step 3: Save images to out folder ─────────────────────────────────────
    print(f"4. Saving images to {OUT_FOLDER}/...")
    os.makedirs(OUT_FOLDER, exist_ok=True)

    saved   = 0
    missing = 0

    for qid in tqdm(target_ids, desc="Saving images"):
        img = id_to_image.get(qid)
        if img is None:
            missing += 1
            if missing <= 5:
                print(f"   WARNING: No image found for question_id={qid}")
            continue

        out_path = os.path.join(OUT_FOLDER, f"{qid}.jpg")
        if not os.path.exists(out_path):
            img_rgb = img.convert("RGB")
            img_rgb.save(out_path, "JPEG", quality=92)
        saved += 1

    print(f"   Saved: {saved} | Missing: {missing}")

    # ── Step 4: Zip the folder ─────────────────────────────────────────────────
    print(f"5. Compressing {OUT_FOLDER}/ -> {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fname in os.listdir(OUT_FOLDER):
            fpath = os.path.join(OUT_FOLDER, fname)
            zipf.write(fpath, fname)   # flat structure: images/ in Colab = unzipped folder

    zip_size_mb = os.path.getsize(ZIP_FILE) / (1024 ** 2)
    print(f"\nDone! {ZIP_FILE} written ({zip_size_mb:.1f} MB).")
    print(f"Upload it to Colab and run: !unzip colab_aokvqa_images.zip -d images/")

if __name__ == "__main__":
    main()
