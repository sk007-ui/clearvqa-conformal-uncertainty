import os
import json
import pickle
import shutil
import zipfile
from huggingface_hub import hf_hub_download

# --- Configuration ---
PKL_FILE = r"outputs_clearvqa\clearvqa_final_logits.pkl"
LOCAL_IMAGE_DIR = r"C:\Users\sathw\Desktop\images_clearvqa\images"
OUT_FOLDER = "colab_clearvqa_images"
ZIP_FILE = "colab_clearvqa_images.zip"

def main():
    print(f"1. Loading {PKL_FILE}...")
    with open(PKL_FILE, "rb") as f:
        outputs = pickle.load(f)
    
    # Extract the first 1000 targets
    target_outputs = outputs[:1000]
    target_ids = set([out["id"] for out in target_outputs])
    print(f"   Targeting {len(target_ids)} unique image IDs based on the logits file.")
    
    print("2. Mapping IDs to exact image filenames via ClearVQA dataset cache...")
    # Use local_files_only to bypass network issues as we downloaded this earlier
    jsonl_path = hf_hub_download(
        repo_id="jian0418/ClearVQA",
        filename="train_annotated.jsonl",
        repo_type="dataset",
        local_files_only=True
    )
    
    id_to_img = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                if ex["question_id"] in target_ids:
                    id_to_img[ex["question_id"]] = ex["image"]
                    
    print(f"   Mapped {len(id_to_img)}/{len(target_ids)} IDs to their image filenames.")
    
    print(f"3. Copying images directly to {OUT_FOLDER}...")
    os.makedirs(OUT_FOLDER, exist_ok=True)
    
    copied = 0
    missing = 0
    
    for q_id in target_ids:
        img_file = id_to_img.get(q_id)
        if not img_file:
            print(f"   Warning: No image mapping found for ID {q_id}")
            continue
            
        src = os.path.join(LOCAL_IMAGE_DIR, img_file)
        dst = os.path.join(OUT_FOLDER, img_file)
        
        # If the image filename contains subdirectories (e.g., 'val2014/xyz.jpg')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            if missing < 5:
                print(f"   Missing file: {src}")
                
    print(f"   Copied: {copied} | Missing: {missing}")
    
    print(f"4. Squeezing copies into {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUT_FOLDER):
            for file in files:
                filepath = os.path.join(root, file)
                # Keep directory structure relative to the OUT_FOLDER
                arcname = os.path.relpath(filepath, OUT_FOLDER)
                zipf.write(filepath, arcname)
                
    print(f"\nDone! Zip archive is fully built at {ZIP_FILE}")

if __name__ == "__main__":
    main()
