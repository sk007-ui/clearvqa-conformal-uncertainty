import os
import json
import pickle
import shutil
import zipfile

# --- Configuration ---
PKL_FILE = r"outputs_vcr\vcr_final_logits.pkl"
VCR_ANNOTS_DIR = r"C:\Users\sathw\Desktop\vcr1annots"
VCR_IMAGES_DIR = r"C:\Users\sathw\Downloads\vcr1images\vcr1images"
OUT_FOLDER = "colab_vcr_images"
ZIP_FILE = "colab_vcr_images.zip"

def main():
    print(f"1. Loading {PKL_FILE}...")
    with open(PKL_FILE, "rb") as f:
        outputs = pickle.load(f)
    
    target_ids = set([out["annot_id"] for out in outputs])
    print(f"   Targeting {len(target_ids)} unique image IDs.")
    
    print("2. Mapping IDs to exact image filenames via val.jsonl...")
    val_jsonl = os.path.join(VCR_ANNOTS_DIR, "val.jsonl")
    
    id_to_img = {}
    with open(val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                if ex["annot_id"] in target_ids:
                    id_to_img[ex["annot_id"]] = ex["img_fn"]
                    
    print(f"   Mapped {len(id_to_img)}/{len(target_ids)} IDs to filenames.")
    
    print(f"3. Copying images directly to {OUT_FOLDER}...")
    os.makedirs(OUT_FOLDER, exist_ok=True)
    
    copied = 0
    missing = 0
    
    for q_id in target_ids:
        img_fn = id_to_img.get(q_id)
        if not img_fn:
            continue
            
        src = os.path.join(VCR_IMAGES_DIR, img_fn)
        dst = os.path.join(OUT_FOLDER, img_fn)
        
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            if missing <= 5:
                print(f"   Missing file: {src}")
                
    print(f"   Copied: {copied} | Missing: {missing}")
    
    print(f"4. Squeezing copies into {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUT_FOLDER):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, OUT_FOLDER)
                zipf.write(filepath, arcname)
                
    print(f"\nDone! Zip archive is fully built at {ZIP_FILE}")

if __name__ == "__main__":
    main()
