import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIGURATION =================
DATASET_BASE_PATH = Path(os.getenv("EXTRACTED_DATA_PATH"))

SPLITS = ['train', 'val', 'test']
FORCE_CLASS_ID = "0"
# =================================================

def unify_labels():
    """
    Goes through all label files and forces the class ID to 0.
    This fixes issues where automated trackers assigned different IDs to the same drone.
    """
    print(f"Starting label unification at: {DATASET_BASE_PATH}")
    total_updated = 0

    for split in SPLITS:
        label_dir = DATASET_BASE_PATH / split / "RAW_POSITIVES"
        
        if not label_dir.exists():
            print(f"Warning: Directory {label_dir} not found. Skipping {split}...")
            continue

        print(f"Processing split: {split}...")
        
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                continue

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    parts[0] = FORCE_CLASS_ID
                    updated_lines.append(" ".join(parts) + "\n")
            
            with open(label_file, 'w') as f:
                f.writelines(updated_lines)
            
            total_updated += 1

    print("\n" + "="*50)
    print("UNIFICATION COMPLETE")
    print(f"Total files updated to Class ID 0: {total_updated}")
    print("="*50)

if __name__ == "__main__":
    unify_labels()