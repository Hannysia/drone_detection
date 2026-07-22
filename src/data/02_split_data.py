import os
import shutil
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_DIR = r"E:\bachelor_data\dataset"
OUTPUT_DIR = r"E:\bachelor_data\dataset_split"  
HF_REPO_ID = "hannusia123123/drone_dataset"
VIDEO_EXT = ('.mp4', '.mov', '.m4v')
# =================================================

def parse_filename(filename):
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return None

def get_frame_count(filepath):
    """Швидке отримання кількості кадрів без читання пікселів"""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def main():
    print(f"[*] Scanning source directory: {SOURCE_DIR}")
    data = []
    
    # Додано tqdm для відображення прогресу зчитування метаданих
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(VIDEO_EXT)]
    for filename in tqdm(files, desc="Reading video metadata"):
        combined_class = parse_filename(filename)
        if combined_class:
            filepath = os.path.join(SOURCE_DIR, filename)
            frames = get_frame_count(filepath)
            data.append({'filename': filename, 'combined_class': combined_class, 'frames': frames})
                
    df = pd.DataFrame(data)
    if df.empty:
        print("[!] Error: No valid video files found.")
        return

    # --- КАТЕГОРИЗАЦІЯ ЗА ТРИВАЛІСТЮ (БІННІНГ) ---
    # pd.qcut автоматично розіб'є всі відео на 3 рівні групи за кількістю кадрів
    df['duration_bin'] = pd.qcut(df['frames'], q=3, labels=['short', 'med', 'long'], duplicates='drop')

    # Створюємо комбінований клас для стратифікації: клас_камера_фон_тривалість
    df['stratify_target'] = df['combined_class'] + "_" + df['duration_bin'].astype(str)

    # --- СТРАТИФІКАЦІЯ (70 / 15 / 15) З БЕЗПЕЧНИМ ВІДКОТОМ ---
    # Якщо деталізований клас (напр., wing_rgb_sky_long) має менше 6 відео, 
    # ми повертаємо його до базового класу (wing_rgb_sky), щоб не зламати спліт
    strat_counts = df['stratify_target'].value_counts()
    safe_strata = strat_counts[strat_counts >= 6].index

    df['final_strat'] = df['stratify_target']
    df.loc[~df['final_strat'].isin(safe_strata), 'final_strat'] = df['combined_class']

    # Фінальна перевірка на рідкісні класи (з твоєї оригінальної логіки)
    final_counts = df['final_strat'].value_counts()
    safe_final_classes = final_counts[final_counts >= 6].index

    df_safe = df[df['final_strat'].isin(safe_final_classes)]
    df_rare = df[~df['final_strat'].isin(safe_final_classes)]
    
    train_safe, temp_safe = train_test_split(df_safe, test_size=0.30, stratify=df_safe['final_strat'], random_state=42)
    val_df, test_df = train_test_split(temp_safe, test_size=0.50, stratify=temp_safe['final_strat'], random_state=42)
    
    train_df = pd.concat([train_safe, df_rare], ignore_index=True)

    # --- ТЕЛЕМЕТРІЯ РОЗПОДІЛУ КАДРІВ ---
    total_frames = df['frames'].sum()
    print("\n📊 РОЗПОДІЛ СИРИХ КАДРІВ ПО СПЛІТАХ:")
    print(f"Train: {train_df['frames'].sum()} кадрів ({train_df['frames'].sum()/total_frames*100:.1f}%) | {len(train_df)} відео")
    print(f"Val:   {val_df['frames'].sum()} кадрів ({val_df['frames'].sum()/total_frames*100:.1f}%) | {len(val_df)} відео")
    print(f"Test:  {test_df['frames'].sum()} кадрів ({test_df['frames'].sum()/total_frames*100:.1f}%) | {len(test_df)} відео")

    # --- ФІЗИЧНЕ КОПІЮВАННЯ ЛОКАЛЬНО ---
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    print(f"\n[*] Preparing local directory structure in: {OUTPUT_DIR}")
    
    for split_name, split_data in splits.items():
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for _, row in tqdm(split_data.iterrows(), total=len(split_data), desc=f"Copying {split_name.upper()}"):
            src = os.path.join(SOURCE_DIR, row['filename'])
            dst = os.path.join(split_dir, row['filename'])
            shutil.copy2(src, dst)

    # --- ЗАВАНТАЖЕННЯ НА HUGGING FACE ---
    print("\n🚀 Uploading the entire dataset to Hugging Face...")
    api = HfApi()
    
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message="Initial commit: Stratified 70/15/15 split (Duration + Morphology)"
    )

    print("\n✅ Upload complete! Hugging Face now holds all three splits in one dataset.")

if __name__ == "__main__":
    main()