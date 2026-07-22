import os
import cv2
import imagehash
from PIL import Image
import itertools
import shutil
from pathlib import Path

# ================= НАЛАШТУВАННЯ =================
VIDEO_DIR = r"E:\bachelor_data\clean_classified"         # Папка, де лежать твої 610 відео
DUPLICATES_DIR = "suspected_duplicates"  # Куди перемістити знайдені дублікати
HASH_THRESHOLD = 5                       # Поріг Геммінга (0 - абсолютна копія, 5 - дуже схожі)
# ================================================

def extract_video_hashes(video_path):
    """Витягує 1 кадр на секунду і повертає список їхніх pHash."""
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    hashes = []
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Беремо 1 кадр на секунду
        if frame_id % fps == 0:
            # Конвертуємо OpenCV (BGR) у формат PIL (RGB) для imagehash
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Рахуємо перцептивний хеш (phash)
            hashes.append(imagehash.phash(pil_image))
            
        frame_id += 1
        
    cap.release()
    return hashes

def find_and_isolate_duplicates():
    video_dir = Path(VIDEO_DIR)
    dup_dir = Path(DUPLICATES_DIR)
    dup_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = [f for f in video_dir.iterdir() if f.suffix.lower() in ['.mp4', '.mov', '.m4v']]
    print(f"🔍 Аналізуємо {len(video_files)} відео. Обчислення хешів...")
    
    # Крок 1: Збираємо хеші для всіх відео
    video_signatures = {}
    for v_path in video_files:
        video_signatures[v_path.name] = extract_video_hashes(v_path)
        
    # Крок 2: Шукаємо перетини
    print("⚔️ Порівняння відео між собою...")
    duplicates_to_move = set()
    
    # itertools.combinations створює всі можливі унікальні пари (Відео А, Відео Б)
    for vid_a, vid_b in itertools.combinations(video_signatures.keys(), 2):
        # Якщо хоча б одне відео вже визнано дублікатом, пропускаємо
        if vid_a in duplicates_to_move or vid_b in duplicates_to_move:
            continue
            
        hashes_a = video_signatures[vid_a]
        hashes_b = video_signatures[vid_b]
        
        is_duplicate = False
        # Порівнюємо кожен збережений кадр Відео А з кожним кадром Відео Б
        for h_a in hashes_a:
            for h_b in hashes_b:
                if h_a - h_b <= HASH_THRESHOLD:
                    is_duplicate = True
                    break
            if is_duplicate:
                break
                
        if is_duplicate:
            print(f"⚠️ Знайдено дублікат: {vid_b} дуже схоже на {vid_a}")
            # Додаємо друге відео в список на виселення
            duplicates_to_move.add(vid_b)
            
    # Крок 3: Переміщення дублікатів
    print("-" * 30)
    print(f"📂 Знайдено потенційних дублікатів: {len(duplicates_to_move)}")
    for dup in duplicates_to_move:
        src = video_dir / dup
        dst = dup_dir / dup
        shutil.move(str(src), str(dst))
        
    print(f"✅ Всі знайдені дублікати переміщено у папку '{DUPLICATES_DIR}'")

if __name__ == "__main__":
    find_and_isolate_duplicates()