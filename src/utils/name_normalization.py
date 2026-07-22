import os
import shutil
from pathlib import Path

def rename_dataset_videos_safely(source_dir, dest_dir):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Створюємо нову папку для фінальних файлів, щоб не було конфліктів
    dest_path.mkdir(parents=True, exist_ok=True)
    
    global_counter = 1
    video_extensions = ['.mp4', '.mov', '.m4v']
    
    # Збираємо всі відео і жорстко сортуємо
    videos = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
    videos.sort()
    
    print(f"Знайдено файлів для нумерації: {len(videos)}\n" + "="*40)
    
    for filepath in videos:
        old_name = filepath.stem
        parts = old_name.split('_')
        
        if len(parts) >= 3:
            # Витягуємо клас (перші три слова)
            prefix = f"{parts[0]}_{parts[1]}_{parts[2]}"
            
            # Генеруємо нове ім'я з наскрізним номером
            new_filename = f"{prefix}_{global_counter:03d}{filepath.suffix}"
            new_filepath = dest_path / new_filename
            
            # КОПІЮЄМО файл у нову папку (найбезпечніший метод)
            shutil.copy2(filepath, new_filepath)
            print(f"✅ [{global_counter}/{len(videos)}] {filepath.name} -> {new_filename}")
            
            global_counter += 1
        else:
            print(f"⏭️ Пропущено (нестандартне ім'я): {filepath.name}")

if __name__ == "__main__":
    # Твоя папка з перевіреними даними
    source = r"E:\bachelor_data\clean_classified"
    
    # НОВА папка, куди скрипт складе ідеально пронумерований датасет
    destination = r"E:\bachelor_data\dataset" 
    
    rename_dataset_videos_safely(source, destination)