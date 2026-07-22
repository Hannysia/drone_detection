import os
import cv2
import json
import shutil
import time
from PIL import Image
from google import genai
from google.genai import types
from tqdm import tqdm

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

with open("key.json", "r") as f:
    key_data = json.load(f)
    PROJECT_ID = key_data["project_id"]

client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location="us-central1"
)

SOURCE_ROOT = r"E:\bachelor_data\episods"

DESTINATION_DIR = r"E:\bachelor_data\clean_classified"
SAVE_DEBUG_FRAMES = False

os.makedirs(DESTINATION_DIR, exist_ok=True)


def extract_keyframes_from_video(video_path, min_frames=3, max_frames=15):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0 or fps <= 0:
        cap.release()
        return []
    
    duration = total_frames / fps
    num_frames = max(min_frames, min(max_frames, int(duration)))

    if duration > 15:
        start_frame = int(total_frames * 0.70)
        end_frame = int(total_frames * 1.0)
    else:
        start_frame = int(total_frames * 0.10)
        end_frame = int(total_frames * 0.90)
        
    usable_range = end_frame - start_frame
    
    if usable_range < num_frames:
        start_frame = 0
        end_frame = total_frames - 1
        usable_range = end_frame - start_frame
    
    num_candidates = max(num_frames * 2, 12)
    candidate_indices = [start_frame + int(usable_range * i / num_candidates) for i in range(1, num_candidates + 1)]
    candidates = []
    
    for idx in candidate_indices:
        if idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            candidates.append((blur_score, frame, idx))
            
    cap.release()
    
    if not candidates:
        return []
        
    top_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:num_frames]
    chronological_candidates = sorted(top_candidates, key=lambda x: x[2])
    
    best_frames = []
    for candidate in chronological_candidates:
        frame = candidate[1]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        best_frames.append(Image.fromarray(frame_rgb))
        
    return best_frames


def classify_single_video(frames):
    prompt = """
        You are an expert in computer vision and military UAV reconnaissance. You have been provided with a chronological sequence of several frames from a SINGLE video recording.
        As this is a single video, the target (if present) is of the same type in all frames. 
        In some frames, the target may not yet be present or may be invisible. Analyse ALL frames, find the one where the physical object in the air is most clearly visible, and use this to draw your final conclusion.

        Return the response STRICTLY in JSON format with the following three fields (use lowercase only):

        1. "target_shape": the morphology (geometric shape) of a military UAV.
        CRITICAL RULES REGARDING ARTIFACTS:
        - Ignore watermarks and logos! There may be logos on the screen in the form of drone icons (e.g., yellow drawings). These are graphics, not targets!
        - Ignore system crosshairs! The cross-shaped marker (crosshairs) in the centre of the screen is part of the interface. It is STRICTLY FORBIDDEN to confuse this cross with a drone of the "cross" shape.
        - The real target is a physical object in the sky.

        Select STRICTLY one option based on the silhouette SHAPE:
        - delta (delta-shaped silhouette, solid triangular wing, arrow-shaped form without a distinct tubular fuselage. This class includes Shahed-type kamikaze drones);
        - cross (cross-shaped silhouette, featuring characteristic X-shaped wings at the front or rear. This class includes the Lancet and similar models. Do not confuse with a sight!);
        - wing (classic straight wing, aeroplane-type airframe with a pronounced long wingspan and narrow fuselage. This class includes reconnaissance wings);
        - no_drone (if there is NO actual drone in the sky in ANY of the frames: only clouds, land, a crosshair or watermarks);

        2. "camera_type": camera type.
        Select ONE option:
        - rgb (daytime colour camera, with natural colours and shadows)
        - thermal (thermal imaging camera: monochrome image without shadows; background is a uniform plane, target is a contrasting silhouette)

        3. "background": the visual complexity of the background DIRECTLY BEHIND THE OBJECT.
        Assess the background only in the frame where the target is most clearly visible. Assess the background area ONLY behind the target’s silhouette.

        Select ONE option:
        - sky (the silhouette is entirely against a background of sky or clouds)
        - uniform (the silhouette is entirely against a uniform surface: water, sea, smooth snow)
        - field (the silhouette is entirely against a 2D geometric background: agricultural fields, roads)
        - forest (the silhouette is entirely against an organic texture: forest, trees)
        - urban (the silhouette is entirely against a hard geometric background: urban buildings, infrastructure)
        - horizon (the background behind the silhouette is mixed: the horizon line cuts through the background directly behind the object)

        Example response:
        {
            "target_shape": "cross",
            "camera_type": "rgb",
            "background": "field"
        }
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[prompt, *frames],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        tqdm.write(f"  [!] Помилка API Gemini: {e}")
        return None

processed_count = 0
skipped_count = 0
error_count = 0

existing_files = os.listdir(DESTINATION_DIR)

print(f"Starting to scan the directory: {SOURCE_ROOT}...")

all_videos = []
for root, dirs, files in os.walk(SOURCE_ROOT):
    for file in files:
        if file.lower().endswith(('.mp4', '.mov', '.m4v')):
            all_videos.append((root, file))

print(f"Videos found for processing: {len(all_videos)}")
print(f"Previously processed files found: {len(existing_files)}\n" + "-"*40)

for root, file in tqdm(all_videos, desc="Video classification", unit="file"):
    current_folder_name = os.path.basename(root)
    
    try:
        video_path = os.path.join(root, file)
        file_name_clean = os.path.splitext(file)[0]
        file_ext = os.path.splitext(file)[1].lower()
        
        expected_suffix = f"_{current_folder_name}_{file_name_clean}{file_ext}"
        already_processed = any(f.endswith(expected_suffix) for f in existing_files)
        
        if already_processed:
            skipped_count += 1
            continue
        
        frames = extract_keyframes_from_video(video_path)
        if not frames:
            tqdm.write(f" [!] Unable to extract frames: {file}")
            error_count += 1
            continue
            
        if SAVE_DEBUG_FRAMES:
            debug_base_dir = "debug_frames"
            video_debug_dir = os.path.join(debug_base_dir, file_name_clean)
            os.makedirs(video_debug_dir, exist_ok=True)
            for i, pil_img in enumerate(frames):
                pil_img.save(os.path.join(video_debug_dir, f"frame_{i:03d}.png"))
                
        classification = classify_single_video(frames)
        if not classification:
            tqdm.write(f" [!] Відповідь API порожня: {file}")
            error_count += 1
            continue
            
        target = classification.get("target_shape", "unknown")
        camera = classification.get("camera_type", "unknown")
        bg = classification.get("background", "unknown")
        
        new_file_name = f"{target}_{camera}_{bg}_{current_folder_name}_{file_name_clean}{file_ext}"
        new_file_path = os.path.join(DESTINATION_DIR, new_file_name)
        
        shutil.copy2(video_path, new_file_path)
        
        existing_files.append(new_file_name)
        processed_count += 1
        
        time.sleep(4.0)
        
    except Exception as e:
        tqdm.write(f" [CRITICAL ERROR] Error with file {file}: {e}")
        error_count += 1
        continue

print("\n" + "=" * 40)
print(f"[Done!] Processing complete.")
print(f"Successfully classified: {processed_count}")
print(f"Skipped (already present): {skipped_count}")
print(f"Errors/corrupted files: {error_count}")
print("="*40)