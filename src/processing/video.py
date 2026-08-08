import cv2
from PIL import Image
import imagehash
from pathlib import Path

def extract_keyframes_from_video(video_path: str, min_frames: int = 3, max_frames: int = 15) -> list[Image.Image]:
    """Extracts high-quality keyframes based on blur score and temporal distribution."""
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
    candidate_indices = [
        start_frame + int(usable_range * i / num_candidates) 
        for i in range(1, num_candidates + 1)
    ]
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


def get_frame_count(filepath: str) -> int:
    """Quickly gets the frame count of a video file without reading pixels."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def get_video_metadata(filepath: str) -> tuple[int, int, float, int]:
    """Gets video width, height, FPS, and total frame count."""
    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, length


def extract_video_hashes(video_path: Path) -> list[imagehash.ImageHash]:
    """Extracts 1 frame per second and returns a list of their perceptual hashes (pHash)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    hashes = []
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_id % fps == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            hashes.append(imagehash.phash(pil_image))
            
        frame_id += 1
        
    cap.release()
    return hashes