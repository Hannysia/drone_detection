import cv2
import numpy as np
from pathlib import Path

# Base directories
BASE_DATASET_DIR = Path("E:/bachelor_data/drone_mot_dataset")
BASE_LABELS_DIR = Path("E:/bachelor_data/labels")

def verify_sequence(seq_name: str, split_name: str):
    seq_dir = BASE_DATASET_DIR / f"drone-{split_name}" / seq_name
    gt_file = seq_dir / "gt/gt.txt"
    original_videos_dir = BASE_LABELS_DIR / split_name
    
    # Find matching original video in the split directory
    video_path = None
    for ext in (".mp4", ".avi", ".mkv", ".mov"):
        temp_path = original_videos_dir / f"{seq_name}{ext}"
        if temp_path.exists():
            video_path = temp_path
            break
            
    if not video_path:
        print(f"Error: Original video for sequence '{seq_name}' not found in {original_videos_dir}")
        return
        
    if not gt_file.exists():
        print(f"Error: Prepared gt.txt not found at {gt_file}")
        return

    # Load normalized annotations
    gt_data = np.loadtxt(gt_file, delimiter=',')
    if gt_data.ndim == 1:
        gt_data = np.array([gt_data])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    print(f"\n--- Visualizing {split_name.upper()} sequence: {seq_name} ---")
    print("Controls: [SPACE] - Pause/Resume | [Q or ESC] - Return to menu | [Any other key] - Next frame")
    
    frame_num = 1
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video sequence.")
                break
                
            # Filter detections for current frame
            frame_dets = gt_data[gt_data[:, 0] == frame_num]
            
            for det in frame_dets:
                track_id = int(det[1])
                left, top, w, h = det[2:6]
                class_id = int(det[7])
                
                x1, y1 = int(left), int(top)
                x2, y2 = int(left + w), int(top + h)
                
                # Draw bounding box (Green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"ID: {track_id} (Cls: {class_id})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display frame info
            cv2.putText(frame, f"Frame: {frame_num}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Dataset Verification", frame)
            frame_num += 1

        # Handle controls
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

def select_sequence_menu(split_name: str):
    split_dir = BASE_DATASET_DIR / f"drone-{split_name}"
    if not split_dir.exists():
        print(f"Error: Directory for split '{split_name}' does not exist.")
        return

    sequences = [d.name for d in split_dir.iterdir() if d.is_dir()]
    if not sequences:
        print(f"No prepared sequences found in split '{split_name}'.")
        return

    while True:
        print(f"\nAvailable '{split_name.upper()}' sequences:")
        for idx, seq in enumerate(sequences):
            print(f"[{idx}] {seq}")
        print("[-1] Back to Main Menu")

        try:
            choice = int(input("\nEnter sequence index: "))
            if choice == -1:
                break
            if 0 <= choice < len(sequences):
                verify_sequence(sequences[choice], split_name)
            else:
                print("Invalid index.")
        except ValueError:
            print("Please enter a valid integer.")

def main():
    if not BASE_DATASET_DIR.exists():
        print(f"Error: Dataset directory does not exist: {BASE_DATASET_DIR}")
        return

    while True:
        print("\n=== Main Verification Menu ===")
        print("[0] Verify TEST split (drone-test)")
        print("[1] Verify VAL split (drone-val)")
        print("[-1] Exit")

        try:
            split_choice = int(input("\nSelect split: "))
            if split_choice == -1:
                break
            elif split_choice == 0:
                select_sequence_menu("test")
            elif split_choice == 1:
                select_sequence_menu("val")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid integer.")

if __name__ == "__main__":
    main()