import os
import sys
import subprocess
import re
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv
import requests
# Load environment variables
load_dotenv()

# ================= CONFIGURATION =================
REPO_ID = "hannusia123123/drone_dataset"
CVAT_HOST = "localhost"
CVAT_PORT = "8080"
LOCAL_DIR = "E:/bachelor_data/labels"

CVAT_USER = os.getenv("CVAT_USER")      
CVAT_PASS = os.getenv("CVAT_PASS")   
# =================================================

api = HfApi()

def run_cmd(cmd):
    """Executes a command in the terminal and returns the combined stdout and stderr"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Command execution error: {cmd}\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}")
        sys.exit(1)
    return result.stdout + "\n" + result.stderr

def pull_batch(split, batch_name):
    """Downloads a batch from HF and automatically creates CVAT tasks for all videos"""
    print(f"📥 Searching for {batch_name} in split '{split}' on Hugging Face...")
    
    batch_prefix = f"cvat_batches/{split}/{batch_name}/"
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    
    # Find all ZIP archives in this batch
    zip_files = [f for f in files if f.startswith(batch_prefix) and f.endswith('.zip')]
    
    if not zip_files:
        print("❌ No annotation archives found in this batch.")
        return

    os.makedirs(LOCAL_DIR, exist_ok=True)

    # Process each ZIP archive in the batch
    for zip_file_repo_path in zip_files:
        # Extract the clean video name
        zip_filename = os.path.basename(zip_file_repo_path)
        base_name = zip_filename.replace('_mot.zip', '')
        
        # Look for the original video in the repository
        video_repo_path = next((f for f in files if f.startswith(f"{split}/{base_name}.") and f.endswith(('.mp4', '.mov', '.m4v'))), None)
        
        if not video_repo_path:
            print(f"⚠️ Skipping {base_name}: could not find the original video on HF.")
            continue
            
        print(f"\n⏳ Downloading {base_name}...")
        local_video_path = hf_hub_download(repo_id=REPO_ID, filename=video_repo_path, repo_type="dataset", local_dir=LOCAL_DIR)
        local_zip_path = hf_hub_download(repo_id=REPO_ID, filename=zip_file_repo_path, repo_type="dataset", local_dir=LOCAL_DIR)

        print(f"🚀 Creating a CVAT task for {base_name}...")
        task_name = f"{base_name}_{split}"
        create_cmd = f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} --auth {CVAT_USER}:{CVAT_PASS} create "{task_name}" --labels "[{{\\"name\\": \\"pedestrian\\\"}}, {{\\"name\\": \\"drone\\\"}}]" local "{local_video_path}"'
        output = run_cmd(create_cmd)
        
        # Flexible regex to catch the Task ID regardless of formatting/case
        match = re.search(r'(?i)ID[\s:]*(\d+)', output)
        if not match:
            print(f"❌ Failed to extract Task ID for {base_name}.")
            print(f"🔍 Raw CVAT Output:\n{output.strip()}")
            continue
            
        task_id = match.group(1)
        print(f"✅ Task created! Task ID: {task_id}")

        print("📦 Uploading SAM 3 annotations to task...")
        # FIX: Changed 'import' to 'upload' for annotations
        upload_cmd = f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} --auth {CVAT_USER}:{CVAT_PASS} upload {task_id} "{local_zip_path}" --format "MOT 1.1"'
        run_cmd(upload_cmd)
        
        print(f"🎉 Done: http://{CVAT_HOST}:{CVAT_PORT}/tasks/{task_id}")

def push_batch(task_id, split, batch_name):
    """Exports annotations from CVAT and pushes them to labels on HF with proper video names"""
    print(f"📤 Exporting annotation from CVAT (Task ID: {task_id})...")
    
    # 1. Ask CVAT for the task name to get the correct video name
    try:
        response = requests.get(
            f"http://{CVAT_HOST}:{CVAT_PORT}/api/tasks/{task_id}",
            auth=(CVAT_USER, CVAT_PASS)
        )
        response.raise_for_status()
        task_info = response.json()
        task_name = task_info.get("name", f"task_{task_id}")
        # Clean up the name (e.g., 'delta_rgb_field_053_train' -> 'delta_rgb_field_053')
        video_name = task_name.replace(f"_{split}", "")
    except Exception as e:
        print(f"⚠️ Could not fetch task name from CVAT: {e}. Using Task ID instead.")
        video_name = f"task_{task_id}"

    os.makedirs(LOCAL_DIR, exist_ok=True)
    export_filename = f"{video_name}.zip"
    export_path = os.path.join(LOCAL_DIR, export_filename)
    
    # 2. Dump annotations
    dump_cmd = f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} --auth {CVAT_USER}:{CVAT_PASS} dump {task_id} "{export_path}" --format "MOT 1.1"'
    run_cmd(dump_cmd)
    
    # 3. Upload to Hugging Face into the correct folder structure
    print(f"☁️ Sending the corrected annotation to Hugging Face as {export_filename}...")
    path_in_repo = f"labels/{split}/{batch_name}/{export_filename}"
    
    api.upload_file(
        path_or_fileobj=export_path,
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Upload manually validated MOT annotations for {video_name}"
    )
    
    print(f"✅ Success! The verified archive has been saved to HF: {path_in_repo}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Import to CVAT:  python cvat_bridge.py pull train batch_001")
        print("  Export to HF:    python cvat_bridge.py push <TASK_ID> train batch_001")
        sys.exit(1)

    command = sys.argv[1]
    if command == "pull" and len(sys.argv) == 4:
        pull_batch(sys.argv[2], sys.argv[3])
    elif command == "push" and len(sys.argv) == 5:
        push_batch(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("❌ Incorrect number of arguments.")