import subprocess
import sys

def run_cmd(cmd: str) -> str:
    """Executes a terminal command and returns combined stdout and stderr."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"❌ Command execution error (exit code {result.returncode})\n"
            f"STDERR: {result.stderr}\n"
            f"STDOUT: {result.stdout}"
        )
        sys.exit(1)
    return result.stdout + "\n" + result.stderr