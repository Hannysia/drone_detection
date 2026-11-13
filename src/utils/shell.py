import os
import subprocess


def run_cmd(args: list[str], *, password: str) -> str:
    env = os.environ.copy()
    env.pop("CVAT_ACCESS_TOKEN", None)
    env.pop("CVAT_PASS", None)
    env["PASS"] = password

    result = subprocess.run(
        args,
        shell=False,
        env=env,
        capture_output=True,
        text=True,
    )

    output = result.stdout + "\n" + result.stderr

    if result.returncode:
        output = output.replace(password, "[REDACTED]")
        print(f"Command failed (exit code {result.returncode})\n{output}")
        raise SystemExit(1)

    return output