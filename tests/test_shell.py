from types import SimpleNamespace

import pytest

from src.integration import cvat
from src.utils import shell


@pytest.mark.parametrize("operation", ["create", "upload", "dump"])
@pytest.mark.parametrize("status", [0, 2])
def test_cvat_cli_credentials(monkeypatch, capsys, operation, status):
    secret = "example-password"
    path = "folder with spaces/file.zip"

    monkeypatch.setattr(cvat, "CVAT_USER", "reviewer")
    monkeypatch.setattr(cvat, "CVAT_PASS", secret)

    def fake_run(args, *, shell, env, capture_output, text):
        assert shell is False
        assert args[args.index("--auth"):] == [
            "--auth", "reviewer", operation, path,
        ]
        assert secret not in repr(args)
        assert env["PASS"] == secret

        return SimpleNamespace(
            returncode=status,
            stdout="Task ID: 700",
            stderr=secret if status else "",
        )

    monkeypatch.setattr(shell.subprocess, "run", fake_run)

    if status:
        with pytest.raises(SystemExit) as exc:
            cvat.run_cvat([operation, path])

        assert exc.value.code == 1
        output = capsys.readouterr().out
    else:
        output = cvat.run_cvat([operation, path])

    assert "Task ID: 700" in output
    assert secret not in output