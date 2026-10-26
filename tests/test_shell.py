
import subprocess
from types import SimpleNamespace

import pytest

from src.utils.shell import run_cmd


@pytest.mark.parametrize(
    "operation",
    ["create", "upload", "dump"],
)
def test_run_cmd_stops_on_cli_failure(monkeypatch, operation):
    command = f"cvat-cli {operation}"

    def fake_run(cmd, **kwargs):
        assert cmd == command
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True

        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr=f"{operation} failed",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc:
        run_cmd(command)

    assert exc.value.code == 1


def test_run_cmd_returns_output_on_success(monkeypatch):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="Task ID: 700",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_cmd("cvat-cli create") == "Task ID: 700\n"


def test_run_cmd_does_not_print_credentials(monkeypatch, capsys):
    secret = "test-secret-do-not-print"

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Authentication failed",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        run_cmd(
            f"cvat-cli --auth user:{secret} upload 695"
        )

    output = capsys.readouterr().out

    assert secret not in output
    assert "Authentication failed" in output
