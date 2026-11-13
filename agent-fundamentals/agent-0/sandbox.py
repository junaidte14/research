"""sandbox.py: where run_command actually runs: on your machine ('local') or inside Docker ('docker')."""
import os
import shutil
import signal
import subprocess
import uuid
from functools import lru_cache

from safety import WORKSPACE, clean_env, truncate

IMAGE = "agent-sandbox"   # built from the Dockerfile in this folder


def backend() -> str:
    """Read at call time, so values loaded from .env are already available."""
    return "docker" if os.getenv("AGENT_SANDBOX", "local").strip().lower() == "docker" else "local"


@lru_cache(maxsize=1)
def docker_ready() -> bool:
    """True if the docker CLI exists, the daemon is running and our image has been built."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=15)
        return result.returncode == 0
    except Exception:
        return False


def describe() -> str:
    if backend() == "docker":
        return "docker (isolated container)" if docker_ready() else "docker requested but NOT ready (commands will be refused)"
    return "local (runs on your machine, approval required)"


# ---------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------
def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the shell AND everything it started. A plain proc.kill() leaves child processes running,
    and on Windows that makes the program hang while waiting for them."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        os.killpg(proc.pid, signal.SIGKILL)


def run_local(command: str, timeout: int):
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=WORKSPACE,
        env=clean_env(),                       # no API keys
        stdin=subprocess.DEVNULL,              # never wait for keyboard input
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=(os.name != "nt"),   # lets us kill the whole process group on Linux/macOS
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, err = proc.communicate()
        return None, out, err


# ---------------------------------------------------------------
# Docker backend
# ---------------------------------------------------------------
def run_docker(command: str, timeout: int):
    name = f"agent-sandbox-{uuid.uuid4().hex[:8]}"
    cmd = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",                       # no internet: nothing can be downloaded or uploaded
        "--memory", "256m", "--cpus", "1",         # resource caps
        "--pids-limit", "128",                     # stops fork bombs
        "--cap-drop", "ALL",                       # drop Linux privileges
        "--security-opt", "no-new-privileges",
        "--read-only", "--tmpfs", "/tmp",          # container filesystem is read-only except /tmp
        "--mount", f"type=bind,source={WORKSPACE},target=/workspace",   # the ONLY host folder it can see
        "-w", "/workspace",
        IMAGE, "sh", "-c", command,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, stdin=subprocess.DEVNULL,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        # Killing the docker CLI alone would leave the container running, so stop it explicitly.
        subprocess.run(["docker", "kill", name], capture_output=True)
        return None, "", ""


# ---------------------------------------------------------------
# Single entry point
# ---------------------------------------------------------------
def run_sandboxed(command: str, timeout: int = 30) -> str:
    timeout = max(1, min(int(timeout), 120))
    if backend() == "docker":
        if not docker_ready():
            # Fail closed: never silently fall back to running on the host.
            return ("Error: AGENT_SANDBOX=docker but Docker is not ready "
                    "(is Docker running and the image built?). Refusing to run on the host.")
        code, out, err = run_docker(command, timeout)
    else:
        code, out, err = run_local(command, timeout)

    if code is None:
        return f"Error: command timed out after {timeout} seconds and was stopped."
    return truncate(f"exit code: {code}\n--- stdout ---\n{out}\n--- stderr ---\n{err}")