"""Run only this project after dependencies finish; do not retry failed jobs."""
from pathlib import Path
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / ".venv/bin/python")
STATUS = ROOT / "cloud_deployment.json"

def write(phase, **extra):
    value = {"phase": phase, "updated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "driver_pid": os.getpid(), **extra}
    temp = STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(STATUS)
    print(json.dumps(value, ensure_ascii=False), flush=True)

try:
    pid = int((ROOT / "dependency_install_pid.txt").read_text())
    started = time.monotonic()
    write("waiting_for_dependencies", dependency_pid=pid)
    while True:
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        if time.monotonic() - started > 1800:
            raise TimeoutError("Dependency installation exceeded 30 minutes")
        time.sleep(10)
    expected = {"numpy": "2.2.6", "pandas": "2.3.3", "matplotlib": "3.10.9", "einops": "0.8.1", "reformer-pytorch": "1.4.4"}
    observed = {name: importlib.metadata.version(name) for name in expected}
    if observed != expected:
        raise RuntimeError(f"Dependency versions mismatch: {observed}")
    write("dependencies_ready", packages=observed)
    for phase, args in [
        ("auditing_data", ["run.py", "--stage", "audit"]),
        ("preparing_data", ["run.py", "--stage", "prepare"]),
        ("regression_checks", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
        ("full_training_and_figures", ["-u", "run.py", "--stage", "all"]),
    ]:
        write(phase)
        subprocess.run([PYTHON, "-B", *args], cwd=ROOT, check=True)
    write("completed", note="49 training jobs, 35 final seed results, metrics and figures finished")
except Exception as error:
    write("failed", error=str(error))
    raise
