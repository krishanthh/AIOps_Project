"""Fixed-command pipeline runner with OS-level exclusion and durable run events."""

import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from brainai.common import ROOT, read_json, write_json

STAGES = ["check", "acquire", "split", "prepare", "train", "evaluate", "package"]
COMMANDS = {stage: [sys.executable, "-m", "dvc", "repro", stage] for stage in STAGES}


def run():
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    with open(ROOT / "reports/pipeline.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another pipeline is running", file=sys.stderr)
            return 2
        previous = read_json(ROOT / "reports/pipeline.json", {})
        if previous.get("status") == "running":
            previous.update(
                status="interrupted",
                finished_at=time.time(),
                error="Previous runner ended without recording completion",
            )
            for stage in previous.get("stages", {}).values():
                if stage.get("status") == "running":
                    stage["status"] = "interrupted"
            write_json(ROOT / "reports/pipeline-runs" / f"{previous['run_id']}.json", previous)
        run_id = uuid.uuid4().hex
        state = {
            "run_id": run_id,
            "pid": os.getpid(),
            "status": "running",
            "started_at": time.time(),
            "stages": {s: {"status": "pending"} for s in STAGES},
        }
        try:
            for stage in STAGES:
                state["stages"][stage] = {"status": "running", "started_at": time.time()}
                write_json(ROOT / "reports/pipeline.json", state)
                started = time.perf_counter()
                with open(ROOT / "logs" / f"{run_id}-{stage}.log", "w") as log:
                    result = subprocess.run(
                        COMMANDS[stage],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={
                            **os.environ,
                            "PATH": str(Path(sys.executable).parent)
                            + os.pathsep
                            + os.environ.get("PATH", ""),
                        },
                    )
                state["stages"][stage].update(
                    {
                        "seconds": time.perf_counter() - started,
                        "status": "success" if result.returncode == 0 else "failed",
                        "exit_code": result.returncode,
                        "log": f"{run_id}-{stage}.log",
                    }
                )
                write_json(ROOT / "reports/pipeline.json", state)
                if result.returncode:
                    raise RuntimeError(f"Stage {stage} failed; see its log")
            state["status"] = "success"
        except (Exception, KeyboardInterrupt) as exc:
            state["status"], state["error"] = "failed", str(exc)
        finally:
            state["finished_at"] = time.time()
            write_json(ROOT / "reports/pipeline.json", state)
            write_json(ROOT / "reports/pipeline-runs" / f"{run_id}.json", state)
        print(json.dumps(state, indent=2))
        return 0 if state["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(run())
