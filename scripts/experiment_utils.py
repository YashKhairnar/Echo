import json
import subprocess
from datetime import datetime
from pathlib import Path


def create_run_dir(root, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{timestamp}_{prefix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_metadata(run_dir, metadata):
    metadata = dict(metadata)
    metadata["created_at"] = datetime.now().isoformat(timespec="seconds")

    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["git_commit"] = "unknown"

    with (run_dir / "config.json").open("w") as file:
        json.dump(metadata, file, indent=2, default=str)

    notes_path = run_dir / "notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            "# Experiment Notes\n\n"
            "## What changed\n\n"
            "## Result\n\n"
            "## What worked\n\n"
            "## What to improve\n"
        )
