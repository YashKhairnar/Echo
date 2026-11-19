import json
import subprocess
from datetime import datetime
from pathlib import Path


def create_run_dir(root, prefix):
    """Make a new timestamped run folder, so a fresh run never overwrites results.

    If a folder with the same name already exists (two runs started in the
    same second), a number is added: ..._2, ..._3, ...
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}_{prefix}"
    for attempt in range(1, 100):
        run_dir = Path(root) / (name if attempt == 1 else f"{name}_{attempt}")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a run folder for {name}")


def save_run_metadata(run_dir, metadata):
    # Store configuration and Git revision next to the metrics and checkpoints.
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
