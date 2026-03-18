"""Pipeline data reset utility.

Empties all intermediate and output directories so the pipeline can be
re-run cleanly from Stage 1 (or any chosen stage).

Raw downloads (``raw_downloads/``) are intentionally preserved to avoid
re-downloading large video files.

Usage::

    python src/utils/clear.py            # interactive confirmation
    python src/utils/clear.py --yes      # skip confirmation (CI / scripts)
        python backend/data_generation/src/utils/clear.py --yes
    python src/utils/clear.py --stage 3  # clear only Stage 3+ outputs
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# clear.py lives at  <data_generation_root>/src/utils/clear.py
# Two levels up →    <data_generation_root>/
_DATA_ROOT = Path(__file__).resolve().parent.parent.parent

# The pipeline checkpoint and log files live one level above data_generation:
#   <project_root>/logs/
# Config: output_dirs.logs = "../backend/data_generation/logs"
# Resolved relative to project root → <project_root>/logs/
_LOGS_DIR = _DATA_ROOT.parent.parent / "logs"

# ---------------------------------------------------------------------------
# Directory groups keyed by the earliest stage that writes to them.
# Clearing stage N also clears everything produced by stages > N.
# ---------------------------------------------------------------------------

_STAGE_DIRS: dict[int, list[str]] = {
    2: ["passed_files", "rejected_files"],
    3: ["preprocessed"],
    4: ["final_files", "validation_failed"],
    5: ["stt_and_vad", "subtitles"],
    6: ["segments"],
    7: ["synthesized"],
    8: ["dataset"],
}


def _rel(path: Path) -> str:
    """Return a short relative path for display."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _clear_dir(path: Path) -> None:
    """Delete and recreate *path* as an empty directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _clear_logs() -> None:
    """Remove all ``.log`` files and the pipeline checkpoint from the logs dir."""
    if not _LOGS_DIR.exists():
        print(f"  skip  {_rel(_LOGS_DIR)}  (does not exist)")
        return

    removed = 0
    for f in _LOGS_DIR.iterdir():
        if f.suffix == ".log" or f.name == "pipeline_checkpoint.json":
            f.unlink()
            removed += 1

    if removed:
        print(f"  cleared  {_rel(_LOGS_DIR)}  ({removed} file(s) removed)")
    else:
        print(f"  empty    {_rel(_LOGS_DIR)}")


def clear_data(from_stage: int = 1, *, yes: bool = False) -> None:
    """Clear all pipeline outputs produced from *from_stage* onward.

    Args:
        from_stage: Earliest stage whose outputs should be deleted (1–8).
            Passing 1 clears everything except raw downloads.
        yes: Skip the interactive confirmation prompt.
    """
    # Collect target directories
    target_dirs: list[Path] = []
    for stage, folder_names in sorted(_STAGE_DIRS.items()):
        if stage >= from_stage:
            for name in folder_names:
                target_dirs.append(_DATA_ROOT / name)

    clear_logs = from_stage <= 1  # logs only make sense to clear on full reset

    # Show what will be deleted
    print()
    print(f"Data root : {_rel(_DATA_ROOT)}")
    print(f"From stage: {from_stage}")
    print()
    print("Will delete:")
    for d in target_dirs:
        marker = "EXISTS" if d.exists() else "absent"
        print(f"  [{marker:6s}]  {_rel(d)}/")
    if clear_logs:
        print(f"  [logs  ]  {_rel(_LOGS_DIR)}/*.log  +  pipeline_checkpoint.json")
    print()

    # Confirmation
    if not yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    # Execute
    print()
    for d in target_dirs:
        if d.exists():
            _clear_dir(d)
            print(f"  cleared  {_rel(d)}/")
        else:
            print(f"  skip     {_rel(d)}/  (does not exist)")

    if clear_logs:
        _clear_logs()

    print()
    print("Done. Pipeline can now be re-run from scratch.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset pipeline output directories for a fresh debug run.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Clear outputs from stage N onward (default: 1 = full reset). "
            "Example: --stage 6 clears only segments/, synthesized/, dataset/."
        ),
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    clear_data(from_stage=args.stage, yes=args.yes)
