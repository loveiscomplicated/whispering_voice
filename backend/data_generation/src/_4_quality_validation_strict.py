"""Strict post-preprocessing audio quality validation (Stage 4).

Validates preprocessed audio files before running STT / VAD.  By this point
all files should be 16 kHz mono WAVs of a consistent length and loudness, so
this stage enforces tight thresholds.

Typical usage::

    python src/_4_quality_validation_strict.py \\
        --input-dir preprocessed \\
        --config config/generation.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.audio_processor import load_audio  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_EXPECTED_SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StrictValidationResult:
    """Holds the outcome of a single audio file strict validation.

    Attributes:
        file_path: Absolute path to the validated file.
        passed: True when all checks pass.
        failure_reasons: List of human-readable failure descriptions.
        format_ok: File extension is in the supported list.
        duration_ms: Audio duration in milliseconds (None if unreadable).
        duration_ok: Duration is within the configured range.
        rms_energy_db: RMS energy in dBFS (None if unreadable).
        energy_ok: Energy is within the configured range.
        sample_rate: Detected sample rate in Hz (None if unreadable).
        sample_rate_ok: Sample rate matches the required value.
        file_intact: File was loaded without errors.
    """

    file_path: str
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    format_ok: bool = False
    duration_ms: float | None = None
    duration_ok: bool = False
    rms_energy_db: float | None = None
    energy_ok: bool = False
    sample_rate: int | None = None
    sample_rate_ok: bool = False
    file_intact: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dictionary suitable for a DataFrame row.

        Returns:
            Dictionary with all fields; ``failure_reasons`` is serialised as a
            semicolon-separated string.
        """
        d = asdict(self)
        d["failure_reasons"] = (
            "; ".join(self.failure_reasons) if self.failure_reasons else ""
        )
        return d


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class QualityValidatorStrict:
    """Validate preprocessed audio files before STT / VAD processing.

    Thresholds are read from the ``quality_validation_strict`` section of the
    pipeline configuration.

    Args:
        config: Pipeline configuration dictionary (from ``load_config``).
        logger: Logger instance used for diagnostic messages.
    """

    def __init__(self, config: dict[str, Any], logger: Any) -> None:
        self._config = config
        self._logger = logger
        self._qv_cfg: dict[str, Any] = config.get("quality_validation_strict", {})

        self._min_duration_ms: float = self._qv_cfg.get("min_audio_length_ms", 1_000)
        self._max_duration_ms: float = self._qv_cfg.get("max_audio_length_ms", 30_000)
        self._min_energy_db: float = self._qv_cfg.get("min_energy_db", -40.0)
        self._max_energy_db: float = self._qv_cfg.get("max_energy_db", -10.0)
        self._required_sr: int = int(
            self._qv_cfg.get("require_sample_rate", _EXPECTED_SAMPLE_RATE)
        )

        self._results: list[StrictValidationResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_audio(self, audio_path: str) -> dict[str, Any]:
        """Validate a single audio file and return the result as a dictionary.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Flat dictionary representation of :class:`StrictValidationResult`.
        """
        result = self._run_checks(audio_path)
        self._results.append(result)
        status = "PASS" if result.passed else "FAIL"
        self._logger.info(
            f"[{status}] {Path(audio_path).name}"
            + (f" — {result.failure_reasons}" if result.failure_reasons else "")
        )
        return result.to_dict()

    def validate_batch(self, directory: str) -> pd.DataFrame:
        """Validate all audio files in a directory.

        Files with unsupported extensions are silently skipped.

        Args:
            directory: Path to the directory containing audio files.

        Returns:
            DataFrame with one row per validated file and columns matching
            :class:`StrictValidationResult` fields.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Input directory not found: {directory}")

        audio_files = sorted(
            p
            for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )

        if not audio_files:
            self._logger.warning(f"No audio files found in: {directory}")
            return pd.DataFrame()

        self._logger.info(f"Validating {len(audio_files)} file(s) in: {directory}")

        rows = [self.validate_audio(str(p)) for p in audio_files]
        df = pd.DataFrame(rows)

        n_pass = int(df["passed"].sum())
        self._logger.info(
            f"Batch done — {n_pass}/{len(df)} passed "
            f"({100 * n_pass / len(df):.1f}%)"
        )
        return df

    def generate_report(self, output_path: str) -> str:
        """Write a CSV report and a JSON list of passing files.

        The method writes two files:

        - ``<output_path>`` — CSV with all validation results.
        - ``<stem>_passed.json`` — JSON array of paths that passed validation.

        Args:
            output_path: Destination path for the CSV report.

        Returns:
            Absolute path to the written CSV file.

        Raises:
            RuntimeError: If ``validate_audio`` / ``validate_batch`` has not
                been called yet.
        """
        if not self._results:
            raise RuntimeError(
                "No validation results to report. "
                "Run validate_audio() or validate_batch() first."
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([r.to_dict() for r in self._results])
        df.to_csv(out, index=False, encoding="utf-8")
        self._logger.info(f"CSV report saved: {out}")

        # Passing files JSON
        passed_paths = [r.file_path for r in self._results if r.passed]
        json_path = out.with_name(out.stem + "_passed.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                    "total": len(self._results),
                    "passed": len(passed_paths),
                    "files": passed_paths,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        self._logger.info(f"Passed-files JSON saved: {json_path}")

        return str(out)

    def copy_files(self, output_dirs: dict[str, str]) -> dict[str, int]:
        """Copy validated files to final/failed output directories.

        Uses ``shutil.copy2`` to preserve file metadata. Existing destination
        files are skipped to make this operation idempotent.

        Args:
            output_dirs: Dictionary with keys ``"final_files"`` and
                ``"validation_failed"`` pointing to destination directories.

        Returns:
            Dictionary with ``"passed"`` and ``"rejected"`` counts.

        Raises:
            RuntimeError: If ``validate_audio`` / ``validate_batch`` has not
                been called yet.
        """
        if not self._results:
            raise RuntimeError(
                "No validation results available. "
                "Run validate_audio() or validate_batch() first."
            )

        final_dir = Path(output_dirs.get("final_files", "./final_files"))
        failed_dir = Path(output_dirs.get("validation_failed", "./validation_failed"))
        final_dir.mkdir(parents=True, exist_ok=True)
        failed_dir.mkdir(parents=True, exist_ok=True)

        n_passed = 0
        n_rejected = 0

        for result in self._results:
            src = Path(result.file_path)
            dest_dir = final_dir if result.passed else failed_dir
            dest = dest_dir / src.name

            if dest.exists():
                self._logger.debug(f"Skipping existing: {dest.name}")
            else:
                try:
                    shutil.copy2(str(src), str(dest))
                    self._logger.debug(
                        f"Copied to {'final' if result.passed else 'failed'}: {src.name}"
                    )
                except OSError as exc:
                    self._logger.error(f"Failed to copy {src.name}: {exc}")
                    continue

            if result.passed:
                n_passed += 1
            else:
                n_rejected += 1

        self._logger.info(f"Files copied — passed: {n_passed}, rejected: {n_rejected}")
        return {"passed": n_passed, "rejected": n_rejected}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_checks(self, audio_path: str) -> StrictValidationResult:
        """Execute all strict validation checks for a single file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Populated :class:`StrictValidationResult` instance.
        """
        result = StrictValidationResult(file_path=str(Path(audio_path).resolve()))
        reasons: list[str] = []

        # 1. Format check
        suffix = Path(audio_path).suffix.lower()
        result.format_ok = suffix in _SUPPORTED_EXTENSIONS
        if not result.format_ok:
            reasons.append(f"unsupported format '{suffix}'")

        # 2. Load audio (corruption / existence check)
        audio: np.ndarray | None = None
        sr: int | None = None
        try:
            audio, sr = load_audio(audio_path, sr=None)
            result.file_intact = True
        except FileNotFoundError:
            reasons.append("file not found")
            result.passed = False
            result.failure_reasons = reasons
            return result
        except Exception as exc:
            reasons.append(f"corrupted or unreadable ({exc})")
            result.passed = False
            result.failure_reasons = reasons
            return result

        # 3. Sample rate check
        result.sample_rate = sr
        result.sample_rate_ok = sr == self._required_sr
        if not result.sample_rate_ok:
            reasons.append(f"sample rate {sr} Hz (expected {self._required_sr} Hz)")

        # 4. Duration check
        duration_ms = len(audio) / sr * 1_000
        result.duration_ms = duration_ms
        result.duration_ok = (
            self._min_duration_ms <= duration_ms <= self._max_duration_ms
        )
        if not result.duration_ok:
            reasons.append(
                f"duration {duration_ms:.0f} ms outside "
                f"[{self._min_duration_ms:.0f}, {self._max_duration_ms:.0f}] ms"
            )

        # 5. RMS energy check
        rms = float(np.sqrt(np.mean(audio**2)))
        rms_db = 20.0 * np.log10(rms + 1e-9)
        result.rms_energy_db = rms_db
        result.energy_ok = self._min_energy_db <= rms_db <= self._max_energy_db
        if not result.energy_ok:
            reasons.append(
                f"RMS energy {rms_db:.1f} dBFS outside "
                f"[{self._min_energy_db:.1f}, {self._max_energy_db:.1f}] dBFS"
            )

        result.failure_reasons = reasons
        result.passed = len(reasons) == 0
        return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict post-preprocessing audio quality validation (Stage 4).",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing preprocessed audio files.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to generation.yaml config file.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for CSV + JSON report (defaults to <logs_dir>).",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy passed/failed files to configured output directories.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (optional).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the strict validation stage.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    args = _parse_args(argv)
    config = load_config(args.config)

    logs_dir: str = config.get("output_dirs", {}).get("logs", "./logs")
    log_file = args.log_file or str(Path(logs_dir) / "4_quality_validation_strict.log")
    log = setup_logger("quality_validation_strict", log_file=log_file)

    validator = QualityValidatorStrict(config=config, logger=log)
    df = validator.validate_batch(args.input_dir)

    if df.empty:
        log.warning("No files were validated. Exiting.")
        sys.exit(0)

    report_dir = args.report_dir or logs_dir
    report_path = str(Path(report_dir) / "quality_validation_strict_report.csv")
    validator.generate_report(report_path)

    if args.copy_files:
        out_dirs = config.get("output_dirs", {})
        validator.copy_files(out_dirs)

    n_pass = int(df["passed"].sum())
    print(f"\nResults: {n_pass}/{len(df)} files passed strict quality validation.")
    print(f"Report : {report_path}")


if __name__ == "__main__":
    main()
