"""Post-processing quality validation for synthesized audio — Stage 5.

Validates synthesized WAV files produced in Stage 4. In addition to the
basic checks performed in Stage 2, this validator:

- Verifies that a JSON sidecar exists and is internally consistent.
- Measures the *actual* SNR of each synthesized file against its source.
- Produces a per-SNR-level breakdown in the final report.

Typical usage::

    python src/5_quality_validation_2.py --config config/generation.yaml
"""

from __future__ import annotations

import argparse
import json
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

_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

# Required keys in the synthesis sidecar JSON
_REQUIRED_META_KEYS = {
    "audio_id",
    "source_asmr",
    "source_noise",
    "noise_type",
    "target_snr_db",
    "audio_characteristics",
}

# Tolerance (dB) between target and measured SNR before flagging
_SNR_TOLERANCE_DB = 3.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult2:
    """Outcome of a single synthesized-audio validation.

    Attributes:
        file_path: Absolute path to the validated WAV file.
        passed: True when all checks pass.
        failure_reasons: Human-readable failure descriptions.

        format_ok: File extension is supported.
        file_intact: File was loaded without errors.
        duration_ms: Audio duration in milliseconds.
        duration_ok: Duration is within the configured range.
        rms_energy_db: RMS energy in dBFS.
        energy_ok: Energy within the configured range.
        sample_rate: Detected sample rate.
        sample_rate_ok: Sample rate equals 16 kHz.

        metadata_exists: JSON sidecar was found.
        metadata_consistent: All required keys present and values valid.
        target_snr_db: Target SNR recorded in the sidecar.
        measured_snr_db: Actual SNR computed from the source signals.
        snr_within_tolerance: ``|target − measured| ≤ _SNR_TOLERANCE_DB``.
    """

    file_path: str
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    format_ok: bool = False
    file_intact: bool = False
    duration_ms: float | None = None
    duration_ok: bool = False
    rms_energy_db: float | None = None
    energy_ok: bool = False
    sample_rate: int | None = None
    sample_rate_ok: bool = False

    metadata_exists: bool = False
    metadata_consistent: bool = False
    target_snr_db: float | None = None
    measured_snr_db: float | None = None
    snr_within_tolerance: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dictionary suitable for a DataFrame row."""
        d = asdict(self)
        d["failure_reasons"] = "; ".join(self.failure_reasons) if self.failure_reasons else ""
        return d


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class QualityValidator2:
    """Validate synthesized audio files after Stage 4.

    Args:
        config: Pipeline configuration dictionary (from ``load_config``).
        logger: Logger instance.
    """

    _EXPECTED_SR = 16_000

    def __init__(self, config: dict[str, Any], logger: Any) -> None:
        self._config = config
        self._logger = logger

        qv = config.get("quality_validation", {})
        self._min_duration_ms: float = qv.get("min_audio_length_ms", 1_000)
        self._max_duration_ms: float = qv.get("max_audio_length_ms", 30_000)
        self._min_energy_db: float = qv.get("min_energy_db", -40.0)
        self._max_energy_db: float = qv.get("max_energy_db", -10.0)

        self._results: list[ValidationResult2] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_audio(self, audio_path: str) -> dict[str, Any]:
        """Validate a single synthesized audio file.

        Args:
            audio_path: Path to the synthesized WAV file.

        Returns:
            Flat dictionary representation of :class:`ValidationResult2`.
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
        """Validate all supported audio files in *directory* (recursive).

        Args:
            directory: Root directory to scan (e.g. ``synthesized/``).

        Returns:
            DataFrame with one row per validated file.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Input directory not found: {directory}")

        audio_files = sorted(
            p for p in dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )

        if not audio_files:
            self._logger.warning(f"No audio files found under: {directory}")
            return pd.DataFrame()

        self._logger.info(
            f"Validating {len(audio_files)} synthesized file(s) under: {directory}"
        )

        rows = [self.validate_audio(str(p)) for p in audio_files]
        df = pd.DataFrame(rows)

        n_pass = int(df["passed"].sum())
        self._logger.info(
            f"Batch done — {n_pass}/{len(df)} passed "
            f"({100 * n_pass / len(df):.1f}%)"
        )
        return df

    def measure_actual_snr(self, asmr_path: str, noise_path: str) -> float:
        """Compute the SNR (dB) between a clean signal and a noise signal.

        Both files are resampled to 16 kHz mono. The shorter array is
        zero-padded to match the longer one before computation.

        Args:
            asmr_path: Path to the clean ASMR audio file.
            noise_path: Path to the noise audio file.

        Returns:
            SNR in dB. Returns ``float('inf')`` when noise is silent.

        Raises:
            FileNotFoundError: If either path does not exist.
        """
        signal, sr = load_audio(asmr_path, sr=self._EXPECTED_SR)
        noise, _ = load_audio(noise_path, sr=self._EXPECTED_SR)

        # Pad to same length for fair RMS comparison
        max_len = max(len(signal), len(noise))
        if len(signal) < max_len:
            signal = np.pad(signal, (0, max_len - len(signal)))
        if len(noise) < max_len:
            noise = np.pad(noise, (0, max_len - len(noise)))

        rms_s = float(np.sqrt(np.mean(signal ** 2)))
        rms_n = float(np.sqrt(np.mean(noise ** 2)))

        if rms_n < 1e-9:
            return float("inf")
        if rms_s < 1e-9:
            return float("-inf")

        return float(20.0 * np.log10(rms_s / rms_n))

    def generate_report(self, output_path: str) -> str:
        """Write CSV report, passed-files JSON, and per-SNR statistics.

        Three files are written:

        - ``<output_path>`` — full CSV with all validation results.
        - ``<stem>_passed.json`` — array of paths that passed.
        - ``<stem>_snr_stats.json`` — pass-rate breakdown by SNR level.

        Args:
            output_path: Destination path for the main CSV report.

        Returns:
            Absolute path to the written CSV file.

        Raises:
            RuntimeError: If no results are available yet.
        """
        if not self._results:
            raise RuntimeError(
                "No results to report. Run validate_audio() or validate_batch() first."
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([r.to_dict() for r in self._results])
        df.to_csv(out, index=False, encoding="utf-8")
        self._logger.info(f"CSV report: {out}")

        # Passed-files JSON
        passed = [r.file_path for r in self._results if r.passed]
        json_path = out.with_name(out.stem + "_passed.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                    "total": len(self._results),
                    "passed": len(passed),
                    "files": passed,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        self._logger.info(f"Passed-files JSON: {json_path}")

        # Per-SNR statistics
        snr_stats = self._compute_snr_stats(df)
        stats_path = out.with_name(out.stem + "_snr_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(snr_stats, f, ensure_ascii=False, indent=2)
        self._logger.info(f"SNR stats JSON: {stats_path}")

        return str(out)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_checks(self, audio_path: str) -> ValidationResult2:
        """Execute all validation checks for a single synthesized file.

        Args:
            audio_path: Path to the synthesized audio file.

        Returns:
            Populated :class:`ValidationResult2`.
        """
        path = Path(audio_path)
        result = ValidationResult2(file_path=str(path.resolve()))
        reasons: list[str] = []

        # 1. Format
        suffix = path.suffix.lower()
        result.format_ok = suffix in _SUPPORTED_EXTENSIONS
        if not result.format_ok:
            reasons.append(f"unsupported format '{suffix}'")

        # 2. Load
        audio: np.ndarray | None = None
        sr: int | None = None
        try:
            audio, sr = load_audio(str(path), sr=None)
            result.file_intact = True
        except FileNotFoundError:
            reasons.append("file not found")
            result.failure_reasons = reasons
            return result
        except Exception as exc:
            reasons.append(f"corrupted or unreadable ({exc})")
            result.failure_reasons = reasons
            return result

        # 3. Sample rate
        result.sample_rate = sr
        result.sample_rate_ok = sr == self._EXPECTED_SR
        if not result.sample_rate_ok:
            reasons.append(f"sample rate {sr} Hz (expected {self._EXPECTED_SR})")

        # 4. Duration
        duration_ms = len(audio) / sr * 1_000
        result.duration_ms = duration_ms
        result.duration_ok = self._min_duration_ms <= duration_ms <= self._max_duration_ms
        if not result.duration_ok:
            reasons.append(
                f"duration {duration_ms:.0f} ms outside "
                f"[{self._min_duration_ms:.0f}, {self._max_duration_ms:.0f}] ms"
            )

        # 5. RMS energy
        rms = float(np.sqrt(np.mean(audio ** 2)))
        rms_db = float(20.0 * np.log10(rms + 1e-9))
        result.rms_energy_db = rms_db
        result.energy_ok = self._min_energy_db <= rms_db <= self._max_energy_db
        if not result.energy_ok:
            reasons.append(
                f"RMS {rms_db:.1f} dBFS outside "
                f"[{self._min_energy_db:.1f}, {self._max_energy_db:.1f}]"
            )

        # 6. Sidecar metadata
        meta = self._load_sidecar(path)
        if meta is None:
            result.metadata_exists = False
            reasons.append("sidecar .json not found")
        else:
            result.metadata_exists = True
            missing_keys = _REQUIRED_META_KEYS - set(meta.keys())
            if missing_keys:
                result.metadata_consistent = False
                reasons.append(f"sidecar missing keys: {missing_keys}")
            else:
                result.metadata_consistent = True
                result.target_snr_db = float(meta["target_snr_db"])

                # 7. Actual SNR measurement
                asmr_src = meta.get("source_asmr", "")
                noise_src = meta.get("source_noise", "")
                if Path(asmr_src).exists() and Path(noise_src).exists():
                    try:
                        measured = self.measure_actual_snr(asmr_src, noise_src)
                        result.measured_snr_db = round(measured, 2)
                        diff = abs(measured - result.target_snr_db)
                        result.snr_within_tolerance = diff <= _SNR_TOLERANCE_DB
                        if not result.snr_within_tolerance:
                            reasons.append(
                                f"SNR deviation {diff:.1f} dB "
                                f"(target={result.target_snr_db}, "
                                f"measured={measured:.1f})"
                            )
                    except Exception as exc:
                        self._logger.debug(f"SNR measurement failed for {path.name}: {exc}")
                else:
                    self._logger.debug(
                        f"Source files not available for SNR check: {path.name}"
                    )

        result.failure_reasons = reasons
        result.passed = len(reasons) == 0
        return result

    def _load_sidecar(self, wav_path: Path) -> dict[str, Any] | None:
        """Load the JSON sidecar adjacent to *wav_path*.

        Args:
            wav_path: Path to the synthesized WAV file.

        Returns:
            Parsed sidecar dictionary, or ``None`` if absent / unreadable.
        """
        sidecar = wav_path.with_suffix(".json")
        if not sidecar.exists():
            return None
        try:
            with open(sidecar, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self._logger.warning(f"Failed to parse sidecar {sidecar.name}: {exc}")
            return None

    def _compute_snr_stats(
        self, df: pd.DataFrame
    ) -> dict[str, Any]:
        """Compute pass-rate statistics grouped by target SNR level.

        Args:
            df: Full results DataFrame.

        Returns:
            Dictionary with overall stats and per-SNR breakdown.
        """
        stats: dict[str, Any] = {
            "total": len(df),
            "passed": int(df["passed"].sum()),
            "pass_rate": round(float(df["passed"].mean()), 4),
            "by_snr": {},
        }

        if "target_snr_db" not in df.columns:
            return stats

        for snr_val, group in df.groupby("target_snr_db"):
            if snr_val is None or (isinstance(snr_val, float) and np.isnan(snr_val)):
                continue
            stats["by_snr"][str(snr_val)] = {
                "total": len(group),
                "passed": int(group["passed"].sum()),
                "pass_rate": round(float(group["passed"].mean()), 4),
                "mean_measured_snr_db": (
                    round(float(group["measured_snr_db"].dropna().mean()), 2)
                    if "measured_snr_db" in group.columns
                    else None
                ),
            }

        return stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-processing quality validation for synthesized audio (Stage 5).",
    )
    parser.add_argument(
        "--config", required=True, help="Path to generation.yaml config file."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Root directory of synthesized files (default: synthesized/ from config).",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for output reports (default: logs/ from config).",
    )
    parser.add_argument("--log-file", default=None, help="Path to log file (optional).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for Stage 5 validation.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    args = _parse_args(argv)
    config = load_config(args.config)

    logs_dir: str = config.get("output_dirs", {}).get("logs", "./logs")
    log_file = args.log_file or str(Path(logs_dir) / "5_quality_validation_2.log")
    log = setup_logger("quality_validation_2", log_file=log_file)

    input_dir = args.input_dir or config["output_dirs"].get("synthesized", "./synthesized")
    validator = QualityValidator2(config=config, logger=log)
    df = validator.validate_batch(input_dir)

    if df.empty:
        log.warning("No files validated. Exiting.")
        sys.exit(0)

    report_dir = args.report_dir or logs_dir
    report_path = str(Path(report_dir) / "quality_validation_2_report.csv")
    validator.generate_report(report_path)

    n_pass = int(df["passed"].sum())
    print(f"\nResults: {n_pass}/{len(df)} synthesized files passed.")
    print(f"Report : {report_path}")


if __name__ == "__main__":
    main()
