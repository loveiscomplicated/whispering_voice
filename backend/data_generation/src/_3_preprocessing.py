"""Audio preprocessing — Stage 3.

Converts raw downloaded audio files into a consistent format suitable for
strict quality validation and STT/VAD processing:

- Resamples to the target sample rate (default 16 kHz).
- Adjusts length: repeats short clips, center-crops long clips.
- RMS-normalises to a target loudness level.
- Applies headroom limiting to avoid clipping.

Typical usage::

    python src/_3_preprocessing.py \\
        --input-dir passed_files \\
        --output-dir preprocessed \\
        --config config/generation.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.audio_processor import load_audio, save_audio  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class AudioPreprocessor:
    """Preprocess raw audio files to a consistent format for STT/VAD.

    Configuration is read from ``config["preprocessing"]``.  All parameters
    have sensible defaults that mirror the strict validator's expectations so
    that preprocessed files are virtually guaranteed to pass Stage 4.

    Args:
        config: Pipeline configuration dictionary (from ``load_config``).
        logger: Logger instance used for diagnostic messages.
    """

    def __init__(self, config: dict[str, Any], logger: Any) -> None:
        self._config = config
        self._logger = logger
        self._pre_cfg: dict[str, Any] = config.get("preprocessing", {})

        self._target_sr: int = int(self._pre_cfg.get("target_sample_rate", 16_000))

        len_adj = self._pre_cfg.get("length_adjustment", {})
        self._min_ms: float = float(len_adj.get("min_ms", 1_000))
        self._max_ms: float = float(len_adj.get("max_ms", 30_000))

        norm_cfg = self._pre_cfg.get("normalization", {})
        self._target_db: float = float(
            self._pre_cfg.get("target_rms_db", norm_cfg.get("target_db", -20.0))
        )
        self._headroom_db: float = float(norm_cfg.get("headroom_db", 3.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Preprocess a single audio file.

        Processing steps:

        1. Load and resample to ``target_sample_rate``.
        2. Adjust length:
           - Shorter than ``min_ms``: repeat via ``np.tile`` until long enough,
             then trim to exactly ``min_ms`` samples.
           - Between ``min_ms`` and ``max_ms``: pass through.
           - Longer than ``max_ms``: take the center ``max_ms`` samples.
        3. RMS-normalise to ``target_db`` dBFS.
        4. Apply headroom limit so that ``max|audio| ≤ 10^(-headroom_db/20)``.

        Args:
            audio_path: Path to the source audio file.

        Returns:
            Preprocessed 1-D float32 audio array at ``target_sample_rate``.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
            ValueError: If the audio is silent (RMS < 1e-9) after loading.
        """
        audio, _ = load_audio(audio_path, sr=self._target_sr)

        audio = self._adjust_length(audio)
        audio = self._rms_normalize(audio, audio_path)
        audio = self._apply_headroom(audio)

        return audio.astype(np.float32)

    def process_batch(self, input_dir: str, output_dir: str) -> dict[str, Any]:
        """Preprocess all audio files in *input_dir* and save to *output_dir*.

        Existing output files are skipped to make the operation idempotent.
        Progress is shown via a tqdm progress bar.

        Args:
            input_dir: Directory containing raw (passed) audio files.
            output_dir: Destination directory for preprocessed WAV files.

        Returns:
            Dictionary with keys:

            - ``"processed"`` (int): Number of successfully preprocessed files.
            - ``"skipped"`` (int): Number of files skipped (output already exists).
            - ``"failed"`` (int): Number of files that raised an exception.
            - ``"output_dir"`` (str): Absolute path to *output_dir*.
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)

        if not in_path.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        out_path.mkdir(parents=True, exist_ok=True)

        audio_files = sorted(
            p
            for p in in_path.iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )

        if not audio_files:
            self._logger.warning(f"No audio files found in: {input_dir}")
            return {
                "processed": 0,
                "skipped": 0,
                "failed": 0,
                "output_dir": str(out_path.resolve()),
            }

        self._logger.info(f"Preprocessing {len(audio_files)} file(s) from: {input_dir}")

        n_processed = 0
        n_skipped = 0
        n_failed = 0

        for audio_file in tqdm(audio_files, desc="Preprocessing", unit="file"):
            out_file = out_path / (audio_file.stem + ".wav")

            if out_file.exists():
                self._logger.debug(f"Skipping existing: {out_file.name}")
                n_skipped += 1
                continue

            try:
                processed = self.preprocess_audio(str(audio_file))
                save_audio(processed, str(out_file), sr=self._target_sr)
                n_processed += 1
                self._logger.debug(f"Preprocessed: {out_file.name}")
            except Exception as exc:
                self._logger.error(f"Failed to preprocess {audio_file.name}: {exc}")
                n_failed += 1

        self._logger.info(
            f"Batch done — processed: {n_processed}, "
            f"skipped: {n_skipped}, failed: {n_failed}"
        )
        return {
            "processed": n_processed,
            "skipped": n_skipped,
            "failed": n_failed,
            "output_dir": str(out_path.resolve()),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adjust_length(self, audio: np.ndarray) -> np.ndarray:
        """Adjust audio length to lie within [min_ms, max_ms].

        Args:
            audio: Input 1-D float32 audio array at ``target_sample_rate``.

        Returns:
            Length-adjusted audio array.
        """
        min_samples = int(self._min_ms / 1_000 * self._target_sr)
        max_samples = int(self._max_ms / 1_000 * self._target_sr)
        n = len(audio)

        if n < min_samples:
            # Repeat until long enough, then trim
            repeats = int(np.ceil(min_samples / max(n, 1)))
            audio = np.tile(audio, repeats)[:min_samples]
        elif n > max_samples:
            # Center-crop
            start = (n - max_samples) // 2
            audio = audio[start : start + max_samples]

        return audio

    def _rms_normalize(self, audio: np.ndarray, audio_path: str = "") -> np.ndarray:
        """RMS-normalise audio to ``target_db`` dBFS.

        Args:
            audio: Input 1-D float32 audio array.
            audio_path: Source path, used only for error messages.

        Returns:
            Normalised audio array.

        Raises:
            ValueError: If the audio RMS is below 1e-9 (effectively silent).
        """
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 1e-9:
            raise ValueError(
                f"Silent audio cannot be normalised: {audio_path or '(unknown)'}"
            )

        current_db = 20.0 * np.log10(rms)
        gain = 10.0 ** ((self._target_db - current_db) / 20.0)
        return (audio * gain).astype(np.float32)

    def _apply_headroom(self, audio: np.ndarray) -> np.ndarray:
        """Scale down audio if the peak exceeds the headroom limit.

        Args:
            audio: Input 1-D float32 audio array (already RMS-normalised).

        Returns:
            Headroom-limited audio array.
        """
        peak_limit = 10.0 ** (-self._headroom_db / 20.0)
        peak = float(np.max(np.abs(audio)))
        if peak > peak_limit:
            audio = (audio * peak_limit / peak).astype(np.float32)
        return audio


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio preprocessing — resample, normalise, and crop (Stage 3).",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing audio files that passed basic validation.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for preprocessed WAVs (default: preprocessed/ from config).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to generation.yaml config file.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (optional).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the preprocessing stage.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    args = _parse_args(argv)
    config = load_config(args.config)

    logs_dir: str = config.get("output_dirs", {}).get("logs", "./logs")
    log_file = args.log_file or str(Path(logs_dir) / "3_preprocessing.log")
    log = setup_logger("preprocessing", log_file=log_file)

    output_dir = args.output_dir or config.get("output_dirs", {}).get(
        "preprocessed", "./preprocessed"
    )

    preprocessor = AudioPreprocessor(config=config, logger=log)
    stats = preprocessor.process_batch(args.input_dir, output_dir)

    print(f"\nPreprocessing complete:")
    print(f"  Processed : {stats['processed']}")
    print(f"  Skipped   : {stats['skipped']}")
    print(f"  Failed    : {stats['failed']}")
    print(f"  Output dir: {stats['output_dir']}")


if __name__ == "__main__":
    main()
