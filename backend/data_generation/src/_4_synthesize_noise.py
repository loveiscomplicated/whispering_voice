"""Background noise synthesis — Stage 4.

Mixes clean ASMR audio with background noise at multiple SNR levels.
Each SNR-level variant is saved as a WAV file with a JSON sidecar containing
synthesis parameters.

Typical usage::

    python src/4_synthesize_noise.py --config config/generation.yaml

    # Custom SNR levels
    python src/4_synthesize_noise.py \\
        --config config/generation.yaml \\
        --snr-levels 5 10 20
"""

from __future__ import annotations

import argparse
import json
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

_SAMPLE_RATE = 16_000
_FADE_DURATION_MS = 50
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------


def _rms(audio: np.ndarray) -> float:
    """Compute root-mean-square energy of an audio array.

    Args:
        audio: 1-D float32 audio array.

    Returns:
        RMS value (always non-negative).
    """
    return float(np.sqrt(np.mean(audio**2)))


def _apply_fade(audio: np.ndarray, sr: int, fade_ms: float = 50.0) -> np.ndarray:
    """Apply a linear fade-in and fade-out to prevent click artefacts.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate in Hz.
        fade_ms: Fade duration in milliseconds for each end.

    Returns:
        Copy of *audio* with fades applied.
    """
    fade_samples = min(int(sr * fade_ms / 1_000), len(audio) // 2)
    if fade_samples == 0:
        return audio.copy()

    result = audio.copy()
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    result[:fade_samples] *= ramp  # fade-in
    result[-fade_samples:] *= ramp[::-1]  # fade-out
    return result


def _loop_to_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Repeat *audio* until it is at least *target_length* samples long,
    then truncate to exactly *target_length*.

    Args:
        audio: 1-D float32 audio array.
        target_length: Desired number of samples.

    Returns:
        Array of exactly *target_length* samples.
    """
    if len(audio) == 0:
        return np.zeros(target_length, dtype=np.float32)
    repeats = int(np.ceil(target_length / len(audio)))
    return np.tile(audio, repeats)[:target_length]


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class NoiseSynthesizer:
    """Mix clean audio with background noise at target SNR levels.

    The synthesis procedure:

    1. Load both signals at 16 kHz mono.
    2. Loop the shorter signal to match the longer one.
    3. Scale noise so that ``SNR = 10·log10(P_signal / P_noise)`` equals the
       requested target.
    4. Sum signals and apply a 50 ms linear fade-in / fade-out.
    5. Peak-normalise to prevent clipping.

    Args:
        config: Pipeline configuration dictionary (from ``load_config``).
        logger: Logger instance used for diagnostic messages.
    """

    def __init__(self, config: dict[str, Any], logger: Any) -> None:
        self._config = config
        self._logger = logger

        synth_cfg = config.get("synthesis", {})
        self._snr_levels: list[float] = [
            float(v) for v in synth_cfg.get("snr_levels_db", [5, 10, 15, 20, 25])
        ]
        self._noise_types: list[str] = synth_cfg.get(
            "noise_types", ["ambient", "traffic", "office"]
        )
        self._output_base = Path(
            config.get("output_dirs", {}).get("synthesized", "./synthesized")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        asmr_path: str,
        noise_path: str,
        snr_db: float,
    ) -> np.ndarray:
        """Mix a clean signal with a noise signal at a target SNR.

        Args:
            asmr_path: Path to the clean (ASMR) audio file.
            noise_path: Path to the background noise audio file.
            snr_db: Desired signal-to-noise ratio in dB.

        Returns:
            Mixed 1-D float32 audio array at 16 kHz, peak-normalised.

        Raises:
            ValueError: If either signal is silent (RMS ≈ 0).
            FileNotFoundError: If either file does not exist.
        """
        signal, _ = load_audio(asmr_path, sr=_SAMPLE_RATE)
        noise, _ = load_audio(noise_path, sr=_SAMPLE_RATE)

        # Match lengths by looping the shorter array
        target_len = max(len(signal), len(noise))
        signal = _loop_to_length(signal, target_len)
        noise = _loop_to_length(noise, target_len)

        # Apply fade to both before mixing to reduce edge artefacts
        signal = _apply_fade(signal, _SAMPLE_RATE, _FADE_DURATION_MS)
        noise = _apply_fade(noise, _SAMPLE_RATE, _FADE_DURATION_MS)

        rms_signal = _rms(signal)
        rms_noise = _rms(noise)

        if rms_signal < 1e-9:
            raise ValueError(f"Clean signal is silent: {asmr_path}")
        if rms_noise < 1e-9:
            raise ValueError(f"Noise signal is silent: {noise_path}")

        # Scale noise: SNR(dB) = 20*log10(rms_s / rms_n_scaled)
        # → rms_n_scaled = rms_s / 10^(snr_db/20)
        scale = rms_signal / (rms_noise * 10 ** (snr_db / 20.0))
        mixed = signal + noise * scale

        # Peak-normalise to [-1, 1] to prevent clipping
        peak = np.max(np.abs(mixed))
        if peak > 1e-9:
            mixed = mixed / peak

        return mixed.astype(np.float32)

    def synthesize_batch(
        self,
        asmr_dir: str,
        noise_dir: str,
        snr_levels: list[float] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, list[str]]:
        """Synthesize all (ASMR × noise × SNR) combinations.

        Output files are organised as::

            output_dir/
                snr_05/ambient/<audio_id>_snr05_ambient.wav
                snr_10/traffic/<audio_id>_snr10_traffic.wav
                …

        Args:
            asmr_dir: Directory containing clean ASMR WAV files.
            noise_dir: Directory containing noise sub-folders named by type
                (``ambient/``, ``traffic/``, ``office/``).
            snr_levels: SNR values in dB. Defaults to ``synthesis.snr_levels_db``
                from config.
            output_dir: Root output directory. Defaults to
                ``output_dirs.synthesized`` from config.

        Returns:
            Dictionary mapping ``"<audio_id>_snr<N>_<noise_type>"`` keys to
            lists of generated file paths.
        """
        levels = snr_levels if snr_levels is not None else self._snr_levels
        out_root = Path(output_dir) if output_dir else self._output_base

        asmr_files = sorted(
            p
            for p in Path(asmr_dir).iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
        if not asmr_files:
            self._logger.warning(f"No audio files found in: {asmr_dir}")
            return {}

        # Collect noise files per type
        noise_files: dict[str, list[Path]] = {}
        for noise_type in self._noise_types:
            type_dir = Path(noise_dir) / noise_type
            if type_dir.is_dir():
                files = sorted(
                    p
                    for p in type_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
                )
                if files:
                    noise_files[noise_type] = files
                else:
                    self._logger.warning(f"No noise files in: {type_dir}")
            else:
                self._logger.warning(f"Noise type dir not found: {type_dir}")

        if not noise_files:
            self._logger.error("No usable noise files found. Aborting batch.")
            return {}

        total = (
            len(asmr_files) * sum(len(v) for v in noise_files.values()) * len(levels)
        )
        self._logger.info(
            f"Synthesizing {total} combinations "
            f"({len(asmr_files)} ASMR × noise × {len(levels)} SNR levels)"
        )

        results: dict[str, list[str]] = {}
        pbar = tqdm(total=total, desc="Synthesizing", unit="file")

        for asmr_path in asmr_files:
            audio_id = asmr_path.stem
            for noise_type, n_files in noise_files.items():
                for noise_path in n_files:
                    for snr in levels:
                        label = f"{audio_id}_snr{int(snr):02d}_{noise_type}"
                        snr_tag = f"snr_{int(snr):02d}"
                        out_dir = out_root / snr_tag / noise_type
                        out_path = out_dir / f"{label}.wav"

                        if out_path.exists():
                            self._logger.debug(f"Skipping existing: {out_path.name}")
                            results.setdefault(label, []).append(str(out_path))
                            pbar.update(1)
                            continue

                        try:
                            mixed = self.synthesize(
                                str(asmr_path), str(noise_path), snr
                            )
                            save_audio(mixed, str(out_path), sr=_SAMPLE_RATE)
                            self._save_synthesis_metadata(
                                out_path,
                                audio_id=audio_id,
                                noise_path=str(noise_path),
                                noise_type=noise_type,
                                snr_db=snr,
                                mixed=mixed,
                            )
                            results.setdefault(label, []).append(str(out_path))
                        except Exception as exc:
                            self._logger.error(f"Failed {label}: {exc}")
                        finally:
                            pbar.update(1)

        pbar.close()
        self._logger.info(
            f"Batch done — {sum(len(v) for v in results.values())} files written"
        )
        return results

    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Compute the actual SNR between two audio arrays.

        Both arrays must be the same length. The SNR is computed as:
        ``SNR(dB) = 20 * log10(rms_signal / rms_noise)``.

        Args:
            signal: Clean signal array.
            noise: Noise-only array (same length as *signal*).

        Returns:
            SNR in dB. Returns ``float('inf')`` when noise is silent, and
            ``float('-inf')`` when the signal is silent.
        """
        rms_s = _rms(signal)
        rms_n = _rms(noise)

        if rms_n < 1e-9:
            return float("inf")
        if rms_s < 1e-9:
            return float("-inf")

        return float(20.0 * np.log10(rms_s / rms_n))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_synthesis_metadata(
        self,
        wav_path: Path,
        audio_id: str,
        noise_path: str,
        noise_type: str,
        snr_db: float,
        mixed: np.ndarray,
    ) -> None:
        """Write a JSON sidecar next to a synthesized WAV file.

        Args:
            wav_path: Path to the synthesized WAV file.
            audio_id: Source ASMR audio identifier.
            noise_path: Path to the noise file used.
            noise_type: Noise category label.
            snr_db: Target SNR in dB used for synthesis.
            mixed: The synthesized audio array (for computing stats).
        """
        rms = _rms(mixed)
        rms_db = float(20.0 * np.log10(rms + 1e-9))
        peak = float(np.max(np.abs(mixed)))
        duration_ms = len(mixed) / _SAMPLE_RATE * 1_000

        meta = {
            "audio_id": audio_id,
            "source_asmr": str(wav_path.parent / f"{audio_id}.wav"),
            "source_noise": noise_path,
            "noise_type": noise_type,
            "target_snr_db": snr_db,
            "fade_duration_ms": _FADE_DURATION_MS,
            "audio_characteristics": {
                "format": "wav",
                "sample_rate": _SAMPLE_RATE,
                "channels": 1,
                "duration_ms": round(duration_ms, 2),
                "rms_energy_db": round(rms_db, 2),
                "peak_amplitude": round(peak, 4),
            },
        }

        meta_path = wav_path.with_suffix(".json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            self._logger.error(f"Failed to save synthesis metadata: {exc}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mix ASMR audio with background noise at multiple SNR levels (Stage 4).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to generation.yaml config file.",
    )
    parser.add_argument(
        "--asmr-dir",
        default=None,
        help="Directory containing clean ASMR audio (default: stt_and_vad/ from config).",
    )
    parser.add_argument(
        "--noise-dir",
        default=None,
        help="Directory containing noise sub-folders (default: raw_downloads/noise/).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--snr-levels",
        nargs="+",
        type=float,
        default=None,
        metavar="DB",
        help="SNR levels in dB (overrides config). Example: --snr-levels 5 10 20",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (optional).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the noise synthesis stage.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    args = _parse_args(argv)
    config = load_config(args.config)

    logs_dir: str = config.get("output_dirs", {}).get("logs", "./logs")
    log_file = args.log_file or str(Path(logs_dir) / "4_synthesize_noise.log")
    log = setup_logger("synthesize_noise", log_file=log_file)

    asmr_dir = args.asmr_dir or config["output_dirs"].get(
        "stt_and_vad", "./stt_and_vad"
    )
    noise_dir = args.noise_dir or str(
        Path(config["output_dirs"].get("raw_downloads", "./raw_downloads")) / "noise"
    )

    synthesizer = NoiseSynthesizer(config=config, logger=log)
    results = synthesizer.synthesize_batch(
        asmr_dir=asmr_dir,
        noise_dir=noise_dir,
        snr_levels=args.snr_levels,
        output_dir=args.output_dir,
    )

    total_files = sum(len(v) for v in results.values())
    out_root = args.output_dir or config["output_dirs"].get(
        "synthesized", "./synthesized"
    )
    print(f"\nSynthesized: {total_files} file(s)")
    print(f"Output dir : {out_root}")


if __name__ == "__main__":
    main()
