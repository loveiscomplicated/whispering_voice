"""Audio processing helpers for the STT data generation pipeline."""

import random
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to the target sample rate.

    Args:
        path: Path to the audio file (WAV, MP3, FLAC, etc.).
        sr: Target sample rate in Hz. Defaults to 16 000 (Whisper native).

    Returns:
        Tuple of (mono audio array with dtype float32, sample rate).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be decoded as audio.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, actual_sr = librosa.load(path, sr=sr, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not load audio from '{path}': {exc}") from exc

    return audio, actual_sr


def get_audio_info(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Compute basic statistics for an audio array.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate in Hz.

    Returns:
        Dictionary with the following keys:

        - ``duration_s`` (float): Duration in seconds.
        - ``duration_ms`` (float): Duration in milliseconds.
        - ``sample_rate`` (int): Sample rate in Hz.
        - ``num_samples`` (int): Total number of samples.
        - ``rms_energy_db`` (float): RMS energy in dBFS.
        - ``peak_amplitude`` (float): Peak absolute amplitude (0–1 range for
          normalized audio).
    """
    num_samples = len(audio)
    duration_s = num_samples / sr

    # RMS energy; guard against all-zero audio to avoid log(0)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_db = 20.0 * np.log10(rms + 1e-9)

    peak = float(np.max(np.abs(audio)))

    return {
        "duration_s": duration_s,
        "duration_ms": duration_s * 1000.0,
        "sample_rate": sr,
        "num_samples": num_samples,
        "rms_energy_db": float(rms_db),
        "peak_amplitude": peak,
    }


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize an audio array to the range [-1, 1].

    If the audio is silent (all zeros), it is returned unchanged to avoid
    division-by-zero.

    Args:
        audio: 1-D float32 audio array.

    Returns:
        Peak-normalized copy of *audio*.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        # Silent audio – return as-is
        return audio.copy()
    return audio / peak


def save_audio(audio: np.ndarray, path: str, sr: int = 16000) -> None:
    """Save an audio array to a WAV file.

    The parent directory is created automatically if it does not exist.

    Args:
        audio: 1-D float32 audio array.
        path: Destination file path (should end with ``.wav``).
        sr: Sample rate in Hz. Defaults to 16 000.

    Raises:
        OSError: If the file cannot be written.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(str(out_path), audio, sr, subtype="PCM_16")
    except Exception as exc:
        raise OSError(f"Failed to save audio to '{path}': {exc}") from exc


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (if available).

    Call this once at the start of the pipeline to ensure reproducible results
    across runs.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch is optional – skip gracefully if not installed
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
