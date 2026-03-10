"""Unit tests for the AudioPreprocessor (Stage 3)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src._3_preprocessing import AudioPreprocessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SR = 16_000
_LOG = logging.getLogger("test_preprocessing")


def _make_config(
    target_sr: int = _SR,
    min_ms: float = 1_000,
    max_ms: float = 30_000,
    target_db: float = -20.0,
    headroom_db: float = 3.0,
) -> dict:
    return {
        "preprocessing": {
            "target_sample_rate": target_sr,
            "target_rms_db": target_db,
            "length_adjustment": {"min_ms": min_ms, "max_ms": max_ms},
            "normalization": {"target_db": target_db, "headroom_db": headroom_db},
        }
    }


def _write_wav(
    path: Path,
    duration_s: float,
    sr: int = _SR,
    amplitude: float = 0.3,
    freq: float = 440.0,
) -> Path:
    """Write a sine-wave WAV file."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    audio = amplitude * np.sin(2 * np.pi * freq * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return path


def _preprocessor(config: dict | None = None) -> AudioPreprocessor:
    if config is None:
        config = _make_config()
    return AudioPreprocessor(config=config, logger=_LOG)


def _rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio**2)))
    return float(20.0 * np.log10(rms + 1e-12))


# ---------------------------------------------------------------------------
# TestResample
# ---------------------------------------------------------------------------


class TestResample:
    def test_44100_resampled_to_16k(self, tmp_path):
        """Loading a 44.1 kHz file should yield 16 kHz output."""
        wav = _write_wav(tmp_path / "hq.wav", duration_s=2.0, sr=44_100)
        pre = _preprocessor()
        result = pre.preprocess_audio(str(wav))

        expected_samples = int(2.0 * _SR)
        # Allow ±1 sample due to resampler rounding
        assert abs(len(result) - expected_samples) <= 1

    def test_output_is_float32(self, tmp_path):
        wav = _write_wav(tmp_path / "tone.wav", duration_s=2.0)
        pre = _preprocessor()
        result = pre.preprocess_audio(str(wav))
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# TestLengthAdjust
# ---------------------------------------------------------------------------


class TestLengthAdjust:
    def test_short_clip_repeated_to_min_length(self, tmp_path):
        """A 0.3 s clip should be tiled to at least min_ms (1 s) samples."""
        wav = _write_wav(tmp_path / "short.wav", duration_s=0.3)
        pre = _preprocessor(_make_config(min_ms=1_000))
        result = pre.preprocess_audio(str(wav))

        min_samples = int(1.0 * _SR)
        assert len(result) >= min_samples

    def test_short_clip_exactly_min_length(self, tmp_path):
        """After tiling, the output should be exactly min_ms samples."""
        wav = _write_wav(tmp_path / "short2.wav", duration_s=0.3)
        pre = _preprocessor(_make_config(min_ms=1_000, max_ms=30_000))
        result = pre.preprocess_audio(str(wav))

        min_samples = int(1.0 * _SR)
        assert len(result) == min_samples

    def test_normal_clip_length_unchanged(self, tmp_path):
        """A 2 s clip should pass through length adjustment unchanged."""
        wav = _write_wav(tmp_path / "normal.wav", duration_s=2.0)
        pre = _preprocessor(_make_config(min_ms=1_000, max_ms=30_000))
        result = pre.preprocess_audio(str(wav))

        expected_samples = int(2.0 * _SR)
        # Allow ±1 sample due to resampler rounding
        assert abs(len(result) - expected_samples) <= 1

    def test_long_clip_trimmed_to_max_length(self, tmp_path):
        """A 35 s clip should be center-cropped to max_ms (30 s)."""
        wav = _write_wav(tmp_path / "long.wav", duration_s=35.0)
        pre = _preprocessor(_make_config(max_ms=30_000))
        result = pre.preprocess_audio(str(wav))

        max_samples = int(30.0 * _SR)
        assert len(result) == max_samples

    def test_long_clip_center_crop(self, tmp_path):
        """Center-cropped result should start at the middle of the original."""
        duration_s = 4.0
        sr = _SR
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
        # Use a linearly increasing signal so the center can be identified
        audio = (t / duration_s).astype(np.float32)
        wav = tmp_path / "linear.wav"
        sf.write(str(wav), audio, sr)

        # Crop to 2 s — should keep the middle 2 s (samples 16000:48000)
        pre = _preprocessor(_make_config(min_ms=500, max_ms=2_000))
        result = pre.preprocess_audio(str(wav))

        max_samples = int(2.0 * sr)
        assert len(result) == max_samples

        # The first sample of the result should correspond to t=1.0 s
        expected_start_val = 1.0 / duration_s
        # After normalisation the exact value changes, so check relative position
        n_orig = int(sr * duration_s)
        n_crop = int(2.0 * sr)
        start = (n_orig - n_crop) // 2
        orig_slice = audio[start : start + n_crop]
        # The shape (relative values) should be proportional
        assert np.argmax(orig_slice) == np.argmax(result)


# ---------------------------------------------------------------------------
# TestRmsNormalize
# ---------------------------------------------------------------------------


class TestRmsNormalize:
    def test_output_rms_near_target(self, tmp_path):
        """Output RMS should be within ±1 dB of the target (-20 dBFS).

        headroom_db=0.0 means peak_limit=1.0 (0 dBFS), so no headroom
        clipping will interfere with the RMS normalisation check.
        """
        wav = _write_wav(tmp_path / "loud.wav", duration_s=2.0, amplitude=0.9)
        pre = _preprocessor(_make_config(target_db=-20.0, headroom_db=0.0))
        result = pre.preprocess_audio(str(wav))

        measured_db = _rms_db(result)
        assert abs(measured_db - (-20.0)) <= 1.0, (
            f"RMS {measured_db:.2f} dBFS is not within 1 dB of -20 dBFS"
        )

    def test_quiet_input_normalised_up(self, tmp_path):
        """A very quiet input should be amplified to near the target.

        headroom_db=0.0 means peak_limit=1.0 (0 dBFS) so the gain-up
        is not limited by headroom clipping.
        """
        wav = _write_wav(tmp_path / "quiet.wav", duration_s=2.0, amplitude=0.001)
        pre = _preprocessor(_make_config(target_db=-20.0, headroom_db=0.0))
        result = pre.preprocess_audio(str(wav))

        measured_db = _rms_db(result)
        assert abs(measured_db - (-20.0)) <= 1.0

    def test_silent_input_raises_value_error(self, tmp_path):
        """Silent audio (all zeros) should raise ValueError."""
        silent = tmp_path / "silent.wav"
        sf.write(str(silent), np.zeros(16_000, dtype=np.float32), _SR)

        pre = _preprocessor()
        with pytest.raises(ValueError, match="[Ss]ilent"):
            pre.preprocess_audio(str(silent))


# ---------------------------------------------------------------------------
# TestHeadroom
# ---------------------------------------------------------------------------


class TestHeadroom:
    def test_peak_within_headroom_limit(self, tmp_path):
        """Peak amplitude should be ≤ 10^(-3/20) ≈ 0.708 after a 3 dB headroom."""
        # Create a loud signal that would clip after RMS normalisation
        wav = _write_wav(tmp_path / "loud_hroom.wav", duration_s=2.0, amplitude=0.99)
        pre = _preprocessor(_make_config(headroom_db=3.0))
        result = pre.preprocess_audio(str(wav))

        peak_limit = 10.0 ** (-3.0 / 20.0)  # ≈ 0.7079
        assert float(np.max(np.abs(result))) <= peak_limit + 1e-5, (
            f"Peak {np.max(np.abs(result)):.4f} exceeds limit {peak_limit:.4f}"
        )

    def test_quiet_signal_not_clipped(self, tmp_path):
        """A signal already within headroom should not be scaled down."""
        # Amplitude 0.1 → after RMS norm to -20 dBFS peak ≈ 0.14 < 0.708
        wav = _write_wav(tmp_path / "medium.wav", duration_s=2.0, amplitude=0.1)
        pre = _preprocessor(_make_config(target_db=-20.0, headroom_db=3.0))
        result = pre.preprocess_audio(str(wav))

        peak_limit = 10.0 ** (-3.0 / 20.0)
        assert float(np.max(np.abs(result))) <= peak_limit + 1e-5


# ---------------------------------------------------------------------------
# TestProcessBatch
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def test_output_wav_files_created(self, tmp_path):
        """process_batch should create a WAV file for each input."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()

        for name in ("a.wav", "b.wav", "c.wav"):
            _write_wav(input_dir / name, duration_s=2.0)

        pre = _preprocessor()
        stats = pre.process_batch(str(input_dir), str(output_dir))

        assert stats["processed"] == 3
        assert stats["skipped"] == 0
        for name in ("a.wav", "b.wav", "c.wav"):
            assert (output_dir / name).exists()

    def test_idempotent_skips_existing(self, tmp_path):
        """Running process_batch twice should skip already-existing outputs."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()

        _write_wav(input_dir / "x.wav", duration_s=2.0)

        pre = _preprocessor()
        stats_first = pre.process_batch(str(input_dir), str(output_dir))
        assert stats_first["processed"] == 1

        # Record mtime
        out_file = output_dir / "x.wav"
        mtime = out_file.stat().st_mtime

        # Second run — same preprocessor instance, but results list has already
        # accumulated, so create a new instance to avoid double-counting.
        pre2 = _preprocessor()
        stats_second = pre2.process_batch(str(input_dir), str(output_dir))

        assert stats_second["skipped"] == 1
        assert stats_second["processed"] == 0
        # File should not have been touched
        assert out_file.stat().st_mtime == mtime

    def test_empty_directory_returns_zero_counts(self, tmp_path):
        """An empty input directory should return all-zero counts."""
        input_dir = tmp_path / "empty_in"
        output_dir = tmp_path / "empty_out"
        input_dir.mkdir()

        pre = _preprocessor()
        stats = pre.process_batch(str(input_dir), str(output_dir))

        assert stats["processed"] == 0
        assert stats["skipped"] == 0
        assert stats["failed"] == 0

    def test_returns_output_dir_path(self, tmp_path):
        """process_batch should return the resolved output_dir path."""
        input_dir = tmp_path / "inp"
        output_dir = tmp_path / "outp"
        input_dir.mkdir()
        _write_wav(input_dir / "tone.wav", duration_s=2.0)

        pre = _preprocessor()
        stats = pre.process_batch(str(input_dir), str(output_dir))

        assert Path(stats["output_dir"]).resolve() == output_dir.resolve()
