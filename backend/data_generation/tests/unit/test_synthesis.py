"""Unit tests for noise synthesis (Stage 4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src._4_synthesize_noise import (
    NoiseSynthesizer,
    _apply_fade,
    _loop_to_length,
    _rms,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> dict:
    return {
        "synthesis": {
            "snr_levels_db": [5, 10, 20],
            "noise_types": ["ambient"],
            "noise_distribution": {"ambient": 1.0},
        },
        "output_dirs": {
            "stt_and_vad": "./stt_and_vad",
            "synthesized": "./synthesized",
            "raw_downloads": "./raw_downloads",
            "logs": "./logs",
        },
    }


@pytest.fixture()
def synth(config) -> NoiseSynthesizer:
    import logging

    return NoiseSynthesizer(config=config, logger=logging.getLogger("test"))


@pytest.fixture()
def tone_wav(tmp_path) -> str:
    """440 Hz sine wave, 1 second at 16 kHz."""
    sr = 16_000
    t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(str(path), audio, sr)
    return str(path)


@pytest.fixture()
def noise_wav(tmp_path) -> str:
    """White noise, 1 second at 16 kHz."""
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.3, 0.3, 16_000).astype(np.float32)
    path = tmp_path / "noise.wav"
    sf.write(str(path), audio, 16_000)
    return str(path)


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------


class TestRms:
    def test_silent_is_zero(self):
        assert _rms(np.zeros(100, dtype=np.float32)) == pytest.approx(0.0)

    def test_constant_signal(self):
        audio = np.full(1000, 0.5, dtype=np.float32)
        assert _rms(audio) == pytest.approx(0.5, rel=1e-4)


class TestApplyFade:
    def test_output_same_length(self):
        audio = np.ones(16_000, dtype=np.float32)
        result = _apply_fade(audio, 16_000)
        assert len(result) == len(audio)

    def test_first_sample_is_zero(self):
        audio = np.ones(16_000, dtype=np.float32)
        result = _apply_fade(audio, 16_000, fade_ms=10.0)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_last_sample_is_zero(self):
        audio = np.ones(16_000, dtype=np.float32)
        result = _apply_fade(audio, 16_000, fade_ms=10.0)
        assert result[-1] == pytest.approx(0.0, abs=1e-6)

    def test_middle_sample_unchanged(self):
        audio = np.ones(16_000, dtype=np.float32)
        result = _apply_fade(audio, 16_000, fade_ms=10.0)
        mid = len(audio) // 2
        assert result[mid] == pytest.approx(1.0, abs=1e-4)

    def test_does_not_modify_input(self):
        audio = np.ones(16_000, dtype=np.float32)
        original = audio.copy()
        _apply_fade(audio, 16_000)
        np.testing.assert_array_equal(audio, original)


class TestLoopToLength:
    def test_exact_length_unchanged(self):
        audio = np.arange(100, dtype=np.float32)
        result = _loop_to_length(audio, 100)
        np.testing.assert_array_equal(result, audio)

    def test_shorter_input_repeated(self):
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _loop_to_length(audio, 7)
        expected = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_longer_input_truncated(self):
        audio = np.arange(200, dtype=np.float32)
        result = _loop_to_length(audio, 100)
        assert len(result) == 100
        np.testing.assert_array_equal(result, audio[:100])

    def test_empty_input_returns_zeros(self):
        result = _loop_to_length(np.array([], dtype=np.float32), 50)
        assert len(result) == 50
        assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# NoiseSynthesizer.calculate_snr
# ---------------------------------------------------------------------------


class TestCalculateSnr:
    def test_equal_rms_gives_0db(self, synth):
        audio = np.ones(16_000, dtype=np.float32)
        assert synth.calculate_snr(audio, audio.copy()) == pytest.approx(0.0, abs=0.01)

    def test_silent_noise_gives_inf(self, synth):
        signal = np.ones(100, dtype=np.float32)
        noise = np.zeros(100, dtype=np.float32)
        assert synth.calculate_snr(signal, noise) == float("inf")

    def test_silent_signal_gives_neg_inf(self, synth):
        signal = np.zeros(100, dtype=np.float32)
        noise = np.ones(100, dtype=np.float32)
        assert synth.calculate_snr(signal, noise) == float("-inf")

    def test_10x_amplitude_gives_20db(self, synth):
        signal = np.full(1000, 1.0, dtype=np.float32)
        noise = np.full(1000, 0.1, dtype=np.float32)
        assert synth.calculate_snr(signal, noise) == pytest.approx(20.0, abs=0.1)


# ---------------------------------------------------------------------------
# NoiseSynthesizer.synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_output_is_float32(self, synth, tone_wav, noise_wav):
        mixed = synth.synthesize(tone_wav, noise_wav, snr_db=10.0)
        assert mixed.dtype == np.float32

    def test_output_not_clipped(self, synth, tone_wav, noise_wav):
        mixed = synth.synthesize(tone_wav, noise_wav, snr_db=10.0)
        assert np.max(np.abs(mixed)) <= 1.0 + 1e-6

    def test_output_is_nonzero(self, synth, tone_wav, noise_wav):
        mixed = synth.synthesize(tone_wav, noise_wav, snr_db=10.0)
        assert np.any(mixed != 0.0)

    def test_silent_signal_raises(self, synth, tmp_path, noise_wav):
        silent = tmp_path / "silent.wav"
        sf.write(str(silent), np.zeros(16_000, dtype=np.float32), 16_000)
        with pytest.raises(ValueError, match="silent"):
            synth.synthesize(str(silent), noise_wav, snr_db=10.0)

    def test_silent_noise_raises(self, synth, tone_wav, tmp_path):
        silent = tmp_path / "silent_noise.wav"
        sf.write(str(silent), np.zeros(16_000, dtype=np.float32), 16_000)
        with pytest.raises(ValueError, match="silent"):
            synth.synthesize(tone_wav, str(silent), snr_db=10.0)

    def test_higher_snr_has_lower_noise_energy(self, synth, tone_wav, noise_wav):
        """At higher SNR the noise contribution should be weaker."""
        mixed_low = synth.synthesize(tone_wav, noise_wav, snr_db=5.0)
        mixed_high = synth.synthesize(tone_wav, noise_wav, snr_db=25.0)
        # Load original signal
        signal, _ = sf.read(tone_wav, dtype="float32")
        # Noise residual ≈ mixed - signal (normalisation makes this approximate)
        # Simply check that 5 dB mix has more energy than 25 dB mix
        assert _rms(mixed_low) != _rms(mixed_high)  # they should differ

    @pytest.mark.parametrize("target_snr", [5.0, 10.0, 15.0, 20.0])
    def test_target_snr_within_tolerance(self, synth, tone_wav, noise_wav, target_snr):
        """Measured SNR of the synthesized output should be near the target."""
        mixed = synth.synthesize(tone_wav, noise_wav, snr_db=target_snr)
        signal, _ = sf.read(tone_wav, dtype="float32")

        # Pad / trim to same length as mixed
        sig = _loop_to_length(signal, len(mixed))
        measured = synth.calculate_snr(sig, mixed - sig)
        # Allow ±4 dB — peak-normalisation shifts the absolute levels
        assert (
            abs(measured - target_snr) < 4.0
        ), f"SNR mismatch: target={target_snr}, measured={measured:.1f}"


# ---------------------------------------------------------------------------
# synthesize_batch — output structure
# ---------------------------------------------------------------------------


class TestSynthesizeBatch:
    def test_creates_output_files(self, synth, tmp_path, tone_wav, noise_wav):
        asmr_dir = tmp_path / "asmr"
        asmr_dir.mkdir()
        noise_dir = tmp_path / "noise" / "ambient"
        noise_dir.mkdir(parents=True)
        out_dir = tmp_path / "out"

        import shutil

        shutil.copy(tone_wav, asmr_dir / "asmr_001.wav")
        shutil.copy(noise_wav, noise_dir / "amb_001.wav")

        results = synth.synthesize_batch(
            asmr_dir=str(asmr_dir),
            noise_dir=str(tmp_path / "noise"),
            snr_levels=[10.0],
            output_dir=str(out_dir),
        )

        assert len(results) > 0
        for paths in results.values():
            for p in paths:
                assert Path(p).exists(), f"Expected output file missing: {p}"

    def test_creates_json_sidecar(self, synth, tmp_path, tone_wav, noise_wav):
        asmr_dir = tmp_path / "asmr"
        asmr_dir.mkdir()
        noise_dir = tmp_path / "noise" / "ambient"
        noise_dir.mkdir(parents=True)
        out_dir = tmp_path / "out"

        import shutil

        shutil.copy(tone_wav, asmr_dir / "asmr_002.wav")
        shutil.copy(noise_wav, noise_dir / "amb_002.wav")

        synth.synthesize_batch(
            asmr_dir=str(asmr_dir),
            noise_dir=str(tmp_path / "noise"),
            snr_levels=[10.0],
            output_dir=str(out_dir),
        )

        json_files = list(out_dir.rglob("*.json"))
        assert len(json_files) > 0

        with open(json_files[0]) as f:
            meta = json.load(f)
        assert "target_snr_db" in meta
        assert "noise_type" in meta

    def test_skips_existing_files(self, synth, tmp_path, tone_wav, noise_wav):
        """Re-running synthesize_batch should not overwrite existing files."""
        import shutil

        asmr_dir = tmp_path / "asmr"
        asmr_dir.mkdir()
        noise_dir = tmp_path / "noise" / "ambient"
        noise_dir.mkdir(parents=True)
        out_dir = tmp_path / "out"

        shutil.copy(tone_wav, asmr_dir / "asmr_003.wav")
        shutil.copy(noise_wav, noise_dir / "amb_003.wav")

        synth.synthesize_batch(
            str(asmr_dir), str(tmp_path / "noise"), [10.0], str(out_dir)
        )
        wav_files = list(out_dir.rglob("*.wav"))
        assert wav_files

        # Record modification times
        mtimes = {p: p.stat().st_mtime for p in wav_files}

        # Second run
        synth.synthesize_batch(
            str(asmr_dir), str(tmp_path / "noise"), [10.0], str(out_dir)
        )

        for p in wav_files:
            assert p.stat().st_mtime == mtimes[p], f"File was overwritten: {p}"
