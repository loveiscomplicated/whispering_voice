"""Unit tests for STT processing (Stage 3 — Whisper side)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> dict:
    return {
        "stt": {
            "model": "whisper-base",
            "language": "ko",
            "min_confidence": 0.85,
            "device": "cpu",
        },
        "vad": {
            "model": "pyannote/segmentation",
            "threshold": 0.5,
            "min_speech_duration_ms": 300,
        },
        "output_dirs": {"stt_and_vad": "./stt_and_vad", "logs": "./logs"},
    }


@pytest.fixture()
def mock_logger() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def processor(config, mock_logger):
    """STTAndVADProcessor with ML backends fully mocked."""
    with (
        patch("src._3_run_stt_and_vad._load_whisper") as mock_wh,
        patch("src._3_run_stt_and_vad._load_pyannote") as mock_pa,
    ):
        from src._3_run_stt_and_vad import STTAndVADProcessor  # noqa: PLC0415

        proc = STTAndVADProcessor(config=config, logger=mock_logger)
        proc._whisper_model = MagicMock()
        proc._vad_pipeline = MagicMock()
        mock_wh.return_value = proc._whisper_model
        mock_pa.return_value = proc._vad_pipeline
        return proc


@pytest.fixture()
def sample_wav(tmp_path) -> str:
    """Write a 1-second 16 kHz silent WAV and return the path."""
    path = tmp_path / "sample.wav"
    audio = np.zeros(16_000, dtype=np.float32)
    sf.write(str(path), audio, 16_000)
    return str(path)


# ---------------------------------------------------------------------------
# run_stt
# ---------------------------------------------------------------------------

class TestRunSTT:
    def _make_whisper_result(self, text: str, avg_logprob: float = -0.1):
        return {
            "text": text,
            "language": "ko",
            "segments": [{"avg_logprob": avg_logprob, "text": text}],
        }

    def test_returns_required_keys(self, processor, sample_wav):
        processor._whisper_model.transcribe.return_value = (
            self._make_whisper_result("안녕하세요")
        )
        result = processor.run_stt(sample_wav)

        required = {
            "transcript", "confidence", "language",
            "model_used", "model_version", "processing_time_ms",
        }
        assert required <= set(result.keys())

    def test_transcript_text_stripped(self, processor, sample_wav):
        processor._whisper_model.transcribe.return_value = (
            self._make_whisper_result("  hello world  ")
        )
        result = processor.run_stt(sample_wav)
        assert result["transcript"] == "hello world"

    def test_confidence_in_range(self, processor, sample_wav):
        processor._whisper_model.transcribe.return_value = (
            self._make_whisper_result("test", avg_logprob=-0.3)
        )
        result = processor.run_stt(sample_wav)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_high_logprob_gives_high_confidence(self, processor, sample_wav):
        processor._whisper_model.transcribe.return_value = (
            self._make_whisper_result("good", avg_logprob=-0.05)
        )
        result = processor.run_stt(sample_wav)
        assert result["confidence"] > 0.9

    def test_low_logprob_gives_low_confidence(self, processor, sample_wav):
        processor._whisper_model.transcribe.return_value = (
            self._make_whisper_result("bad", avg_logprob=-2.5)
        )
        result = processor.run_stt(sample_wav)
        assert result["confidence"] < 0.15

    def test_empty_segments_yields_zero_confidence(self, processor, sample_wav):
        processor._whisper_model.transcribe.return_value = {
            "text": "",
            "language": "ko",
            "segments": [],
        }
        result = processor.run_stt(sample_wav)
        assert result["confidence"] == 0.0

    def test_model_used_reflects_config(self, processor, sample_wav):
        processor._whisper_model.transcribe.return_value = (
            self._make_whisper_result("x")
        )
        result = processor.run_stt(sample_wav)
        assert "whisper-base" in result["model_used"]

    def test_processing_time_non_negative(self, processor, sample_wav):
        processor._whisper_model.transcribe.return_value = (
            self._make_whisper_result("timing")
        )
        result = processor.run_stt(sample_wav)
        assert result["processing_time_ms"] >= 0


# ---------------------------------------------------------------------------
# generate_metadata schema
# ---------------------------------------------------------------------------

class TestGenerateMetadataSchema:
    """Verify the output of generate_metadata matches the CLAUDE.md schema."""

    def _stt_result(self):
        return {
            "transcript": "테스트",
            "confidence": 0.92,
            "language": "ko",
            "model_used": "whisper-base",
            "model_version": "v20230314",
            "processing_time_ms": 120,
        }

    def _vad_result(self):
        return {
            "segments": [
                {
                    "segment_id": 0,
                    "start_ms": 100.0,
                    "end_ms": 2500.0,
                    "duration_ms": 2400.0,
                    "confidence": 0.98,
                }
            ],
            "total_speech_duration_ms": 2400.0,
            "total_silence_duration_ms": 600.0,
            "speech_ratio": 0.8,
        }

    def test_top_level_keys_present(self, processor, sample_wav):
        meta = processor.generate_metadata(
            "asmr_001", self._stt_result(), self._vad_result(), sample_wav
        )
        for key in ("audio_id", "processing_pipeline_version", "stt_result",
                    "vad_result", "audio_characteristics"):
            assert key in meta, f"Missing top-level key: {key}"

    def test_stt_result_schema(self, processor, sample_wav):
        meta = processor.generate_metadata(
            "asmr_001", self._stt_result(), self._vad_result(), sample_wav
        )
        stt = meta["stt_result"]
        for key in ("transcript", "language", "confidence_score", "model_used",
                    "model_version"):
            assert key in stt, f"Missing stt_result key: {key}"

    def test_vad_result_schema(self, processor, sample_wav):
        meta = processor.generate_metadata(
            "asmr_001", self._stt_result(), self._vad_result(), sample_wav
        )
        vad = meta["vad_result"]
        for key in ("segments", "total_speech_duration_ms",
                    "total_silence_duration_ms", "speech_ratio"):
            assert key in vad, f"Missing vad_result key: {key}"

    def test_audio_characteristics_schema(self, processor, sample_wav):
        meta = processor.generate_metadata(
            "asmr_001", self._stt_result(), self._vad_result(), sample_wav
        )
        ac = meta["audio_characteristics"]
        for key in ("format", "sample_rate", "channels", "duration_ms",
                    "rms_energy_db", "peak_amplitude"):
            assert key in ac, f"Missing audio_characteristics key: {key}"

    def test_audio_id_matches(self, processor, sample_wav):
        meta = processor.generate_metadata(
            "custom_id", self._stt_result(), self._vad_result(), sample_wav
        )
        assert meta["audio_id"] == "custom_id"

    def test_low_confidence_flag_set(self, processor, sample_wav):
        processor._whisper_model.transcribe.return_value = {
            "text": "low",
            "language": "ko",
            "segments": [{"avg_logprob": -3.0, "text": "low"}],
        }
        processor._vad_pipeline.return_value = MagicMock(
            itertracks=MagicMock(return_value=[])
        )
        meta = processor.process_audio(sample_wav)
        assert meta.get("low_confidence") is True
