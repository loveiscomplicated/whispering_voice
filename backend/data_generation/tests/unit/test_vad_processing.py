"""Unit tests for VAD processing (Stage 3 — Pyannote side)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(start: float, end: float, confidence: float = 1.0):
    """Create a mock Pyannote segment object."""
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.confidence = confidence
    return seg


def _diarization_from_segments(segments):
    """Build a fake diarization object whose itertracks returns *segments*."""
    diar = MagicMock()
    # Each entry is (segment, track, label)
    diar.itertracks.return_value = [
        (seg, f"track_{i}", "SPEAKER_00")
        for i, seg in enumerate(segments)
    ]
    return diar


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> dict:
    return {
        "stt": {"model": "whisper-base", "language": "ko",
                "min_confidence": 0.85, "device": "cpu"},
        "vad": {"model": "pyannote/segmentation",
                "threshold": 0.5, "min_speech_duration_ms": 300},
        "output_dirs": {"stt_and_vad": "./stt_and_vad", "logs": "./logs"},
    }


@pytest.fixture()
def mock_logger() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def processor(config, mock_logger):
    with (
        patch("src._3_run_stt_and_vad._load_whisper"),
        patch("src._3_run_stt_and_vad._load_pyannote"),
    ):
        from src._3_run_stt_and_vad import STTAndVADProcessor  # noqa: PLC0415

        proc = STTAndVADProcessor(config=config, logger=mock_logger)
        proc._whisper_model = MagicMock()
        proc._vad_pipeline = MagicMock()
        return proc


@pytest.fixture()
def sample_wav(tmp_path) -> str:
    path = tmp_path / "sample.wav"
    sf.write(str(path), np.zeros(16_000, dtype=np.float32), 16_000)
    return str(path)


# ---------------------------------------------------------------------------
# run_vad — segment extraction
# ---------------------------------------------------------------------------

class TestRunVAD:
    def test_returns_required_keys(self, processor, sample_wav):
        processor._vad_pipeline.return_value = _diarization_from_segments([
            _make_segment(0.1, 2.5)
        ])
        result = processor.run_vad(sample_wav)

        for key in ("segments", "total_speech_duration_ms",
                    "total_silence_duration_ms", "speech_ratio"):
            assert key in result

    def test_single_segment_extracted(self, processor, sample_wav):
        processor._vad_pipeline.return_value = _diarization_from_segments([
            _make_segment(0.0, 1.0)
        ])
        result = processor.run_vad(sample_wav)
        assert len(result["segments"]) == 1

    def test_segment_fields_present(self, processor, sample_wav):
        processor._vad_pipeline.return_value = _diarization_from_segments([
            _make_segment(0.5, 2.0)
        ])
        result = processor.run_vad(sample_wav)
        seg = result["segments"][0]
        for key in ("segment_id", "start_ms", "end_ms", "duration_ms", "confidence"):
            assert key in seg, f"Missing segment key: {key}"

    def test_start_end_ms_conversion(self, processor, sample_wav):
        processor._vad_pipeline.return_value = _diarization_from_segments([
            _make_segment(0.1, 2.5)
        ])
        result = processor.run_vad(sample_wav)
        seg = result["segments"][0]
        assert abs(seg["start_ms"] - 100.0) < 1
        assert abs(seg["end_ms"] - 2500.0) < 1

    def test_short_segments_filtered(self, processor, sample_wav):
        """Segments shorter than min_speech_duration_ms (300 ms) are dropped."""
        processor._vad_pipeline.return_value = _diarization_from_segments([
            _make_segment(0.0, 0.1),   # 100 ms — too short
            _make_segment(0.5, 1.0),   # 500 ms — OK
        ])
        result = processor.run_vad(sample_wav)
        assert len(result["segments"]) == 1
        assert result["segments"][0]["duration_ms"] == pytest.approx(500.0, abs=1)

    def test_low_confidence_segments_filtered(self, processor, sample_wav):
        """Segments below threshold (0.5) are dropped."""
        processor._vad_pipeline.return_value = _diarization_from_segments([
            _make_segment(0.0, 1.0, confidence=0.3),  # below threshold
            _make_segment(1.0, 2.0, confidence=0.9),  # above threshold
        ])
        result = processor.run_vad(sample_wav)
        assert len(result["segments"]) == 1
        assert result["segments"][0]["confidence"] == pytest.approx(0.9)

    def test_no_segments_returns_zeros(self, processor, sample_wav):
        processor._vad_pipeline.return_value = _diarization_from_segments([])
        result = processor.run_vad(sample_wav)
        assert result["segments"] == []
        assert result["total_speech_duration_ms"] == 0.0
        assert result["speech_ratio"] == 0.0

    def test_speech_ratio_range(self, processor, sample_wav):
        processor._vad_pipeline.return_value = _diarization_from_segments([
            _make_segment(0.0, 0.5),
        ])
        result = processor.run_vad(sample_wav)
        assert 0.0 <= result["speech_ratio"] <= 1.0

    def test_total_speech_duration_sum(self, processor, sample_wav):
        processor._vad_pipeline.return_value = _diarization_from_segments([
            _make_segment(0.0, 0.5),   # 500 ms
            _make_segment(0.6, 1.1),   # 500 ms
        ])
        result = processor.run_vad(sample_wav)
        assert result["total_speech_duration_ms"] == pytest.approx(1000.0, abs=2)

    def test_silence_duration_non_negative(self, processor, sample_wav):
        processor._vad_pipeline.return_value = _diarization_from_segments([
            _make_segment(0.0, 0.5),
        ])
        result = processor.run_vad(sample_wav)
        assert result["total_silence_duration_ms"] >= 0.0

    def test_multiple_segments_ids_sequential(self, processor, sample_wav):
        segments = [_make_segment(i * 0.5, i * 0.5 + 0.4) for i in range(4)]
        processor._vad_pipeline.return_value = _diarization_from_segments(segments)
        result = processor.run_vad(sample_wav)
        ids = [s["segment_id"] for s in result["segments"]]
        assert ids == sorted(ids)
