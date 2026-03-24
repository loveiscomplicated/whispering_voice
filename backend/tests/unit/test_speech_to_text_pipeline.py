"""
SpeechToTextPipeline 단위 테스트
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from backend.stt_core.pipeline.speech_to_text_pipeline import (
    SpeechToTextPipeline,
    AudioData,
    STTResult,
    DummySTTModel,
    STTModel,
)


# ---------------------------------------------------------------------------
# 헬퍼 팩토리
# ---------------------------------------------------------------------------


def _make_audio_data(
    sr: int = 16000, duration_ms: int = 1000, channels: int = 1
) -> AudioData:
    n = int(sr * duration_ms / 1000)
    audio = np.random.randn(n).astype(np.float32)
    return AudioData(audio=audio, sample_rate=sr, channels=channels, duration_ms=duration_ms)


def _make_preprocessor(output: np.ndarray = None) -> MagicMock:
    preprocessor = MagicMock()
    if output is None:
        output = np.random.randn(16000).astype(np.float32)
    preprocessor.process.return_value = output
    preprocessor.process_streaming_chunk.return_value = output
    preprocessor.reset_streaming_state.return_value = None
    return preprocessor


def _make_stt_model(text: str = "안녕하세요", confidence: float = 0.9) -> MagicMock:
    model = MagicMock(spec=STTModel)
    model.recognize.return_value = STTResult(
        text=text, confidence=confidence, duration_ms=1000.0, language="ko"
    )
    return model


def _make_pipeline(text: str = "테스트", confidence: float = 0.9, **cfg) -> tuple:
    config = {"verbose": False, **cfg}
    preprocessor = _make_preprocessor()
    model = _make_stt_model(text=text, confidence=confidence)
    pipeline = SpeechToTextPipeline(preprocessor, model, config)
    return pipeline, preprocessor, model


# ---------------------------------------------------------------------------
# DummySTTModel
# ---------------------------------------------------------------------------


def test_dummy_stt_model_returns_stt_result():
    model = DummySTTModel()
    audio = np.random.randn(16000).astype(np.float32)
    result = model.recognize(audio)

    assert isinstance(result, STTResult)
    assert result.language == "ko"
    assert 0.0 <= result.confidence <= 1.0


def test_dummy_stt_model_result_text_is_from_list():
    model = DummySTTModel()
    audio = np.random.randn(16000).astype(np.float32)
    result = model.recognize(audio)

    assert result.text in model.test_results


def test_dummy_stt_model_cycles_through_results():
    model = DummySTTModel()
    audio = np.random.randn(16000).astype(np.float32)
    n = len(model.test_results)
    texts = [model.recognize(audio).text for _ in range(n + 1)]

    # 순환: 첫 번째와 n+1번째가 같음
    assert texts[0] == texts[n]


def test_dummy_stt_model_increments_call_count():
    model = DummySTTModel()
    audio = np.random.randn(16000).astype(np.float32)

    assert model.call_count == 0
    model.recognize(audio)
    assert model.call_count == 1
    model.recognize(audio)
    assert model.call_count == 2


def test_dummy_stt_model_confidence_depends_on_audio_length():
    model = DummySTTModel()
    short = np.random.randn(100).astype(np.float32)
    long_ = np.random.randn(100000).astype(np.float32)

    r_short = model.recognize(short)
    r_long = model.recognize(long_)

    assert r_long.confidence >= r_short.confidence


# ---------------------------------------------------------------------------
# AudioData / STTResult 데이터클래스
# ---------------------------------------------------------------------------


def test_audio_data_fields():
    data = _make_audio_data()
    assert data.sample_rate == 16000
    assert data.channels == 1
    assert data.duration_ms == 1000


def test_stt_result_fields():
    result = STTResult(text="hello", confidence=0.85, duration_ms=500.0)
    assert result.text == "hello"
    assert result.confidence == 0.85
    assert result.duration_ms == 500.0
    assert result.language == "ko"  # default


def test_stt_result_custom_language():
    result = STTResult(text="hello", confidence=0.9, duration_ms=100.0, language="en")
    assert result.language == "en"


# ---------------------------------------------------------------------------
# 파이프라인 초기화
# ---------------------------------------------------------------------------


def test_pipeline_default_config():
    preprocessor = _make_preprocessor()
    model = _make_stt_model()
    pipeline = SpeechToTextPipeline(preprocessor, model)

    assert pipeline.streaming_mode is False
    assert pipeline.save_preprocessed is False
    assert pipeline.verbose is True
    assert pipeline.total_processed == 0
    assert pipeline.total_recognized == 0
    assert pipeline.avg_confidence == 0.0
    assert pipeline.preprocessed_audios == []


def test_pipeline_custom_config():
    preprocessor = _make_preprocessor()
    model = _make_stt_model()
    config = {"streaming_mode": True, "save_preprocessed": True, "verbose": False}
    pipeline = SpeechToTextPipeline(preprocessor, model, config)

    assert pipeline.streaming_mode is True
    assert pipeline.save_preprocessed is True
    assert pipeline.verbose is False


# ---------------------------------------------------------------------------
# recognize_file
# ---------------------------------------------------------------------------


def test_recognize_file_returns_stt_result():
    pipeline, _, _ = _make_pipeline(text="안녕하세요", confidence=0.95)
    result = pipeline.recognize_file(_make_audio_data())

    assert isinstance(result, STTResult)
    assert result.text == "안녕하세요"
    assert result.confidence == 0.95


def test_recognize_file_calls_preprocessor_with_audio_data():
    pipeline, preprocessor, _ = _make_pipeline()
    audio_data = _make_audio_data()

    pipeline.recognize_file(audio_data)

    preprocessor.process.assert_called_once_with(audio_data)


def test_recognize_file_calls_stt_model_with_preprocessed():
    processed = np.ones(16000, dtype=np.float32)
    pipeline, preprocessor, model = _make_pipeline()
    preprocessor.process.return_value = processed

    pipeline.recognize_file(_make_audio_data())

    model.recognize.assert_called_once_with(processed)


def test_recognize_file_updates_total_processed():
    pipeline, _, _ = _make_pipeline()

    pipeline.recognize_file(_make_audio_data())
    pipeline.recognize_file(_make_audio_data())

    assert pipeline.total_processed == 2


def test_recognize_file_updates_total_recognized():
    pipeline, _, _ = _make_pipeline()

    pipeline.recognize_file(_make_audio_data())

    assert pipeline.total_recognized == 1


def test_recognize_file_updates_avg_confidence_single():
    pipeline, _, _ = _make_pipeline(confidence=0.8)

    pipeline.recognize_file(_make_audio_data())

    assert pipeline.avg_confidence == pytest.approx(0.8, abs=1e-6)


def test_recognize_file_updates_avg_confidence_multiple():
    preprocessor = _make_preprocessor()
    model = MagicMock(spec=STTModel)
    model.recognize.side_effect = [
        STTResult(text="a", confidence=0.8, duration_ms=1000.0, language="ko"),
        STTResult(text="b", confidence=0.6, duration_ms=1000.0, language="ko"),
    ]
    pipeline = SpeechToTextPipeline(preprocessor, model, {"verbose": False})

    pipeline.recognize_file(_make_audio_data())
    pipeline.recognize_file(_make_audio_data())

    # alpha=0.5 → avg = (1-0.5)*0.8 + 0.5*0.6 = 0.7
    assert pipeline.avg_confidence == pytest.approx(0.7, abs=1e-6)


def test_recognize_file_saves_preprocessed_when_flag_true():
    pipeline, _, _ = _make_pipeline(save_preprocessed=True)

    pipeline.recognize_file(_make_audio_data())
    pipeline.recognize_file(_make_audio_data())

    assert len(pipeline.preprocessed_audios) == 2


def test_recognize_file_does_not_save_preprocessed_by_default():
    pipeline, _, _ = _make_pipeline()

    pipeline.recognize_file(_make_audio_data())

    assert len(pipeline.preprocessed_audios) == 0


# ---------------------------------------------------------------------------
# 실시간 처리 (streaming)
# ---------------------------------------------------------------------------


def test_start_streaming_resets_preprocessor_state():
    pipeline, preprocessor, _ = _make_pipeline()

    pipeline.start_streaming()

    preprocessor.reset_streaming_state.assert_called_once()


def test_recognize_chunk_with_non_empty_text_returns_result():
    pipeline, _, _ = _make_pipeline(text="실시간 음성")
    chunk = np.random.randn(16000).astype(np.float32)

    result = pipeline.recognize_chunk(chunk)

    assert result is not None
    assert result.text == "실시간 음성"


def test_recognize_chunk_with_empty_text_returns_none():
    preprocessor = _make_preprocessor()
    model = MagicMock(spec=STTModel)
    model.recognize.return_value = STTResult(
        text="", confidence=0.0, duration_ms=500.0, language="ko"
    )
    pipeline = SpeechToTextPipeline(preprocessor, model, {"verbose": False})

    result = pipeline.recognize_chunk(np.random.randn(16000).astype(np.float32))

    assert result is None


def test_recognize_chunk_increments_total_processed():
    pipeline, _, _ = _make_pipeline(text="텍스트")
    chunk = np.random.randn(16000).astype(np.float32)

    pipeline.recognize_chunk(chunk)

    assert pipeline.total_processed == 1


def test_recognize_chunk_increments_total_recognized_only_for_non_empty():
    preprocessor = _make_preprocessor()
    model = MagicMock(spec=STTModel)
    model.recognize.side_effect = [
        STTResult(text="있음", confidence=0.9, duration_ms=1000.0, language="ko"),
        STTResult(text="", confidence=0.0, duration_ms=1000.0, language="ko"),
    ]
    pipeline = SpeechToTextPipeline(preprocessor, model, {"verbose": False})

    chunk = np.random.randn(16000).astype(np.float32)
    pipeline.recognize_chunk(chunk)
    pipeline.recognize_chunk(chunk)

    assert pipeline.total_processed == 2
    assert pipeline.total_recognized == 1


def test_recognize_chunk_saves_preprocessed_when_flag_true():
    pipeline, _, _ = _make_pipeline(text="텍스트", save_preprocessed=True)
    chunk = np.random.randn(16000).astype(np.float32)

    pipeline.recognize_chunk(chunk)

    assert len(pipeline.preprocessed_audios) == 1


def test_stop_streaming_does_not_raise():
    pipeline, _, _ = _make_pipeline()
    pipeline.stop_streaming()  # should not raise


# ---------------------------------------------------------------------------
# get_statistics
# ---------------------------------------------------------------------------


def test_get_statistics_initial():
    pipeline, _, _ = _make_pipeline()
    stats = pipeline.get_statistics()

    assert stats["total_processed"] == 0
    assert stats["total_recognized"] == 0
    assert stats["avg_confidence"] == 0.0
    assert stats["recognition_rate"] == 0


def test_get_statistics_after_recognition():
    pipeline, _, _ = _make_pipeline(confidence=0.9)

    pipeline.recognize_file(_make_audio_data())
    stats = pipeline.get_statistics()

    assert stats["total_processed"] == 1
    assert stats["total_recognized"] == 1
    assert stats["recognition_rate"] == 1.0


def test_get_statistics_recognition_rate_zero_when_no_processed():
    pipeline, _, _ = _make_pipeline()
    stats = pipeline.get_statistics()

    assert stats["recognition_rate"] == 0


# ---------------------------------------------------------------------------
# print_statistics
# ---------------------------------------------------------------------------


def test_print_statistics_outputs_to_stdout(capsys):
    pipeline, _, _ = _make_pipeline()

    pipeline.print_statistics()

    captured = capsys.readouterr()
    assert len(captured.out) > 0


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_clears_statistics():
    pipeline, _, _ = _make_pipeline()
    pipeline.recognize_file(_make_audio_data())

    pipeline.reset()

    assert pipeline.total_processed == 0
    assert pipeline.total_recognized == 0
    assert pipeline.avg_confidence == 0.0


def test_reset_clears_preprocessed_audios():
    pipeline, _, _ = _make_pipeline(save_preprocessed=True)
    pipeline.recognize_file(_make_audio_data())

    pipeline.reset()

    assert pipeline.preprocessed_audios == []


def test_reset_calls_preprocessor_reset():
    pipeline, preprocessor, _ = _make_pipeline()

    pipeline.reset()

    preprocessor.reset_streaming_state.assert_called_once()
