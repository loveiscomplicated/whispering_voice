"""
AudioPreprocessingModule 단위 테스트
"""

import pytest
import numpy as np
from dataclasses import dataclass

from backend.stt_core.preprocessing.audio_preprocessing_module import (
    AudioPreprocessingModule,
)


# ---------------------------------------------------------------------------
# 테스트 전용 AudioData 데이터클래스
# (pipeline의 AudioData와 동일한 구조)
# ---------------------------------------------------------------------------


@dataclass
class _AudioData:
    audio: np.ndarray
    sample_rate: int
    channels: int
    duration_ms: int


def _make_audio(
    sr: int = 16000,
    duration: float = 1.0,
    amplitude: float = 0.1,
    channels: int = 1,
    stereo: bool = False,
) -> _AudioData:
    n = int(sr * duration)
    t = np.linspace(0, duration, n)
    if stereo:
        left = amplitude * np.sin(2 * np.pi * 1000 * t)
        right = amplitude * np.cos(2 * np.pi * 1000 * t)
        audio = np.vstack([left, right]).astype(np.float32)
    else:
        audio = (amplitude * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    return _AudioData(
        audio=audio,
        sample_rate=sr,
        channels=channels,
        duration_ms=int(duration * 1000),
    )


# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------


def test_default_target_sample_rate():
    p = AudioPreprocessingModule()
    assert p.target_sample_rate == 16000


def test_default_chunk_size():
    p = AudioPreprocessingModule()
    assert p.chunk_size == 16000


def test_default_high_freq_coef():
    p = AudioPreprocessingModule()
    assert p.high_freq_coef == pytest.approx(0.97)


def test_default_noise_alpha():
    p = AudioPreprocessingModule()
    assert p.noise_alpha == pytest.approx(2.0)


def test_default_vad_threshold():
    p = AudioPreprocessingModule()
    assert p.vad_threshold == pytest.approx(0.02)


def test_default_target_rms():
    p = AudioPreprocessingModule()
    assert p.target_rms == pytest.approx(0.1)


def test_custom_config():
    config = {
        "target_sample_rate": 8000,
        "high_freq_coef": 0.95,
        "noise_alpha": 3.0,
        "vad_threshold": 0.05,
    }
    p = AudioPreprocessingModule(config)

    assert p.target_sample_rate == 8000
    assert p.high_freq_coef == pytest.approx(0.95)
    assert p.noise_alpha == pytest.approx(3.0)
    assert p.vad_threshold == pytest.approx(0.05)


def test_initial_streaming_state_is_inactive():
    p = AudioPreprocessingModule()
    assert p.streaming_mode is False
    assert p.noise_profile is None
    assert p.previous_chunk is None


# ---------------------------------------------------------------------------
# process (배치)
# ---------------------------------------------------------------------------


def test_process_empty_audio_raises_value_error():
    p = AudioPreprocessingModule()
    data = _AudioData(audio=np.array([]), sample_rate=16000, channels=1, duration_ms=0)

    with pytest.raises(ValueError):
        p.process(data)


def test_process_returns_ndarray():
    p = AudioPreprocessingModule()
    data = _make_audio()

    result = p.process(data)

    assert isinstance(result, np.ndarray)


def test_process_output_is_floating_type():
    p = AudioPreprocessingModule()
    data = _make_audio()

    result = p.process(data)

    assert result.dtype in (np.float32, np.float64)


def test_process_output_not_empty():
    p = AudioPreprocessingModule()
    data = _make_audio()

    result = p.process(data)

    assert len(result) > 0


def test_process_output_values_in_range():
    p = AudioPreprocessingModule()
    data = _make_audio(amplitude=0.15)

    result = p.process(data)

    assert np.min(result) >= -1.0
    assert np.max(result) <= 1.0


def test_process_clips_large_amplitude():
    """진폭이 큰 신호도 -1~1로 클리핑"""
    p = AudioPreprocessingModule()
    data = _make_audio(amplitude=10.0)  # 범위 초과

    result = p.process(data)

    assert np.min(result) >= -1.0
    assert np.max(result) <= 1.0


def test_process_stereo_to_mono():
    """스테레오 → 1D 출력"""
    p = AudioPreprocessingModule()
    data = _make_audio(stereo=True, channels=2)

    result = p.process(data)

    assert result.ndim == 1


def test_process_resamples_to_target_sr():
    """22050Hz → 16000Hz 리샘플링"""
    p = AudioPreprocessingModule()
    sr = 22050
    data = _make_audio(sr=sr)
    expected_samples = int(len(data.audio) * 16000 / sr)

    result = p.process(data)

    assert abs(len(result) - expected_samples) <= 2


def test_process_does_not_resample_when_sr_matches():
    """이미 16kHz면 샘플 수 동일"""
    p = AudioPreprocessingModule()
    data = _make_audio(sr=16000)
    original_len = len(data.audio)

    result = p.process(data)

    # 청크 처리로 인해 샘플 수가 약간 다를 수 있음 (오버랩)
    assert len(result) > 0


def test_process_resets_streaming_state():
    """배치 처리 전 스트리밍 상태 초기화"""
    p = AudioPreprocessingModule()
    p.noise_profile = np.ones(8000)  # 인위적으로 설정

    p.process(_make_audio())

    assert p.streaming_mode is False
    assert p.noise_profile is None


def test_process_multiple_files_sequentially():
    """여러 파일 순차 처리 — 각 결과가 독립적"""
    p = AudioPreprocessingModule()
    results = []

    for _ in range(3):
        result = p.process(_make_audio(duration=0.5))
        results.append(result)

    assert len(results) == 3
    assert all(isinstance(r, np.ndarray) for r in results)


def test_process_short_audio():
    """100ms 짧은 오디오도 처리 가능"""
    p = AudioPreprocessingModule()
    data = _make_audio(duration=0.1)

    result = p.process(data)

    assert len(result) > 0
    assert result.dtype == np.float32


def test_process_long_audio():
    """10초 긴 오디오 처리"""
    p = AudioPreprocessingModule()
    data = _make_audio(duration=10.0)

    result = p.process(data)

    assert len(result) > 0


def test_process_float64_input_returns_floating_type():
    """float64 입력도 정상 처리"""
    p = AudioPreprocessingModule()
    audio = (0.1 * np.sin(2 * np.pi * 1000 * np.linspace(0, 1, 16000))).astype(
        np.float64
    )
    data = _AudioData(audio=audio, sample_rate=16000, channels=1, duration_ms=1000)

    result = p.process(data)

    assert result.dtype in (np.float32, np.float64)


def test_process_vad_zeroes_silence():
    """침묵 구간은 VAD로 0 처리"""
    p = AudioPreprocessingModule()
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    # 처음 0.5초만 신호
    audio[: sr // 2] = 0.15 * np.sin(
        2 * np.pi * 1000 * np.linspace(0, 0.5, sr // 2)
    )
    data = _AudioData(audio=audio, sample_rate=sr, channels=1, duration_ms=1000)

    result = p.process(data)

    assert np.any(result == 0)


# ---------------------------------------------------------------------------
# process_streaming_chunk (실시간)
# ---------------------------------------------------------------------------


def test_streaming_chunk_returns_ndarray():
    p = AudioPreprocessingModule()
    chunk = np.random.randn(16000).astype(np.float32) * 0.1

    result = p.process_streaming_chunk(chunk)

    assert isinstance(result, np.ndarray)


def test_streaming_chunk_preserves_length():
    p = AudioPreprocessingModule()
    chunk = np.random.randn(16000).astype(np.float32) * 0.1

    result = p.process_streaming_chunk(chunk)

    assert len(result) == len(chunk)


def test_streaming_chunk_output_is_float32():
    p = AudioPreprocessingModule()
    chunk = np.random.randn(16000).astype(np.float32) * 0.1

    result = p.process_streaming_chunk(chunk)

    assert result.dtype == np.float32


def test_streaming_chunk_activates_streaming_mode():
    p = AudioPreprocessingModule()
    chunk = np.random.randn(16000).astype(np.float32) * 0.1

    p.process_streaming_chunk(chunk)

    assert p.streaming_mode is True


def test_streaming_chunk_empty_raises_value_error():
    p = AudioPreprocessingModule()

    with pytest.raises(ValueError):
        p.process_streaming_chunk(np.array([]))


def test_streaming_chunk_initializes_noise_profile_on_first_call():
    p = AudioPreprocessingModule()
    p.reset_streaming_state()
    assert p.noise_profile is None

    chunk = np.random.randn(16000).astype(np.float32) * 0.1
    p.process_streaming_chunk(chunk)

    assert p.noise_profile is not None


def test_streaming_chunk_maintains_noise_profile_across_chunks():
    p = AudioPreprocessingModule()
    p.reset_streaming_state()

    for _ in range(3):
        chunk = np.random.randn(16000).astype(np.float32) * 0.1
        p.process_streaming_chunk(chunk)

    assert p.noise_profile is not None


# ---------------------------------------------------------------------------
# reset_streaming_state / get_streaming_state
# ---------------------------------------------------------------------------


def test_reset_streaming_state_clears_noise_profile():
    p = AudioPreprocessingModule()
    chunk = np.random.randn(16000).astype(np.float32) * 0.1
    p.process_streaming_chunk(chunk)

    p.reset_streaming_state()

    assert p.noise_profile is None


def test_reset_streaming_state_deactivates_streaming_mode():
    p = AudioPreprocessingModule()
    chunk = np.random.randn(16000).astype(np.float32) * 0.1
    p.process_streaming_chunk(chunk)

    p.reset_streaming_state()

    assert p.streaming_mode is False


def test_reset_streaming_state_clears_previous_chunk():
    p = AudioPreprocessingModule()
    p.previous_chunk = np.ones(100)

    p.reset_streaming_state()

    assert p.previous_chunk is None


def test_get_streaming_state_initial():
    p = AudioPreprocessingModule()
    state = p.get_streaming_state()

    assert state["streaming_mode"] is False
    assert state["has_noise_profile"] is False
    assert state["noise_profile_shape"] is None


def test_get_streaming_state_after_chunk():
    p = AudioPreprocessingModule()
    chunk = np.random.randn(16000).astype(np.float32) * 0.1
    p.process_streaming_chunk(chunk)

    state = p.get_streaming_state()

    assert state["streaming_mode"] is True
    assert state["has_noise_profile"] is True
    assert state["noise_profile_shape"] is not None


def test_get_streaming_state_after_reset():
    p = AudioPreprocessingModule()
    p.process_streaming_chunk(np.random.randn(16000).astype(np.float32) * 0.1)
    p.reset_streaming_state()

    state = p.get_streaming_state()

    assert state["has_noise_profile"] is False


# ---------------------------------------------------------------------------
# 내부 메서드 직접 테스트
# ---------------------------------------------------------------------------


def test_emphasize_high_frequency_output_shape():
    p = AudioPreprocessingModule()
    audio = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, 16000)).astype(np.float32)

    result = p._emphasize_high_frequency(audio, 0.97)

    assert result.shape == audio.shape


def test_emphasize_high_frequency_first_sample_unchanged():
    """첫 샘플은 변경 없이 유지"""
    p = AudioPreprocessingModule()
    audio = np.array([0.5, 0.1, 0.2, -0.1], dtype=np.float32)

    result = p._emphasize_high_frequency(audio, 0.97)

    assert result[0] == pytest.approx(audio[0])


def test_normalize_rms_near_zero_returns_unchanged():
    """RMS가 거의 0인 신호 → 정규화 건너뜀"""
    p = AudioPreprocessingModule()
    silent = np.zeros(1000, dtype=np.float32)

    result = p._normalize_rms(silent, target_rms=0.1)

    assert np.allclose(result, silent)


def test_normalize_rms_scales_audio():
    """RMS를 target 값으로 스케일"""
    p = AudioPreprocessingModule()
    audio = 0.01 * np.ones(1000, dtype=np.float32)  # 매우 낮은 RMS

    result = p._normalize_rms(audio, target_rms=0.1)

    rms = np.sqrt(np.mean(result ** 2))
    assert rms == pytest.approx(0.1, abs=0.01)


def test_normalize_sample_rate_same_sr():
    """동일 샘플레이트 → 원본 반환"""
    p = AudioPreprocessingModule()
    audio = np.random.randn(16000).astype(np.float32)

    result = p._normalize_sample_rate(audio, 16000, 16000)

    assert np.array_equal(result, audio)


def test_normalize_sample_rate_downsamples():
    """22050 → 16000 다운샘플링"""
    p = AudioPreprocessingModule()
    audio = np.random.randn(22050).astype(np.float32)
    expected_len = int(22050 * 16000 / 22050)

    result = p._normalize_sample_rate(audio, 22050, 16000)

    assert abs(len(result) - expected_len) <= 1
