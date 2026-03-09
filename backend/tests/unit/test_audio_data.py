"""
AudioData 클래스 단위 테스트
"""

import pytest
import numpy as np

from backend.stt_core.audio_input.audio_data import AudioData


def test_audio_data_creation():
    """AudioData 객체 생성 테스트"""
    audio = np.random.randn(16000)  # 1초 오디오

    data = AudioData(
        audio=audio, sample_rate=16000, channels=1, bit_depth=16, format="PCM"
    )

    assert data.sample_rate == 16000
    assert data.channels == 1
    assert abs(data.duration_ms - 1000.0) < 1.0  # 대략 1초


def test_audio_data_chunking():
    """오디오 청크 분할 테스트"""
    audio = np.random.randn(32000)  # 2초 오디오 (16kHz)
    data = AudioData(
        audio=audio, sample_rate=16000, channels=1, bit_depth=16, format="PCM"
    )

    chunks = list(data.get_chunks(8000))  # 0.5초씩 분할

    assert len(chunks) == 4


def test_audio_data_duration():
    """음성 길이 계산 테스트"""
    audio = np.random.randn(48000)  # 3초
    data = AudioData(
        audio=audio, sample_rate=16000, channels=1, bit_depth=16, format="PCM"
    )

    assert abs(data.get_duration_seconds() - 3.0) < 0.01
