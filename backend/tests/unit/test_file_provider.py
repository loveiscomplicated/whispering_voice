"""
FileAudioProvider 단위 테스트
"""

import os as _real_os
import sys

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# librosa is not installed in the test environment; stub it before the module
# import so the module-level `import librosa` in file_provider.py doesn't fail.
# All actual librosa calls are already mocked per-test via patch(_LOAD).
sys.modules.setdefault("librosa", MagicMock())

from backend.stt_core.input_providers.file_provider import FileAudioProvider
from backend.stt_core.audio_input.audio_data import AudioData, InvalidAudioFormatError

# 모듈 경로 상수
_MOD = "backend.stt_core.input_providers.file_provider"
# os 모듈 자체를 교체해야 pkg_resources 등 다른 모듈의 os.path 에 영향을 주지 않음
_OS = f"{_MOD}.os"
_LOAD = f"{_MOD}.librosa.load"


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_mono_audio(sr: int = 16000, duration: float = 1.0) -> np.ndarray:
    n = int(sr * duration)
    return np.random.randn(n).astype(np.float32)


def _make_stereo_audio(sr: int = 16000, duration: float = 0.5) -> np.ndarray:
    n = int(sr * duration)
    return np.random.randn(2, n).astype(np.float32)


def _connect(provider: FileAudioProvider, audio: np.ndarray, sr: int = 16000):
    """공통 connect 헬퍼 — file_provider.os 전체를 교체하여 글로벌 os 불변"""
    with patch(_OS) as mock_os, \
         patch(_LOAD, return_value=(audio, sr)):
        mock_os.path.exists.return_value = True
        provider.connect()


# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------


def test_init_stores_file_path():
    provider = FileAudioProvider("audio.wav")
    assert provider.file_path == "audio.wav"


def test_init_mono_true_by_default():
    provider = FileAudioProvider("audio.wav")
    assert provider.mono is True


def test_init_mono_false():
    provider = FileAudioProvider("audio.wav", mono=False)
    assert provider.mono is False


def test_init_not_connected():
    provider = FileAudioProvider("audio.wav")
    assert provider.is_connected() is False


def test_init_no_audio_buffer():
    provider = FileAudioProvider("audio.wav")
    assert provider._audio_buffer is None


# ---------------------------------------------------------------------------
# connect — 실패 케이스
# ---------------------------------------------------------------------------


def test_connect_file_not_found_raises():
    provider = FileAudioProvider("nonexistent.wav")
    with patch(_OS) as mock_os:
        mock_os.path.exists.return_value = False
        with pytest.raises(FileNotFoundError):
            provider.connect()


def test_connect_file_not_found_does_not_set_connected():
    provider = FileAudioProvider("nonexistent.wav")
    with patch(_OS) as mock_os:
        mock_os.path.exists.return_value = False
        with pytest.raises(FileNotFoundError):
            provider.connect()
    assert provider.is_connected() is False


def test_connect_librosa_error_raises_invalid_format():
    """librosa.load 실패 → InvalidAudioFormatError"""
    provider = FileAudioProvider("bad.wav")
    with patch(_OS) as mock_os, \
         patch(_LOAD, side_effect=Exception("decode error")):
        mock_os.path.exists.return_value = True
        with pytest.raises(InvalidAudioFormatError):
            provider.connect()


# ---------------------------------------------------------------------------
# connect — 성공 케이스
# ---------------------------------------------------------------------------


def test_connect_mono_success():
    audio = _make_mono_audio()
    provider = FileAudioProvider("test.wav")

    with patch(_OS) as mock_os, \
         patch(_LOAD, return_value=(audio, 16000)):
        mock_os.path.exists.return_value = True
        result = provider.connect()

    assert result is True
    assert provider.is_connected() is True
    assert provider._sample_rate == 16000
    assert provider._channels == 1


def test_connect_stereo_success():
    """2D 오디오 배열 → channels=2"""
    audio = _make_stereo_audio()
    provider = FileAudioProvider("stereo.wav", mono=False)

    with patch(_OS) as mock_os, \
         patch(_LOAD, return_value=(audio, 44100)):
        mock_os.path.exists.return_value = True
        provider.connect()

    assert provider._channels == 2
    assert provider._sample_rate == 44100


def test_connect_preserves_original_sample_rate():
    """sr=None이므로 원본 샘플레이트 유지"""
    audio = _make_mono_audio(sr=48000)
    provider = FileAudioProvider("test.wav")

    with patch(_OS) as mock_os, \
         patch(_LOAD, return_value=(audio, 48000)) as mock_load:
        mock_os.path.exists.return_value = True
        provider.connect()

    # librosa.load가 sr=None으로 호출됐는지 확인
    _, kwargs = mock_load.call_args
    assert kwargs.get("sr") is None


# ---------------------------------------------------------------------------
# disconnect
# ---------------------------------------------------------------------------


def test_disconnect_clears_buffer():
    audio = _make_mono_audio()
    provider = FileAudioProvider("test.wav")
    _connect(provider, audio)

    provider.disconnect()

    assert provider._audio_buffer is None
    assert provider.is_connected() is False


def test_disconnect_without_connect_does_not_raise():
    provider = FileAudioProvider("test.wav")
    provider.disconnect()  # should not raise


# ---------------------------------------------------------------------------
# receive_audio
# ---------------------------------------------------------------------------


def test_receive_audio_not_connected_raises():
    provider = FileAudioProvider("test.wav")
    with pytest.raises(ConnectionError):
        provider.receive_audio()


def test_receive_audio_returns_audio_data():
    audio = _make_mono_audio()
    provider = FileAudioProvider("test.wav")
    _connect(provider, audio)

    result = provider.receive_audio()

    assert isinstance(result, AudioData)
    assert result.sample_rate == 16000
    assert result.channels == 1
    assert result.format == "PCM"
    assert result.source == "file"


def test_receive_audio_contains_correct_samples():
    audio = _make_mono_audio()
    provider = FileAudioProvider("test.wav")
    _connect(provider, audio)

    result = provider.receive_audio()

    assert np.array_equal(result.audio, audio)


def test_receive_audio_metadata_has_file_info():
    audio = _make_mono_audio()
    provider = FileAudioProvider("/data/audio/test.wav")
    _connect(provider, audio)

    result = provider.receive_audio()

    assert result.metadata["file_path"] == "/data/audio/test.wav"
    assert result.metadata["file_name"] == "test.wav"


# ---------------------------------------------------------------------------
# get_source_info
# ---------------------------------------------------------------------------


def test_get_source_info_disconnected():
    provider = FileAudioProvider("/path/to/audio.wav")
    info = provider.get_source_info()

    assert info["type"] == "file"
    assert info["status"] == "disconnected"
    assert info["file_path"] == "/path/to/audio.wav"


def test_get_source_info_connected():
    audio = _make_mono_audio(sr=16000, duration=2.0)
    provider = FileAudioProvider("/path/to/audio.wav")
    _connect(provider, audio, sr=16000)

    info = provider.get_source_info()

    assert info["type"] == "file"
    assert info["status"] == "connected"
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert info["format"] == "PCM"
    assert "duration_ms" in info
    assert info["duration_ms"] == pytest.approx(2000, abs=10)
