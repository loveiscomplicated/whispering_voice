"""
AudioInputModule 단위 테스트
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from backend.stt_core.audio_input.audio_input_module import AudioInputModule
from backend.stt_core.audio_input.audio_data import (
    AudioData,
    InvalidAudioFormatError,
    CorruptedAudioDataError,
)
from backend.stt_core.input_providers.base import IAudioInputProvider


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_audio_data(
    sample_rate: int = 16000,
    channels: int = 1,
    fmt: str = "PCM",
    n_samples: int = 16000,
) -> AudioData:
    audio = np.random.randn(n_samples).astype(np.float32)
    return AudioData(
        audio=audio,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=16,
        format=fmt,
    )


def _make_mock_provider(
    connected: bool = True,
    audio_data: AudioData = None,
    source_type: str = "mock",
) -> MagicMock:
    provider = MagicMock(spec=IAudioInputProvider)
    provider.connect.return_value = True
    provider.is_connected.return_value = connected
    provider.get_source_info.return_value = {"type": source_type}
    if audio_data is not None:
        provider.receive_audio.return_value = audio_data
    return provider


# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------


def test_init_stores_provider():
    provider = _make_mock_provider()
    module = AudioInputModule(provider)
    assert module.provider is provider


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


def test_connect_delegates_to_provider():
    provider = _make_mock_provider()
    module = AudioInputModule(provider)

    result = module.connect()

    provider.connect.assert_called_once()
    assert result is True


def test_connect_propagates_exception():
    provider = _make_mock_provider()
    provider.connect.side_effect = RuntimeError("connect failed")
    module = AudioInputModule(provider)

    with pytest.raises(RuntimeError):
        module.connect()


def test_connect_returns_false_when_provider_returns_false():
    provider = _make_mock_provider()
    provider.connect.return_value = False
    module = AudioInputModule(provider)

    result = module.connect()

    assert result is False


# ---------------------------------------------------------------------------
# disconnect
# ---------------------------------------------------------------------------


def test_disconnect_delegates_to_provider():
    provider = _make_mock_provider()
    module = AudioInputModule(provider)

    module.disconnect()

    provider.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# is_connected
# ---------------------------------------------------------------------------


def test_is_connected_true():
    provider = _make_mock_provider(connected=True)
    module = AudioInputModule(provider)
    assert module.is_connected() is True


def test_is_connected_false():
    provider = _make_mock_provider(connected=False)
    module = AudioInputModule(provider)
    assert module.is_connected() is False


# ---------------------------------------------------------------------------
# receive_and_parse — 성공
# ---------------------------------------------------------------------------


def test_receive_and_parse_returns_audio_data():
    audio_data = _make_audio_data()
    provider = _make_mock_provider(audio_data=audio_data)
    module = AudioInputModule(provider)

    result = module.receive_and_parse()

    assert result is audio_data
    provider.receive_audio.assert_called_once()


@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100, 48000])
def test_receive_and_parse_all_supported_sample_rates(sample_rate):
    audio_data = _make_audio_data(sample_rate=sample_rate)
    provider = _make_mock_provider(audio_data=audio_data)
    module = AudioInputModule(provider)

    result = module.receive_and_parse()

    assert result.sample_rate == sample_rate


@pytest.mark.parametrize("fmt", ["PCM", "WAV", "MP3"])
def test_receive_and_parse_all_supported_formats(fmt):
    audio_data = _make_audio_data(fmt=fmt)
    provider = _make_mock_provider(audio_data=audio_data)
    module = AudioInputModule(provider)

    result = module.receive_and_parse()

    assert result is audio_data


def test_receive_and_parse_stereo_audio_valid():
    audio_data = _make_audio_data(channels=2)
    provider = _make_mock_provider(audio_data=audio_data)
    module = AudioInputModule(provider)

    result = module.receive_and_parse()

    assert result.channels == 2


def test_receive_and_parse_silent_audio_does_not_raise():
    """무음(전부 0) → 경고이지만 예외 아님"""
    audio = np.zeros(16000, dtype=np.float32)
    audio_data = AudioData(
        audio=audio, sample_rate=16000, channels=1, bit_depth=16, format="PCM"
    )
    provider = _make_mock_provider(audio_data=audio_data)
    module = AudioInputModule(provider)

    result = module.receive_and_parse()

    assert result is audio_data


# ---------------------------------------------------------------------------
# receive_and_parse — 검증 실패
# ---------------------------------------------------------------------------


def test_receive_and_parse_unsupported_sample_rate_raises():
    audio_data = _make_audio_data(sample_rate=11025)  # 지원 안 함
    provider = _make_mock_provider(audio_data=audio_data)
    module = AudioInputModule(provider)

    with pytest.raises(InvalidAudioFormatError):
        module.receive_and_parse()


def test_receive_and_parse_unsupported_format_raises():
    audio_data = _make_audio_data(fmt="OGG_VORBIS")  # 지원 안 함
    provider = _make_mock_provider(audio_data=audio_data)
    module = AudioInputModule(provider)

    with pytest.raises(InvalidAudioFormatError):
        module.receive_and_parse()


def test_receive_and_parse_empty_audio_raises():
    audio_data = AudioData(
        audio=np.array([]),
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        format="PCM",
    )
    provider = _make_mock_provider(audio_data=audio_data)
    module = AudioInputModule(provider)

    with pytest.raises(CorruptedAudioDataError):
        module.receive_and_parse()


def test_receive_and_parse_too_many_channels_raises():
    """채널 수 3 이상 → InvalidAudioFormatError"""
    audio_data = _make_audio_data(channels=5)
    provider = _make_mock_provider(audio_data=audio_data)
    module = AudioInputModule(provider)

    with pytest.raises(InvalidAudioFormatError):
        module.receive_and_parse()


def test_receive_and_parse_zero_channels_raises():
    audio_data = _make_audio_data(channels=0)
    provider = _make_mock_provider(audio_data=audio_data)
    module = AudioInputModule(provider)

    with pytest.raises(InvalidAudioFormatError):
        module.receive_and_parse()


def test_receive_and_parse_propagates_provider_exception():
    """provider.receive_audio() 예외 → 그대로 전파"""
    provider = _make_mock_provider()
    provider.receive_audio.side_effect = ConnectionError("not connected")
    module = AudioInputModule(provider)

    with pytest.raises(Exception):
        module.receive_and_parse()


# ---------------------------------------------------------------------------
# get_source_info
# ---------------------------------------------------------------------------


def test_get_source_info_delegates_to_provider():
    provider = _make_mock_provider(source_type="file")
    module = AudioInputModule(provider)

    info = module.get_source_info()

    provider.get_source_info.assert_called_once()
    assert info["type"] == "file"
