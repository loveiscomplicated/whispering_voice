"""
BluetoothAudioProvider 단위 테스트
"""

import pytest
from unittest.mock import MagicMock

from backend.stt_core.input_providers.bluetooth_provider import BluetoothAudioProvider
from backend.stt_core.audio_input.audio_data import InvalidAudioFormatError


# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------


def test_init_stores_device_name():
    provider = BluetoothAudioProvider(device_name="MyEarbuds")
    assert provider.device_name == "MyEarbuds"


def test_init_stores_timeout():
    provider = BluetoothAudioProvider(device_name="Dev", timeout_ms=3000)
    assert provider.timeout_ms == 3000


def test_init_default_timeout():
    provider = BluetoothAudioProvider(device_name="Dev")
    assert provider.timeout_ms == 5000


def test_init_not_connected():
    provider = BluetoothAudioProvider(device_name="Dev")
    assert provider.is_connected() is False


def test_init_no_socket():
    provider = BluetoothAudioProvider(device_name="Dev")
    assert provider._bluetooth_socket is None


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


def test_connect_returns_true():
    provider = BluetoothAudioProvider(device_name="Dev")
    result = provider.connect()
    assert result is True


def test_connect_sets_connected_state():
    provider = BluetoothAudioProvider(device_name="Dev")
    provider.connect()
    assert provider.is_connected() is True


# ---------------------------------------------------------------------------
# disconnect
# ---------------------------------------------------------------------------


def test_disconnect_after_connect():
    provider = BluetoothAudioProvider(device_name="Dev")
    provider.connect()
    provider.disconnect()
    assert provider.is_connected() is False


def test_disconnect_without_connect_does_not_raise():
    """연결 없이 disconnect 해도 예외 없음"""
    provider = BluetoothAudioProvider(device_name="Dev")
    provider.disconnect()  # should not raise
    assert provider.is_connected() is False


def test_disconnect_closes_socket_when_present():
    """소켓이 있을 때 close() 호출"""
    provider = BluetoothAudioProvider(device_name="Dev")
    mock_socket = MagicMock()
    provider._bluetooth_socket = mock_socket
    provider._is_connected = True

    provider.disconnect()

    mock_socket.close.assert_called_once()
    assert provider.is_connected() is False


def test_disconnect_handles_socket_close_error():
    """소켓 close() 에러도 예외 없이 처리"""
    provider = BluetoothAudioProvider(device_name="Dev")
    mock_socket = MagicMock()
    mock_socket.close.side_effect = OSError("socket error")
    provider._bluetooth_socket = mock_socket
    provider._is_connected = True

    provider.disconnect()  # should not raise

    assert provider.is_connected() is False


# ---------------------------------------------------------------------------
# receive_audio
# ---------------------------------------------------------------------------


def test_receive_audio_not_connected_raises_connection_error():
    provider = BluetoothAudioProvider(device_name="Dev")

    with pytest.raises(ConnectionError):
        provider.receive_audio()


def test_receive_audio_connected_raises_invalid_audio_format():
    """NotImplementedError가 InvalidAudioFormatError로 래핑됨"""
    provider = BluetoothAudioProvider(device_name="Dev")
    provider.connect()

    with pytest.raises(InvalidAudioFormatError):
        provider.receive_audio()


# ---------------------------------------------------------------------------
# get_source_info
# ---------------------------------------------------------------------------


def test_get_source_info_disconnected():
    provider = BluetoothAudioProvider(device_name="TestDevice")
    info = provider.get_source_info()

    assert info["type"] == "bluetooth"
    assert info["device_name"] == "TestDevice"
    assert info["status"] == "disconnected"


def test_get_source_info_connected():
    provider = BluetoothAudioProvider(device_name="TestDevice")
    provider.connect()
    info = provider.get_source_info()

    assert info["type"] == "bluetooth"
    assert info["device_name"] == "TestDevice"
    assert info["status"] == "connected"


def test_get_source_info_has_audio_fields():
    provider = BluetoothAudioProvider(device_name="Dev")
    info = provider.get_source_info()

    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert info["format"] == "PCM"
