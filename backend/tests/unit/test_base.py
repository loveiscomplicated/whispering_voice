"""
IAudioInputProvider 추상 클래스 및 커스텀 예외 단위 테스트
"""

import pytest
import backend.stt_core.input_providers.base as base_module
from backend.stt_core.input_providers.base import IAudioInputProvider


# ---------------------------------------------------------------------------
# 테스트용 구체 구현체
# ---------------------------------------------------------------------------


class _ConcreteProvider(IAudioInputProvider):
    """추상 메서드를 모두 구현한 최소 구체 클래스"""

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def receive_audio(self):
        return None

    def get_source_info(self) -> dict:
        return {"type": "test", "status": "connected"}


# ---------------------------------------------------------------------------
# 추상 클래스
# ---------------------------------------------------------------------------


def test_abstract_class_cannot_be_instantiated():
    """IAudioInputProvider 직접 인스턴스화 불가"""
    with pytest.raises(TypeError):
        IAudioInputProvider()


def test_concrete_provider_can_be_instantiated():
    """추상 메서드를 모두 구현하면 인스턴스화 가능"""
    provider = _ConcreteProvider()
    assert provider is not None


def test_concrete_provider_connect():
    provider = _ConcreteProvider()
    assert provider.connect() is True


def test_concrete_provider_disconnect():
    """disconnect는 None 반환, 예외 없음"""
    provider = _ConcreteProvider()
    result = provider.disconnect()
    assert result is None


def test_concrete_provider_is_connected():
    provider = _ConcreteProvider()
    assert provider.is_connected() is True


def test_concrete_provider_receive_audio():
    provider = _ConcreteProvider()
    assert provider.receive_audio() is None


def test_concrete_provider_get_source_info():
    provider = _ConcreteProvider()
    info = provider.get_source_info()
    assert isinstance(info, dict)
    assert info["type"] == "test"


def test_partial_implementation_raises_type_error():
    """일부만 구현하면 TypeError"""

    class _Partial(IAudioInputProvider):
        def connect(self):
            return True
        # disconnect, is_connected, receive_audio, get_source_info 미구현

    with pytest.raises(TypeError):
        _Partial()


# ---------------------------------------------------------------------------
# 커스텀 예외 계층
# ---------------------------------------------------------------------------


def test_audio_input_exception_is_base_exception():
    exc = base_module.AudioInputException("base error")
    assert isinstance(exc, Exception)
    assert str(exc) == "base error"


def test_connection_error_is_audio_input_exception():
    exc = base_module.ConnectionError("conn")
    assert isinstance(exc, base_module.AudioInputException)


def test_timeout_error_is_audio_input_exception():
    exc = base_module.TimeoutError("timed out")
    assert isinstance(exc, base_module.AudioInputException)


def test_invalid_audio_format_error_is_audio_input_exception():
    exc = base_module.InvalidAudioFormatError("bad format")
    assert isinstance(exc, base_module.AudioInputException)


def test_corrupted_audio_data_error_is_audio_input_exception():
    exc = base_module.CorruptedAudioDataError("corrupted")
    assert isinstance(exc, base_module.AudioInputException)


def test_bluetooth_connection_error_is_connection_error():
    exc = base_module.BluetoothConnectionError("bt failed")
    assert isinstance(exc, base_module.ConnectionError)
    assert isinstance(exc, base_module.AudioInputException)


def test_file_not_found_error_is_connection_error():
    exc = base_module.FileNotFoundError("no file")
    assert isinstance(exc, base_module.ConnectionError)
    assert isinstance(exc, base_module.AudioInputException)


def test_exceptions_can_be_raised_and_caught():
    """예외를 실제로 raise/catch"""
    with pytest.raises(base_module.ConnectionError):
        raise base_module.ConnectionError("test")

    with pytest.raises(base_module.AudioInputException):
        raise base_module.TimeoutError("test")
