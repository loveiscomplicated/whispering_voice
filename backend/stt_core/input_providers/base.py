"""
오디오 입력 소스 추상 클래스 (Strategy Pattern)

모든 오디오 입력 제공자가 구현해야 할 인터페이스:
- FileAudioProvider: 파일에서 음성 수신
- BluetoothAudioProvider: 블루투스 디바이스에서 음성 수신
"""

from abc import ABC, abstractmethod
from typing import Dict
import logging


logger = logging.getLogger(__name__)


class IAudioInputProvider(ABC):
    """
    모든 오디오 입력 소스가 구현해야 할 추상 인터페이스

    Strategy Pattern을 사용하여 입력 소스를 교체 가능하게 설계

    예시:
        # 파일 입력
        file_provider = FileAudioProvider("audio.wav")
        pipeline = SpeechToTextPipeline(file_provider, config)

        # 블루투스 입력 (동일한 코드!)
        bt_provider = BluetoothAudioProvider("Whisper Device")
        pipeline = SpeechToTextPipeline(bt_provider, config)
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        입력 소스에 연결

        파일의 경우: 파일 로드
        블루투스의 경우: 디바이스와 연결

        Returns:
            bool: 연결 성공 여부

        Raises:
            ConnectionError: 연결 실패
            FileNotFoundError: 파일을 찾을 수 없음
            TimeoutError: 연결 타임아웃
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        입력 소스와의 연결 해제

        파일의 경우: 리소스 해제
        블루투스의 경우: 소켓 종료
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        입력 소스 연결 상태 확인

        Returns:
            bool: 연결된 상태면 True, 아니면 False
        """
        pass

    @abstractmethod
    def receive_audio(self):
        """
        음성 데이터 수신

        파일의 경우: 파일 전체 데이터 반환
        블루투스의 경우: 버퍼된 데이터 반환

        Returns:
            AudioData: 오디오 데이터 객체

        Raises:
            ConnectionError: 연결되지 않음
            TimeoutError: 데이터 수신 타임아웃
            InvalidAudioFormatError: 지원하지 않는 포맷
        """
        pass

    @abstractmethod
    def get_source_info(self) -> Dict:
        """
        입력 소스의 메타데이터 반환

        Returns:
            Dict: 입력 소스 정보

        예시:
            {
                "type": "file" | "bluetooth",
                "source_name": "audio.wav" | "Whisper Device",
                "sample_rate": 16000,
                "channels": 1,
                "format": "PCM",
                "status": "connected" | "disconnected",
                "duration_ms": 5000  # 파일의 경우
            }
        """
        pass


# ============================================================================
# 커스텀 예외 클래스
# ============================================================================


class AudioInputException(Exception):
    """오디오 입력 관련 기본 예외"""

    pass


class ConnectionError(AudioInputException):
    """입력 소스 연결 오류"""

    pass


class TimeoutError(AudioInputException):
    """데이터 수신 타임아웃"""

    pass


class InvalidAudioFormatError(AudioInputException):
    """지원하지 않는 오디오 포맷"""

    pass


class CorruptedAudioDataError(AudioInputException):
    """손상된 오디오 데이터"""

    pass


class BluetoothConnectionError(ConnectionError):
    """블루투스 연결 오류"""

    pass


class FileNotFoundError(ConnectionError):
    """파일을 찾을 수 없음"""

    pass


# ============================================================================
# 사용 예시
# ============================================================================

"""
# 예시 1: 파일 입력
from backend.stt_core.input_providers.file_provider import FileAudioProvider

provider = FileAudioProvider("test_audio.wav")
provider.connect()
audio_data = provider.receive_audio()
provider.disconnect()

# 예시 2: 블루투스 입력
from backend.stt_core.input_providers.bluetooth_provider import BluetoothAudioProvider

provider = BluetoothAudioProvider("Whisper Device")
provider.connect()
audio_data = provider.receive_audio()
provider.disconnect()

# 예시 3: 파이프라인에서 사용 (입력 소스와 무관)
from backend.stt_core.pipeline.speech_to_text_pipeline import SpeechToTextPipeline
from backend.stt_core.stt_models.whisper_model import WhisperModel

config = load_config("config.json")
stt_model = WhisperModel(model_size="base")

# 파일 입력으로 테스트
file_provider = FileAudioProvider("test_audio.wav")
pipeline = SpeechToTextPipeline(file_provider, stt_model, config)
result = pipeline.process_audio()

# 블루투스 입력으로 실제 사용 (코드 변경 없음!)
bt_provider = BluetoothAudioProvider("Whisper Device")
pipeline = SpeechToTextPipeline(bt_provider, stt_model, config)
result = pipeline.process_audio()
"""
