"""
블루투스 기반 오디오 입력 제공자
Strategy Pattern 구현 - 실제 하드웨어용
"""

import logging
from typing import Dict, Optional

from stt_core.input_providers.base import IAudioInputProvider
from stt_core.audio_input.audio_data import AudioData, InvalidAudioFormatError


logger = logging.getLogger(__name__)


class BluetoothAudioProvider(IAudioInputProvider):
    """블루투스 디바이스에서 음성 수신"""

    def __init__(self, device_name: str, timeout_ms: int = 5000):
        """
        Args:
            device_name: 블루투스 디바이스 이름
            timeout_ms: 수신 타임아웃 (밀리초)
        """
        self.device_name = device_name
        self.timeout_ms = timeout_ms
        self._bluetooth_socket = None
        self._is_connected = False

    def connect(self) -> bool:
        """
        블루투스 디바이스에 연결

        Returns:
            bool: 연결 성공 여부

        Raises:
            Exception: 블루투스 연결 실패
        """
        try:
            logger.info(f"Connecting to Bluetooth device: {self.device_name}")

            # 플랫폼별 블루투스 API 사용
            # iOS: CoreBluetooth (Swift에서 구현)
            # Android: BluetoothAdapter (Kotlin에서 구현)
            # 이 부분은 모바일 앱에서 처리하고,
            # 수신된 데이터를 Python 백엔드로 전달

            # TODO: 모바일 앱과의 통신 인터페이스 정의
            # 지금은 파일럿 테스트를 위해 FileAudioProvider 사용

            self._is_connected = True
            logger.info(f"✓ Connected to Bluetooth device: {self.device_name}")

            return True

        except Exception as e:
            logger.error(f"✗ Bluetooth connection failed: {e}")
            raise

    def disconnect(self) -> None:
        """블루투스 연결 해제"""
        if self._bluetooth_socket:
            try:
                self._bluetooth_socket.close()
            except Exception as e:
                logger.warning(f"Error closing socket: {e}")

        self._is_connected = False
        logger.info(f"Disconnected from Bluetooth device: {self.device_name}")

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self._is_connected

    def receive_audio(self) -> AudioData:
        """
        블루투스에서 음성 데이터 수신

        Returns:
            AudioData: 수신한 음성 데이터

        Raises:
            ConnectionError: 연결되지 않음
            TimeoutError: 수신 타임아웃
            InvalidAudioFormatError: 잘못된 포맷
        """
        if not self._is_connected:
            raise ConnectionError("Bluetooth device not connected")

        try:
            logger.debug(f"Receiving audio from Bluetooth device...")

            # TODO: 실제 블루투스 데이터 수신 구현
            # buffer = self._bluetooth_socket.recv(timeout=self.timeout_ms)
            # audioData = self._parse_bluetooth_buffer(buffer)

            raise NotImplementedError(
                "Bluetooth audio reception not yet implemented. "
                "Use FileAudioProvider for pilot testing."
            )

        except TimeoutError:
            logger.error("✗ Bluetooth audio reception timeout")
            raise TimeoutError(f"Audio reception timeout ({self.timeout_ms}ms)")

        except Exception as e:
            logger.error(f"✗ Bluetooth audio reception failed: {e}")
            raise InvalidAudioFormatError(f"Invalid Bluetooth audio: {e}")

    def get_source_info(self) -> Dict:
        """입력 소스의 메타데이터 반환"""
        return {
            "type": "bluetooth",
            "device_name": self.device_name,
            "status": "connected" if self._is_connected else "disconnected",
            "sample_rate": 16000,
            "channels": 1,
            "format": "PCM",
        }
