"""
파일 기반 오디오 입력 제공자 (파일럿 테스트용)
Strategy Pattern 구현
"""

import logging
import os
from typing import Dict

import numpy as np
import librosa

from backend.stt_core.input_providers.base import IAudioInputProvider
from backend.stt_core.audio_input.audio_data import AudioData, InvalidAudioFormatError


logger = logging.getLogger(__name__)


class FileAudioProvider(IAudioInputProvider):
    """오디오 파일에서 음성 수신 (파일럿 테스트용)"""

    def __init__(self, file_path: str, mono: bool = True):
        """
        Args:
            file_path: 오디오 파일 경로
            mono: True인 경우 모노로 변환
        """
        self.file_path = file_path
        self.mono = mono
        self._is_connected = False
        self._audio_buffer = None
        self._sample_rate = None
        self._channels = None

    def connect(self) -> bool:
        """
        오디오 파일 로드

        Returns:
            bool: 로드 성공 여부

        Raises:
            FileNotFoundError: 파일을 찾을 수 없음
            InvalidAudioFormatError: 지원하지 않는 포맷
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Audio file not found: {self.file_path}")

            logger.info(f"Loading audio file: {self.file_path}")

            # librosa로 오디오 파일 로드
            self._audio_buffer, self._sample_rate = librosa.load(
                self.file_path, sr=None, mono=self.mono  # 원본 샘플 레이트 유지
            )

            # 채널 정보 추출
            if self._audio_buffer.ndim == 1:
                self._channels = 1
            else:
                self._channels = self._audio_buffer.shape[0]

            self._is_connected = True

            logger.info(
                f"✓ Audio file loaded: "
                f"sr={self._sample_rate}Hz, ch={self._channels}, "
                f"duration={len(self._audio_buffer)/self._sample_rate:.2f}s"
            )

            return True

        except FileNotFoundError as e:
            logger.error(f"✗ File not found: {e}")
            raise

        except Exception as e:
            logger.error(f"✗ Failed to load audio file: {e}")
            raise InvalidAudioFormatError(
                f"Failed to load audio file: {self.file_path}\\n{str(e)}"
            )

    def disconnect(self) -> None:
        """파일 리소스 해제"""
        if self._audio_buffer is not None:
            del self._audio_buffer
            self._audio_buffer = None

        self._is_connected = False
        logger.info(f"Disconnected from file: {self.file_path}")

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self._is_connected

    def receive_audio(self) -> AudioData:
        """
        파일에서 음성 데이터 반환

        Returns:
            AudioData: 오디오 데이터 객체

        Raises:
            ConnectionError: 파일이 로드되지 않음
            InvalidAudioFormatError: 오디오 데이터 포맷 오류
        """
        if not self._is_connected:
            raise ConnectionError("Audio file not loaded. Call connect() first.")

        try:
            # AudioData 객체 생성
            audio_data = AudioData(
                audio=self._audio_buffer,
                sample_rate=self._sample_rate,
                channels=self._channels,
                bit_depth=32,  # librosa는 float32로 로드
                format="PCM",
                source="file",
                metadata={
                    "file_path": self.file_path,
                    "file_name": os.path.basename(self.file_path),
                },
            )

            logger.info(f"Read audio from file: {audio_data}")

            return audio_data

        except Exception as e:
            logger.error(f"Failed to read audio: {e}")
            raise InvalidAudioFormatError(f"Failed to read audio: {e}")

    def get_source_info(self) -> Dict:
        """입력 소스의 메타데이터 반환"""
        if not self._is_connected:
            return {
                "type": "file",
                "status": "disconnected",
                "file_path": self.file_path,
            }

        return {
            "type": "file",
            "status": "connected",
            "file_path": self.file_path,
            "file_name": os.path.basename(self.file_path),
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "format": "PCM",
            "duration_ms": int(len(self._audio_buffer) / self._sample_rate * 1000),
        }
