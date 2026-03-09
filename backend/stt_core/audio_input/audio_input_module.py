"""
오디오 입력 모듈
입력 소스와 무관하게 음성 데이터를 안전하게 처리
"""

import logging
from typing import Optional

import numpy as np

from stt_core.audio_input.audio_data import (
    AudioData,
    InvalidAudioFormatError,
    CorruptedAudioDataError,
)
from stt_core.input_providers.base import IAudioInputProvider


logger = logging.getLogger(__name__)


class AudioInputModule:
    """음성 입력 처리 모듈"""

    # 지원하는 샘플 레이트
    SUPPORTED_SAMPLE_RATES = [8000, 16000, 44100, 48000]

    # 지원하는 포맷
    SUPPORTED_FORMATS = ["PCM", "WAV", "MP3"]

    def __init__(self, audio_provider: IAudioInputProvider):
        """
        Args:
            audio_provider: IAudioInputProvider를 구현한 입력 소스
        """
        self.provider = audio_provider
        logger.info(
            f"AudioInputModule initialized with {audio_provider.__class__.__name__}"
        )

    def connect(self) -> bool:
        """입력 소스에 연결"""
        try:
            result = self.provider.connect()
            if result:
                logger.info(f"✓ Connected to {self.provider.get_source_info()['type']}")
            return result
        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            raise

    def disconnect(self) -> None:
        """입력 소스와의 연결 해제"""
        self.provider.disconnect()
        logger.info("Disconnected from audio provider")

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self.provider.is_connected()

    def receive_and_parse(self) -> AudioData:
        """
        입력 소스에서 음성 데이터 수신 및 검증

        Returns:
            AudioData: 검증된 음성 데이터

        Raises:
            ConnectionError: 연결되지 않음
            InvalidAudioFormatError: 잘못된 포맷
            CorruptedAudioDataError: 손상된 데이터
        """
        try:
            # 1. 입력 소스에서 데이터 수신
            audio_data = self.provider.receive_audio()
            logger.debug(f"Received audio: {audio_data}")

            # 2. 데이터 검증
            self._validate_audio_data(audio_data)

            # 3. 로깅
            source_info = self.provider.get_source_info()
            logger.info(
                f"✓ Audio received from {source_info['type']}: "
                f"duration={audio_data.duration_ms:.2f}ms, "
                f"sr={audio_data.sample_rate}Hz"
            )

            return audio_data

        except Exception as e:
            logger.error(f"✗ Audio reception failed: {e}")
            raise

    def _validate_audio_data(self, audio_data: AudioData) -> None:
        """
        오디오 데이터 검증

        Args:
            audio_data: 검증할 오디오 데이터

        Raises:
            InvalidAudioFormatError: 포맷 오류
            CorruptedAudioDataError: 데이터 손상
        """
        # 1. 샘플 레이트 확인
        if audio_data.sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise InvalidAudioFormatError(
                f"Unsupported sample rate: {audio_data.sample_rate}Hz. "
                f"Supported: {self.SUPPORTED_SAMPLE_RATES}"
            )

        # 2. 포맷 확인
        if audio_data.format not in self.SUPPORTED_FORMATS:
            raise InvalidAudioFormatError(
                f"Unsupported format: {audio_data.format}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        # 3. 데이터 길이 확인
        if len(audio_data.audio) == 0:
            raise CorruptedAudioDataError("Audio data is empty")

        # 4. 모든 샘플이 0인지 확인 (무음)
        if np.all(audio_data.audio == 0):
            logger.warning("⚠ Audio contains only silence (all zeros)")

        # 5. 채널 수 확인
        if audio_data.channels < 1 or audio_data.channels > 2:
            raise InvalidAudioFormatError(
                f"Invalid number of channels: {audio_data.channels}. "
                f"Must be 1 (mono) or 2 (stereo)"
            )

        logger.debug("✓ Audio data validation passed")

    def get_source_info(self) -> dict:
        """입력 소스 정보 반환"""
        return self.provider.get_source_info()
