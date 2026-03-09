"""
오디오 데이터 클래스
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from datetime import datetime


@dataclass
class AudioData:
    """음성 데이터를 표현하는 데이터 클래스"""

    # 필수 필드
    audio: np.ndarray  # 음성 샘플 (1D 또는 2D numpy array)
    sample_rate: int  # 샘플 레이트 (Hz, 일반적으로 16000)
    channels: int  # 채널 수 (1=mono, 2=stereo)
    bit_depth: int  # 비트 깊이 (16, 24, 32)
    format: str  # 포맷 ("PCM", "WAV", "MP3", etc.)

    # 계산된 필드
    duration_ms: float = field(init=False)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 메타데이터
    metadata: Dict = field(default_factory=dict)
    source: str = field(default="unknown")  # "bluetooth" or "file"

    def __post_init__(self):
        """duration_ms 자동 계산"""
        if len(self.audio) > 0:
            self.duration_ms = (len(self.audio) / self.sample_rate) * 1000
        else:
            self.duration_ms = 0

    def get_chunks(self, chunk_samples: int):
        """
        오디오를 청크로 나누어 반환 (메모리 효율)

        Args:
            chunk_samples: 청크 크기 (샘플 수)

        Yields:
            np.ndarray: 청크 데이터
        """
        for start in range(0, len(self.audio), chunk_samples):
            end = min(start + chunk_samples, len(self.audio))
            yield self.audio[start:end]

    def get_duration_seconds(self) -> float:
        """음성 길이를 초 단위로 반환"""
        return self.duration_ms / 1000

    def __str__(self) -> str:
        return (
            f"AudioData(duration={self.duration_ms:.2f}ms, "
            f"sr={self.sample_rate}Hz, ch={self.channels}, "
            f"fmt={self.format})"
        )


# 커스텀 예외
class InvalidAudioFormatError(Exception):
    """잘못된 오디오 포맷 예외"""

    pass


class CorruptedAudioDataError(Exception):
    """손상된 오디오 데이터 예외"""

    pass


class AudioProcessingError(Exception):
    """오디오 처리 중 발생한 예외"""

    pass
