# 속삭임 감지 음성 변환 시스템 - 모듈 초기 코드 템플릿

프로젝트 구조를 생성한 후, 각 핵심 모듈에 들어갈 초기 코드 템플릿입니다.

---

## 1. AudioData 클래스 (기본 데이터 구조)

**파일**: `backend/stt_core/audio_input/audio_data.py`

```python
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
    audio: np.ndarray              # 음성 샘플 (1D 또는 2D numpy array)
    sample_rate: int               # 샘플 레이트 (Hz, 일반적으로 16000)
    channels: int                  # 채널 수 (1=mono, 2=stereo)
    bit_depth: int                 # 비트 깊이 (16, 24, 32)
    format: str                    # 포맷 ("PCM", "WAV", "MP3", etc.)
    
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
```

---

## 2. FileAudioProvider (파일럿 테스트용 입력 소스)

**파일**: `backend/stt_core/input_providers/file_provider.py`

```python
"""
파일 기반 오디오 입력 제공자 (파일럿 테스트용)
Strategy Pattern 구현
"""

import logging
import os
from typing import Dict

import numpy as np
import librosa

from .base import IAudioInputProvider
from ..audio_input.audio_data import AudioData, InvalidAudioFormatError


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
                self.file_path,
                sr=None,           # 원본 샘플 레이트 유지
                mono=self.mono
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
                f"Failed to load audio file: {self.file_path}\n{str(e)}"
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
                    "file_name": os.path.basename(self.file_path)
                }
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
                "file_path": self.file_path
            }
        
        return {
            "type": "file",
            "status": "connected",
            "file_path": self.file_path,
            "file_name": os.path.basename(self.file_path),
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "format": "PCM",
            "duration_ms": int(len(self._audio_buffer) / self._sample_rate * 1000)
        }
```

---

## 3. AudioInputModule (입력 처리)

**파일**: `backend/stt_core/audio_input/audio_input_module.py`

```python
"""
오디오 입력 모듈
입력 소스와 무관하게 음성 데이터를 안전하게 처리
"""

import logging
from typing import Optional

import numpy as np

from .audio_data import AudioData, InvalidAudioFormatError, CorruptedAudioDataError
from ..input_providers.base import IAudioInputProvider


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
        logger.info(f"AudioInputModule initialized with {audio_provider.__class__.__name__}")
    
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
```

---

## 4. BluetoothAudioProvider (실제 하드웨어용)

**파일**: `backend/stt_core/input_providers/bluetooth_provider.py`

```python
"""
블루투스 기반 오디오 입력 제공자
Strategy Pattern 구현 - 실제 하드웨어용
"""

import logging
from typing import Dict, Optional

from .base import IAudioInputProvider
from ..audio_input.audio_data import AudioData, InvalidAudioFormatError


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
            "format": "PCM"
        }
```

---

## 5. 프로젝트 설정 (Config)

**파일**: `backend/stt_core/pipeline/config.py`

```python
"""
파이프라인 설정 관리
"""

import json
import logging
from typing import Dict, Any
from pathlib import Path


logger = logging.getLogger(__name__)


class Config:
    """애플리케이션 설정 관리 클래스"""
    
    def __init__(self, config_path: str = "backend/config/config.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"✓ Config loaded from {self.config_path}")
                return config
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return self._get_default_config()
        
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "inputSource": {
                "type": "file",
                "filePath": "test_audio.wav"
            },
            "audioInput": {
                "maxDuration_seconds": 300,
                "timeout_ms": 5000
            },
            "preprocessing": {
                "targetSampleRate": 16000,
                "chunkSize_seconds": 2
            },
            "stt": {
                "model": "whisper",
                "modelSize": "base"
            },
            "pipeline": {
                "processingMode": "hybrid"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        설정 값 조회
        
        Args:
            key: 설정 키 (점으로 구분, 예: "stt.model")
            default: 기본값
        
        Returns:
            설정 값
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def __getitem__(self, key: str) -> Dict:
        """딕셔너리처럼 접근"""
        return self.config.get(key, {})
```

---

## 6. 기본 테스트 코드

**파일**: `backend/tests/unit/test_audio_data.py`

```python
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
        audio=audio,
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        format="PCM"
    )
    
    assert data.sample_rate == 16000
    assert data.channels == 1
    assert abs(data.duration_ms - 1000.0) < 1.0  # 대략 1초


def test_audio_data_chunking():
    """오디오 청크 분할 테스트"""
    audio = np.random.randn(32000)  # 2초 오디오 (16kHz)
    data = AudioData(
        audio=audio,
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        format="PCM"
    )
    
    chunks = list(data.get_chunks(8000))  # 0.5초씩 분할
    
    assert len(chunks) == 4


def test_audio_data_duration():
    """음성 길이 계산 테스트"""
    audio = np.random.randn(48000)  # 3초
    data = AudioData(
        audio=audio,
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        format="PCM"
    )
    
    assert abs(data.get_duration_seconds() - 3.0) < 0.01
```

---

## 7. 파일럿 테스트 스크립트

**파일**: `scripts/run_pilot_test.py`

```python
#!/usr/bin/env python
"""
파일럿 테스트 실행 스크립트
"""

import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.stt_core.input_providers.file_provider import FileAudioProvider
from backend.stt_core.audio_input.audio_input_module import AudioInputModule
from backend.stt_core.pipeline.config import Config


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pilot_test(audio_file_path: str):
    """
    파일 입력을 사용한 파일럿 테스트
    
    Args:
        audio_file_path: 테스트 음성 파일 경로
    """
    logger.info("=" * 60)
    logger.info("🚀 파일럿 테스트 시작")
    logger.info("=" * 60)
    
    try:
        # 1. 설정 로드
        logger.info("\n[1/5] 설정 로드 중...")
        config = Config()
        logger.info(f"✓ 설정 로드 완료")
        
        # 2. 입력 소스 생성 (파일)
        logger.info("\n[2/5] 파일 입력 소스 생성 중...")
        if not os.path.exists(audio_file_path):
            logger.error(f"✗ 파일을 찾을 수 없음: {audio_file_path}")
            logger.info("💡 테스트 파일을 생성하려면:")
            logger.info("   python scripts/generate_test_audio.py")
            return
        
        file_provider = FileAudioProvider(audio_file_path)
        logger.info(f"✓ 파일 입력 소스 생성 완료")
        
        # 3. 입력 모듈 생성
        logger.info("\n[3/5] 입력 모듈 초기화 중...")
        audio_input_module = AudioInputModule(file_provider)
        logger.info(f"✓ 입력 모듈 초기화 완료")
        
        # 4. 음성 수신
        logger.info("\n[4/5] 음성 파일 읽기 중...")
        audio_input_module.connect()
        audio_data = audio_input_module.receive_and_parse()
        logger.info(f"✓ 음성 데이터 수신 완료:")
        logger.info(f"  - 길이: {audio_data.duration_ms:.2f}ms")
        logger.info(f"  - 샘플 레이트: {audio_data.sample_rate}Hz")
        logger.info(f"  - 채널: {audio_data.channels}")
        logger.info(f"  - 포맷: {audio_data.format}")
        
        # 5. 메타데이터 출력
        logger.info("\n[5/5] 수신 정보:")
        source_info = audio_input_module.get_source_info()
        for key, value in source_info.items():
            logger.info(f"  - {key}: {value}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 파일럿 테스트 성공!")
        logger.info("=" * 60)
        
        audio_input_module.disconnect()
        
        return audio_data
        
    except Exception as e:
        logger.error(f"\n✗ 파일럿 테스트 실패: {e}", exc_info=True)
        logger.info("\n🔍 문제 해결:")
        logger.info("1. 파일 경로 확인")
        logger.info("2. 파일이 올바른 오디오 포맷인지 확인 (WAV, MP3 등)")
        logger.info("3. 의존성 설치 확인: pip install librosa")
        return None


if __name__ == "__main__":
    # 테스트 음성 파일 경로
    test_audio_path = "test_audio.wav"
    
    # 파일 없으면 기본 경로 사용
    if not os.path.exists(test_audio_path):
        logger.warning(f"테스트 파일 없음: {test_audio_path}")
        logger.info("다음을 실행하여 테스트 파일을 생성하세요:")
        logger.info("  python scripts/generate_test_audio.py")
    else:
        run_pilot_test(test_audio_path)
```

---

## 8. 테스트 음성 파일 생성 스크립트

**파일**: `scripts/generate_test_audio.py`

```python
#!/usr/bin/env python
"""
테스트용 음성 파일 생성 스크립트
"""

import numpy as np
from scipy.io import wavfile
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_audio(output_path: str = "test_audio.wav"):
    """
    테스트용 음성 파일 생성
    
    Args:
        output_path: 출력 파일 경로
    """
    logger.info("🎵 테스트 음성 파일 생성 중...")
    
    # 파라미터
    sample_rate = 16000  # Hz
    duration_seconds = 3  # 3초
    frequencies = [440, 500, 600]  # 여러 주파수를 혼합
    
    # 샘플 생성
    t = np.linspace(0, duration_seconds, duration_seconds * sample_rate)
    
    # 여러 주파수의 사인파를 혼합
    audio = np.zeros_like(t)
    for freq in frequencies:
        audio += 0.2 * np.sin(2 * np.pi * freq * t)
    
    # 정규화
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 0.8).astype(np.float32)  # 80% 볼륨
    
    # WAV 파일로 저장
    wavfile.write(output_path, sample_rate, audio)
    
    logger.info(f"✓ 테스트 음성 파일 생성 완료")
    logger.info(f"  - 파일: {output_path}")
    logger.info(f"  - 길이: {duration_seconds}초")
    logger.info(f"  - 샘플 레이트: {sample_rate}Hz")
    logger.info(f"  - 주파수: {frequencies}")


if __name__ == "__main__":
    generate_test_audio()
```

---

## 사용 방법

### 1단계: 프로젝트 구조 생성

```bash
bash create_project_structure.sh
```

### 2단계: 위의 파일들을 해당 위치에 생성

각 섹션의 코드를 해당 파일에 복사

### 3단계: 개발 환경 설정

```bash
bash scripts/setup_dev_env.sh
source venv/bin/activate
```

### 4단계: 테스트 파일 생성

```bash
python scripts/generate_test_audio.py
```

### 5단계: 파일럿 테스트 실행

```bash
python scripts/run_pilot_test.py
```

---

## 다음 단계

1. ✅ **입력 모듈**: `AudioInputModule` + `FileAudioProvider` (완성)
2. 🔄 **전처리 모듈**: `AudioPreprocessingModule` (구현 예정)
3. 🔄 **STT 모듈**: `STTModule` + `WhisperModel` (구현 예정)
4. 🔄 **결과 처리**: `ResultProcessingModule` (구현 예정)
5. 🔄 **파이프라인**: `SpeechToTextPipeline` (구현 예정)
