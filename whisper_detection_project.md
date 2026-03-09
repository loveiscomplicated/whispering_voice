# 속삭임 감지 음성 변환 시스템 - 소프트웨어 모듈

**Speech-to-Text Processing Module**

---

## 프로젝트 개요

블루투스 하드웨어 디바이스에서 전송된 음성 신호를 수신하여 전처리한 후, 오픈소스 STT(Speech-to-Text) 모델을 활용하여 텍스트로 변환하는 모바일 애플리케이션의 핵심 소프트웨어 모듈 개발 프로젝트입니다.

**핵심 흐름**: 음성 입력 (블루투스 또는 파일) → 전처리 → 노이즈 제거 → STT 변환 → 텍스트 출력

---

## 개발 범위

### 포함 사항 ✓
- 블루투스 또는 파일에서 수신한 음성 신호 처리 (입력 소스 추상화)
- 오디오 신호 전처리 (정규화, 노이즈 제거)
- Whisper 또는 Vosk를 활용한 STT 변환
- 속삭임에 특화된 처리 및 파인튜닝 가능한 구조
- 변환된 텍스트 출력 및 신뢰도 표시
- UI에서 결과 표시

### 제외 사항 ✗
- 블루투스 통신 인터페이스 (하드웨어팀 담당)
- 음성 데이터 수신 구현 (이미 수신된 데이터를 입력받음)
- 하드웨어 드라이버 개발

---

## 기술 스택

| 구분 | 항목 | 상태 |
|------|------|------|
| **타겟 플랫폼** | 모바일 (iOS/Android) | 미정 |
| **개발 언어** | Swift(iOS) / Kotlin(Android) 또는 React Native/Flutter | 미정 |
| **음성 인식 모델** | Whisper 또는 Vosk (오픈소스) | 확정 |
| **음성 처리** | Core Audio(iOS) / MediaPlayer(Android) / librosa | 미정 |
| **입력 소스** | Bluetooth 또는 Audio File (WAV, MP3 등) | 확정 (추상화) |

---

## 소프트웨어 아키텍처

### 전체 흐름도

```
┌──────────────────────────────────────────────────────────────┐
│                    Audio Input Sources                        │
│  ┌──────────────────────┬──────────────────────┐             │
│  │  Bluetooth Device    │   Audio File (WAV)   │             │
│  │  (실제 하드웨어)     │   (파일럿 테스트)    │             │
│  └──────────┬───────────┴──────────┬───────────┘             │
│             │                      │                          │
│             └──────────┬───────────┘                          │
│                        │                                       │
└────────────────────────┼───────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│           IAudioInputProvider (Interface)                     │
│  - receiveAudio(): AudioData                                 │
│  - isConnected(): bool                                       │
│  - connect(): void                                           │
│  - disconnect(): void                                        │
└────────┬─────────────────────────────────────────────────────┘
         │
         ├─ BluetoothAudioProvider (실제 하드웨어)
         └─ FileAudioProvider (파일럿 테스트)
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                  AudioInputModule                             │
│    (입력 소스와 무관하게 처리)                               │
│  - Buffer Management                                         │
│  - Format Parsing & Validation                              │
│  - AudioData Object Creation                                │
│  - Error Handling                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              AudioPreprocessingModule                         │
│        (속삭임 특화 음성 신호 전처리)                        │
│  - Sample Rate Normalization (16kHz)                        │
│  - High-Frequency Emphasis (Whisper-Specific)              │
│  - Spectral Subtraction Noise Removal                       │
│  - Voice Activity Detection (VAD)                           │
│  - RMS Amplitude Normalization                              │
│  - Chunked Processing (2-second chunks)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   STTModule                                   │
│       (모델화 및 확장 가능한 음성 인식)                      │
│  - ISTTModel Interface                                       │
│  - WhisperModel / VoskModel Implementation                  │
│  - Model Loading/Unloading                                  │
│  - Inference Execution                                      │
│  - CustomWhisperModel Ready (Future Fine-tuning)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              ResultProcessingModule                           │
│         (결과 처리 및 포맷팅)                               │
│  - Text Output Formatting                                   │
│  - Confidence Score Calculation                             │
│  - Timestamp Management                                     │
│  - Error Handling & Logging                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              SpeechToTextPipeline                             │
│         (Orchestrator - 전체 파이프라인 조율)               │
│  - Input Provider Selection                                 │
│  - Hierarchical Error Handling                              │
│  - Processing Mode Selection (Batch/Streaming/Hybrid)       │
│  - Configuration Management                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                      UI Layer                                 │
│             (사용자 인터페이스)                              │
│  - Display Text Result                                      │
│  - Show Confidence Score                                    │
│  - Show Processing Status                                   │
│  - Display Errors                                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 입력 소스 추상화 (Strategy Pattern)

### IAudioInputProvider 인터페이스

```python
class IAudioInputProvider:
    """모든 오디오 입력 소스가 구현해야 할 인터페이스"""
    
    def connect() -> bool:
        """
        입력 소스에 연결
        
        반환:
            bool: 연결 성공 여부
        """
        pass
    
    def disconnect() -> None:
        """입력 소스와의 연결 해제"""
        pass
    
    def isConnected() -> bool:
        """입력 소스 연결 상태 확인"""
        pass
    
    def receiveAudio() -> AudioData:
        """
        음성 데이터 수신
        
        반환:
            AudioData: 수신한 음성 데이터
            
        발생 Exception:
            - ConnectionError: 연결 실패
            - TimeoutError: 데이터 수신 타임아웃
            - InvalidAudioFormatError: 지원하지 않는 포맷
        """
        pass
    
    def getSourceInfo() -> Dict:
        """
        입력 소스의 메타데이터 반환
        
        반환:
            {
                "type": "bluetooth" | "file",
                "sampleRate": 16000,
                "channels": 1,
                "format": "PCM",
                "duration_ms": 5000 (파일의 경우)
            }
        """
        pass
```

### BluetoothAudioProvider (실제 하드웨어)

```python
class BluetoothAudioProvider(IAudioInputProvider):
    """블루투스 디바이스에서 음성 수신"""
    
    def __init__(self, deviceName: str, timeout_ms: int = 5000):
        self.deviceName = deviceName
        self.timeout_ms = timeout_ms
        self.bluetoothSocket = None
        self.isConnected = False
    
    def connect() -> bool:
        """블루투스 디바이스에 연결"""
        try:
            # 플랫폼별 블루투스 API 사용
            # iOS: CoreBluetooth
            # Android: BluetoothAdapter
            self.bluetoothSocket = self._connectToBluetooth(self.deviceName)
            self.isConnected = True
            logger.info(f"Connected to Bluetooth device: {self.deviceName}")
            return True
        except Exception as e:
            logger.error(f"Bluetooth connection failed: {e}")
            raise BluetoothConnectionError(f"Failed to connect to {self.deviceName}")
    
    def receiveAudio() -> AudioData:
        """블루투스에서 음성 데이터 수신"""
        if not self.isConnected:
            raise ConnectionError("Bluetooth device not connected")
        
        try:
            # 블루투스 버퍼에서 데이터 읽기
            buffer = self.bluetoothSocket.recv(timeout=self.timeout_ms)
            
            # 음성 데이터 파싱 및 검증
            audioData = self._parseBluetoothBuffer(buffer)
            
            logger.info(f"Received audio from Bluetooth: {audioData.duration_ms}ms")
            return audioData
            
        except TimeoutError:
            raise TimeoutError("Bluetooth audio reception timeout")
        except Exception as e:
            raise InvalidAudioFormatError(f"Invalid Bluetooth audio format: {e}")
    
    def disconnect() -> None:
        """블루투스 연결 해제"""
        if self.bluetoothSocket:
            self.bluetoothSocket.close()
        self.isConnected = False
        logger.info(f"Disconnected from Bluetooth device: {self.deviceName}")
    
    def getSourceInfo() -> Dict:
        return {
            "type": "bluetooth",
            "deviceName": self.deviceName,
            "sampleRate": 16000,
            "channels": 1,
            "format": "PCM"
        }
```

### FileAudioProvider (파일럿 테스트용)

```python
class FileAudioProvider(IAudioInputProvider):
    """오디오 파일에서 음성 수신 (파일럿 테스트용)"""
    
    def __init__(self, filePath: str):
        self.filePath = filePath
        self.audioFile = None
        self.isConnected = False
        self.audioBuffer = None
        self.sampleRate = None
        self.channels = None
    
    def connect() -> bool:
        """오디오 파일 로드"""
        try:
            # librosa 또는 scipy를 사용하여 오디오 파일 로드
            import librosa
            
            self.audioBuffer, self.sampleRate = librosa.load(
                self.filePath,
                sr=None,  # 원본 샘플 레이트 유지
                mono=False
            )
            
            # 채널 정보 추출
            if self.audioBuffer.ndim == 1:
                self.channels = 1
            else:
                self.channels = self.audioBuffer.shape[0]
            
            self.isConnected = True
            logger.info(f"Loaded audio file: {self.filePath}")
            logger.info(f"Sample Rate: {self.sampleRate}, Channels: {self.channels}")
            return True
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {self.filePath}")
        except Exception as e:
            raise InvalidAudioFormatError(f"Failed to load audio file: {e}")
    
    def receiveAudio() -> AudioData:
        """파일에서 음성 데이터 반환"""
        if not self.isConnected:
            raise ConnectionError("Audio file not loaded")
        
        try:
            # 전체 오디오 파일을 AudioData로 변환
            audioData = AudioData(
                audio=self.audioBuffer,
                sampleRate=self.sampleRate,
                channels=self.channels,
                bitDepth=32,  # float32
                format="PCM",
                duration_ms=int(len(self.audioBuffer) / self.sampleRate * 1000),
                timestamp=datetime.now().isoformat(),
                metadata={
                    "filePath": self.filePath,
                    "source": "file"
                }
            )
            
            logger.info(f"Read audio from file: {audioData.duration_ms}ms")
            return audioData
            
        except Exception as e:
            raise InvalidAudioFormatError(f"Failed to read audio file: {e}")
    
    def disconnect() -> None:
        """파일 리소스 해제"""
        if self.audioBuffer is not None:
            del self.audioBuffer
            self.audioBuffer = None
        self.isConnected = False
        logger.info(f"Closed audio file: {self.filePath}")
    
    def getSourceInfo() -> Dict:
        if not self.isConnected:
            return {"type": "file", "status": "not_connected"}
        
        return {
            "type": "file",
            "filePath": self.filePath,
            "sampleRate": self.sampleRate,
            "channels": self.channels,
            "format": "PCM",
            "duration_ms": int(len(self.audioBuffer) / self.sampleRate * 1000)
        }
```

---

## 모듈 상세 설계

### 1. AudioInputModule (수정됨)
**책임**: 입력 소스와 무관하게 음성 데이터를 안전하게 처리

#### 주요 기능

```python
class AudioInputModule:
    def __init__(self, audioProvider: IAudioInputProvider):
        """
        audioProvider: IAudioInputProvider를 구현한 객체
        """
        self.provider = audioProvider
    
    def connect() -> bool:
        """입력 소스에 연결"""
        return self.provider.connect()
    
    def disconnect() -> None:
        """입력 소스와의 연결 해제"""
        self.provider.disconnect()
    
    def receiveAndParse() -> AudioData:
        """입력 소스에서 음성 데이터 수신 및 검증"""
        try:
            # 입력 소스와 무관하게 같은 방식으로 처리
            audioData = self.provider.receiveAudio()
            
            # 데이터 검증
            self.validateAudioData(audioData)
            
            logger.info(f"Received audio from {self.provider.getSourceInfo()['type']}: "
                       f"{audioData.duration_ms}ms, {audioData.sampleRate}Hz")
            
            return audioData
            
        except TimeoutError as e:
            raise TimeoutError(f"Audio reception timeout: {e}")
        except InvalidAudioFormatError as e:
            raise InvalidAudioFormatError(f"Invalid audio format: {e}")
    
    def validateAudioData(audioData: AudioData) -> bool:
        """데이터 검증"""
        # 샘플 레이트 확인
        if audioData.sampleRate not in [8000, 16000, 44100, 48000]:
            raise InvalidAudioFormatError(
                f"Unsupported sample rate: {audioData.sampleRate}"
            )
        
        # 데이터가 모두 0인지 확인 (무음)
        if len(audioData.audio) == 0:
            raise CorruptedAudioDataError("Audio data is empty")
        
        # 데이터 손상 여부 확인
        if numpy.all(audioData.audio == 0):
            logger.warning("Audio data contains only silence")
        
        return True
    
    def getSourceInfo() -> Dict:
        """입력 소스 정보 반환"""
        return self.provider.getSourceInfo()
```

---

### 2. AudioPreprocessingModule
**책임**: 음성 신호를 STT 모델에 최적화된 상태로 변환 (속삭임 특화)

#### 주요 기능

```python
def process(audioData):
    """속삭임에 특화된 전처리"""
    
    # Chunked Processing (메모리 효율)
    CHUNK_SIZE = 16000 * 2  # 2초 단위
    
    for chunk in audioData.getChunks(CHUNK_SIZE):
        # 1. 샘플 레이트 정규화 (16kHz)
        normalized = normalizeSampleRate(chunk, 16000)
        
        # 2. 고주파 강화 (속삭임은 고주파 성분이 많음)
        emphasized = emphasizeHighFrequency(normalized, coef=0.97)
        
        # 3. 속삭임 특화 노이즈 제거 (Spectral Subtraction)
        denoised = removeNoise(emphasized, alpha=3.0)
        
        # 4. 신호 증폭 (왜곡 방지)
        amplified = amplifySignal(denoised, targetRMS=0.3)
        
        # 5. VAD (무음 구간 제거, 속삭임용 낮은 임계값)
        if detectVoiceActivity(amplified, threshold=0.05):
            yield amplified
```

---

### 3. STTModule (Strategy Pattern)
**책임**: 음성 인식 모델 관리 및 추론 실행 (모델 독립적)

```python
class ISTTModel:
    """모든 STT 모델이 구현해야 할 인터페이스"""
    
    def loadModel(modelPath: str) -> None:
        pass
    
    def transcribe(audioData: AudioData) -> Dict:
        pass
    
    def unloadModel() -> None:
        pass

class STTModule:
    def __init__(self, model_provider: ISTTModel):
        self.model = model_provider
    
    def inference(audioData: AudioData):
        return self.model.transcribe(audioData)
```

---

### 4. ResultProcessingModule
**책임**: STT 결과를 사용자에게 표시 가능한 형태로 변환

---

### 5. SpeechToTextPipeline (Orchestrator)
**책임**: 전체 파이프라인을 조율하고 입력 소스 선택

#### 파이프라인 초기화 및 사용

```python
class SpeechToTextPipeline:
    def __init__(self, audioProvider: IAudioInputProvider, config: Dict):
        """
        audioProvider: BluetoothAudioProvider 또는 FileAudioProvider
        config: 설정 정보
        """
        self.audioInputModule = AudioInputModule(audioProvider)
        self.audioPreprocessingModule = AudioPreprocessingModule()
        self.sttModule = STTModule(STTModelFactory.create(config["stt"]))
        self.resultProcessingModule = ResultProcessingModule()
        self.config = config
    
    def processAudio(mode="hybrid"):
        """전체 파이프라인 실행"""
        # Phase 1: 연결
        if not self.audioInputModule.connect():
            return {"status": "error", "code": "CONNECTION_FAILED"}
        
        try:
            # Phase 2: 입력 처리
            audioData = self.audioInputModule.receiveAndParse()
            
            # Phase 3: 전처리
            cleanAudio = self.audioPreprocessingModule.process(audioData)
            
            # Phase 4: STT 변환
            result = self.sttModule.inference(cleanAudio)
            
            # Phase 5: 결과 처리
            finalResult = self.resultProcessingModule.process(result)
            
            return finalResult
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {"status": "error", "message": str(e)}
        
        finally:
            # 연결 해제
            self.audioInputModule.disconnect()

# ============ 사용 예시 ============

# 1. 실제 하드웨어 (블루투스)
bluetoothProvider = BluetoothAudioProvider(deviceName="Whisper Device")
pipeline_hw = SpeechToTextPipeline(bluetoothProvider, config)
result = pipeline_hw.processAudio()

# 2. 파일럿 테스트 (음성 파일)
fileProvider = FileAudioProvider(filePath="/path/to/test_audio.wav")
pipeline_test = SpeechToTextPipeline(fileProvider, config)
result = pipeline_test.processAudio()

# 같은 Pipeline 코드로 둘 다 작동!
```

---

## 파일럿 테스트 가이드

### 테스트용 음성 파일 준비

```python
# test_audio.wav 준비 방법
# 1. 온라인 음성 변환 서비스로 텍스트 → 음성 변환
# 2. 속삭임 음성 샘플 다운로드
# 3. 파이썬으로 합성 음성 생성

# Python으로 테스트 음성 생성 (선택사항)
import numpy as np
from scipy.io import wavfile

# 간단한 사인파 음성 생성 (440Hz)
duration_seconds = 3
sample_rate = 16000
t = np.linspace(0, duration_seconds, duration_seconds * sample_rate)
audio_data = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

# WAV 파일로 저장
wavfile.write('test_audio.wav', sample_rate, audio_data)
```

### 파일럿 테스트 실행 스크립트

```python
# pilot_test.py
import os
from pathlib import Path

def runPilotTest(audioFilePath: str, mode: str = "hybrid"):
    """
    파일럿 테스트 실행
    
    Args:
        audioFilePath: 테스트 음성 파일 경로
        mode: "batch", "streaming", "hybrid"
    
    Returns:
        Dict: 처리 결과
    """
    
    # 1. 입력 소스 선택 (파일)
    if not os.path.exists(audioFilePath):
        print(f"Error: Audio file not found: {audioFilePath}")
        return None
    
    fileProvider = FileAudioProvider(filePath=audioFilePath)
    
    # 2. Pipeline 생성
    config = loadConfig("config.json")
    pipeline = SpeechToTextPipeline(fileProvider, config)
    
    # 3. 모델 로드
    print("Loading model...")
    sttModel = WhisperModel(modelSize="base")
    sttModel.loadModel()
    pipeline.sttModule = STTModule(sttModel)
    
    # 4. 파이프라인 실행
    print(f"Processing audio file: {audioFilePath}")
    print(f"Mode: {mode}")
    result = pipeline.processAudio(mode=mode)
    
    # 5. 결과 출력
    print("\n=== Results ===")
    if result["status"] == "success":
        print(f"Text: {result['text']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Processing Time: {result['processingTime_ms']}ms")
    else:
        print(f"Error: {result['message']}")
    
    return result

# 실행
if __name__ == "__main__":
    # 테스트 음성 파일 경로
    testAudioPath = "test_audio.wav"
    
    # 파일럿 테스트 실행
    result = runPilotTest(testAudioPath, mode="hybrid")
    
    # 여러 테스트 파일로 실행
    testFiles = [
        "test_audio_whisper_1.wav",
        "test_audio_whisper_2.wav",
        "test_audio_normal_speech.wav"
    ]
    
    print("\n=== Running Multiple Tests ===")
    for testFile in testFiles:
        if os.path.exists(testFile):
            print(f"\nTesting: {testFile}")
            result = runPilotTest(testFile)
        else:
            print(f"Skipping: {testFile} (not found)")
```

### 파일과 블루투스 동시 테스트

```python
# integration_test.py - 블루투스와 파일 모두 테스트

def compareInputSources(audioFilePath: str, bluetoothDeviceName: str):
    """
    같은 파일을 두 가지 입력 소스로 처리 후 결과 비교
    
    실제 배포: 블루투스
    파일럿 테스트: 같은 파일 사용
    """
    
    config = loadConfig("config.json")
    
    # 1. 파일 입력 (파일럿 테스트)
    print("=== File Input Test ===")
    fileProvider = FileAudioProvider(filePath=audioFilePath)
    pipeline_file = SpeechToTextPipeline(fileProvider, config)
    result_file = pipeline_file.processAudio()
    print(f"File Result: {result_file['text']}")
    
    # 2. 블루투스 입력 (실제 테스트)
    print("\n=== Bluetooth Input Test ===")
    bluetoothProvider = BluetoothAudioProvider(deviceName=bluetoothDeviceName)
    pipeline_bt = SpeechToTextPipeline(bluetoothProvider, config)
    result_bt = pipeline_bt.processAudio()
    print(f"Bluetooth Result: {result_bt['text']}")
    
    # 3. 결과 비교
    print("\n=== Comparison ===")
    print(f"File Confidence: {result_file.get('confidence', 'N/A')}")
    print(f"BT Confidence: {result_bt.get('confidence', 'N/A')}")
    
    if result_file['text'] == result_bt['text']:
        print("✓ Results match!")
    else:
        print("✗ Results differ")

# 실행
if __name__ == "__main__":
    compareInputSources(
        audioFilePath="test_audio.wav",
        bluetoothDeviceName="Whisper Device"
    )
```

---

## 설정 관리 (Configuration)

```json
{
  "inputSource": {
    "type": "file",
    "filePath": "/path/to/test_audio.wav"
  },
  "audioInput": {
    "maxDuration_seconds": 300,
    "supportedFormats": ["PCM", "WAV"],
    "timeout_ms": 5000
  },
  "preprocessing": {
    "targetSampleRate": 16000,
    "chunkSize_seconds": 2,
    "chunkOverlap_ms": 500,
    "preEmphasisCoef": 0.97,
    "noiseRemovalAlpha": 3.0,
    "noiseRemovalBeta": 0.001,
    "targetRMS": 0.3,
    "vadThreshold": 0.05,
    "maxAudioDuration_seconds": 300,
    "memoryLimit_MB": 150
  },
  "stt": {
    "model": "whisper",
    "modelSize": "base",
    "language": "ko",
    "confidenceThreshold": 0.7
  },
  "pipeline": {
    "processingMode": "hybrid",
    "streamingChunkSize_ms": 1000,
    "streamingInterval_ms": 500,
    "timeout_ms": 30000
  },
  "logging": {
    "level": "INFO",
    "maxFileSize_MB": 10,
    "maxBackups": 5
  }
}
```

---

## 개발 단계

### Phase 1: 아키텍처 설계 (1주)
- ✓ 플랫폼 선정 (iOS/Android)
- ✓ 입력 소스 추상화 (IAudioInputProvider)
- ✓ 모듈 인터페이스 정의
- ✓ Strategy Pattern으로 모델 독립성 확보
- ✓ 에러 처리 전략 수립

### Phase 2: 입력 소스 & 전처리 모듈 (1-2주)
- BluetoothAudioProvider 구현
- FileAudioProvider 구현
- AudioInputModule 개발
- 속삭임 특화 전처리 알고리즘 구현
- Chunked Processing 최적화
- **파일럿 테스트 가능**

### Phase 3: STTModule 통합 (1-2주)
- WhisperModel 또는 VoskModel 선택 및 통합
- ISTTModel 인터페이스 구현
- 모바일 환경 최적화 (메모리, CPU)
- 성능 테스트

### Phase 4: ResultProcessingModule & Pipeline (1주)
- 결과 처리 및 포맷팅
- Hierarchical Error Handling 구현
- Hybrid/Batch/Streaming 모드 구현
- 통합 테스트

### Phase 5: 최적화 및 배포 준비 (1주)
- 성능 프로파일링
- 배터리 및 메모리 최적화
- 다양한 음성 샘플로 테스트
- **BluetoothAudioProvider로 실제 하드웨어 테스트**
- 파인튜닝 모델 교체 테스트 (CustomWhisperModel)

**총 예상 기간**: 4-6주

---

## 테스트 전략

### Phase 1: 파일 기반 단위 테스트 (파일럿)
```python
test_fileAudioProvider_load()
test_fileAudioProvider_invalid_format()
test_audioInputModule_with_file()
test_preprocessing_with_file()
test_full_pipeline_with_file()
```

### Phase 2: 실제 하드웨어 통합 테스트
```python
test_bluetoothAudioProvider_connect()
test_bluetoothAudioProvider_disconnect()
test_audioInputModule_with_bluetooth()
test_full_pipeline_with_bluetooth()
```

### Phase 3: 동일성 검증
```python
# 같은 음성을 파일과 블루투스로 입력받아 결과 비교
test_file_vs_bluetooth_consistency()
```

---

## 모듈 간 의존성

```
┌─────────────────┐
│   UI Layer      │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────┐
│  SpeechToTextPipeline        │
│  (Orchestrator)              │
└──────────────────────────────┘
         │
         ├─ IAudioInputProvider
         │  ├─ BluetoothAudioProvider
         │  └─ FileAudioProvider
         │
         ├─ AudioInputModule
         ├─ AudioPreprocessingModule
         ├─ STTModule (ISTTModel)
         │  ├─ WhisperModel
         │  ├─ VoskModel
         │  └─ CustomWhisperModel
         └─ ResultProcessingModule
```

**설계 원칙:**
- ✓ 입력 소스 추상화 (IAudioInputProvider)
- ✓ 각 모듈 독립적으로 테스트 가능
- ✓ 파일 입력으로 파일럿 테스트 가능
- ✓ 블루투스 입력으로 실제 배포 가능
- ✓ Pipeline 코드는 입력 소스와 무관

---

## 주요 고려사항

### 성능
- ✓ **레이턴시**: 음성 입력부터 텍스트 출력까지 < 2초 (Hybrid 모드)
- ✓ **정확도**: 속삭임 음성 인식 정확도 90% 이상
- ✓ **리소스**: 메모리 < 200MB, 배터리 효율적 사용

### 호환성
- ✓ 다양한 입력 소스 지원 (Bluetooth, WAV, MP3 등)
- ✓ 다양한 속삭임 특성 지원
- ✓ 배경 노이즈 환경에서의 안정성
- ✓ 다양한 기기(iOS/Android)에서의 일관성

### 안정성
- ✓ Hierarchical Error Handling
- ✓ 입력 소스별 에러 처리
- ✓ 메모리 부족 상황 처리 (Chunked Processing)
- ✓ 우아한 실패 (Graceful Degradation)

### 확장성
- ✓ 새로운 입력 소스 추가 용이 (IAudioInputProvider 구현)
- ✓ 새로운 STT 모델 추가 용이 (ISTTModel 구현)
- ✓ 파인튜닝 모델 추가 시 한 줄로 교체 가능

---

## 파인튜닝 모델 추가 (향후)

### 1단계: 데이터 수집 및 모델 학습
```python
# Fine-tuning Whisper with custom dataset
python -m openai.whisper --model=base --task=transcribe \
  --language=ko --device=cuda \
  --output_dir=/path/to/finetuned_model \
  /path/to/training_data
```

### 2단계: 모델 통합 (한 줄 변경!)
```python
# 기존 코드
sttModule = STTModule(WhisperModel(modelSize="base"))

# 파인튜닝 후
sttModule = STTModule(CustomWhisperModel(modelPath="/path/to/finetuned_model"))

# 나머지 코드는 변경 없음!
result = pipeline.processAudio()
```

---

## 성공 지표

- ✓ 속삭임 음성 인식 정확도 90% 이상
- ✓ 음성 입력부터 텍스트 출력까지 지연 시간 < 2초
- ✓ 배경 노이즈 환경에서 안정적 동작
- ✓ 메모리 사용량 < 200MB
- ✓ 배터리 소비 최소화 (1시간 연속 사용 시 배터리 10% 이상 소비)
- ✓ **파일 입력으로 파일럿 테스트 가능**
- ✓ **블루투스 입력으로 실제 배포 가능**
- ✓ 모듈 간 느슨한 결합 구조 (Strategy Pattern)
- ✓ 각 모듈 단위 테스트 커버리지 > 80%
- ✓ 파인튜닝 모델 교체 시 최소 코드 변경

---

## 참고 자료

- **Whisper**: https://github.com/openai/whisper
- **Vosk**: https://alphacephei.com/vosk/
- **librosa** (음성 처리): https://librosa.org/
- **Core Audio** (iOS): https://developer.apple.com/av-foundation/
- **MediaPlayer** (Android): https://developer.android.com/guide/topics/media

---

**문서 버전**: 3.0  
**작성일**: 2026-02-20  
**상태**: 최종 (입력 소스 추상화 및 파일럿 테스트 지원 완료)