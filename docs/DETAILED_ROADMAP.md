# 🗓️ 속삭임 감지 음성 변환 시스템 - 6주 상세 로드맵

---

## 📋 전체 일정 요약

```
Week 1: 입력/전처리 모듈 + 파일럿 테스트     (Day 1-5)
Week 2: STT 모듈 + 통합 테스트             (Day 6-10)
Week 3: 블루투스 모듈 + 실제 테스트        (Day 11-15)
Week 4-5: 모바일 앱 개발                  (Day 16-25)
Week 6: 최적화 및 파인튜닝                (Day 26-30)
```

---

# 📅 WEEK 1: 입력/전처리 모듈 + 파일럿 테스트

**목표**: FileAudioProvider와 AudioPreprocessingModule 완성, 파일럿 테스트 성공

**산출물**:
- ✅ FileAudioProvider 구현
- ✅ AudioInputModule 구현
- ✅ AudioPreprocessingModule 구현
- ✅ 단위 테스트 (커버리지 > 80%)
- ✅ 파일럿 테스트 성공

---

## 📅 Day 1 (Monday) - 프로젝트 초기화 & 환경 설정

### 아침 (09:00-12:00)

#### 1. 프로젝트 구조 생성
```bash
# 1-1. 프로젝트 폴더 생성
python init_project.py

# 1-2. 의존성 설치
bash scripts/setup_dev_env.sh
source venv/bin/activate
```

#### 2. 환경 확인
```bash
# 2-1. Python 버전 확인
python --version
# 기대값: Python 3.8 이상

# 2-2. 가상환경 활성화 확인
which python
# 기대값: /path/to/venv/bin/python

# 2-3. 주요 패키지 확인
python -c "import librosa, numpy, scipy, pytest; print('✓ OK')"
```

#### 3. 기본 설정
```bash
# 3-1. Git 초기화
git init
git add .
git commit -m "Initial project structure"

# 3-2. .env 파일 생성
cp .env.example .env
# .env 파일 수정 (필요시)

# 3-3. VS Code 설정
mkdir -p .vscode
# .vscode/settings.json 생성 (아래 참고)
```

### 오후 (13:00-17:00)

#### 4. 모듈 코드 구현 시작
```bash
# 4-1. module_code_templates.md의 코드 복사 시작
# backend/stt_core/audio_input/audio_data.py
# → AudioData 클래스 코드 붙여넣기

# 4-2. 코드 포맷팅 (black)
black backend/stt_core/audio_input/audio_data.py

# 4-3. 타입 체크 (mypy)
mypy backend/stt_core/audio_input/audio_data.py
```

#### 5. 문서 검토
- ✅ project_directory_structure.md 읽기
- ✅ 전체 구조 이해하기
- ✅ 각 모듈의 책임 파악

### 체크리스트
```
✅ init_project.py 실행 완료
✅ 의존성 설치 완료
✅ Git 초기화 완료
✅ 테스트 음성 파일 생성 (generate_test_audio.py 실행)
✅ AudioData 클래스 구현 시작
```

---

## 📅 Day 2 (Tuesday) - AudioData & FileAudioProvider 구현

### 아침 (09:00-12:00)

#### 1. AudioData 클래스 완성
```python
# backend/stt_core/audio_input/audio_data.py

# 작업 항목:
- AudioData 데이터클래스 구현
- __post_init__에서 duration_ms 자동 계산
- get_chunks() 메서드 구현 (청킹)
- get_duration_seconds() 메서드 구현
- 커스텀 예외 클래스 정의:
  - InvalidAudioFormatError
  - CorruptedAudioDataError
  - AudioProcessingError

# 테스트
python -m pytest backend/tests/unit/test_audio_data.py -v
```

#### 2. 기본 예외 클래스 구현
```python
# backend/stt_core/input_providers/exceptions.py

- ConnectionError (상속)
- TimeoutError (상속)
- BluetoothConnectionError
- InvalidAudioFormatError
```

### 오후 (13:00-17:00)

#### 3. IAudioInputProvider 인터페이스 구현
```python
# backend/stt_core/input_providers/base.py

# 메서드 정의:
- connect() -> bool
- disconnect() -> None
- is_connected() -> bool
- receive_audio() -> AudioData
- get_source_info() -> Dict
```

#### 4. FileAudioProvider 구현
```python
# backend/stt_core/input_providers/file_provider.py

# 구현 항목:
- __init__(file_path, mono=True)
- connect(): librosa로 파일 로드
  - 샘플 레이트 유지
  - 채널 정보 추출
  - 로깅 추가
- disconnect(): 리소스 해제
- is_connected(): 상태 반환
- receive_audio(): AudioData 생성 및 반환
- get_source_info(): 메타데이터 반환

# 테스트
python -m pytest backend/tests/unit/test_file_provider.py -v
```

### 체크리스트
```
✅ AudioData 클래스 완성 및 테스트
✅ 예외 클래스 정의
✅ IAudioInputProvider 인터페이스 정의
✅ FileAudioProvider 기본 구현
✅ FileAudioProvider 테스트 작성
```

---

## 📅 Day 3 (Wednesday) - AudioInputModule & 입력 검증

### 아침 (09:00-12:00)

#### 1. AudioInputModule 구현
```python
# backend/stt_core/audio_input/audio_input_module.py

# 구현 항목:
- SUPPORTED_SAMPLE_RATES 정의: [8000, 16000, 44100, 48000]
- SUPPORTED_FORMATS 정의: ["PCM", "WAV", "MP3"]
- __init__(audio_provider)
- connect() -> bool
- disconnect() -> None
- is_connected() -> bool
- receive_and_parse() -> AudioData
  - 데이터 수신
  - 검증
  - 로깅
- _validate_audio_data():
  - 샘플 레이트 확인
  - 포맷 확인
  - 데이터 길이 확인
  - 무음 감지 (모두 0인지 확인)
  - 채널 수 확인
- get_source_info() -> Dict
```

#### 2. 입력 검증 로직 테스트
```bash
# test_audio_input_module.py 작성
- test_valid_audio_acceptance()
- test_invalid_sample_rate_rejection()
- test_empty_audio_rejection()
- test_all_zero_audio_warning()
- test_invalid_channels_rejection()
```

### 오후 (13:00-17:00)

#### 3. 파일 입력 통합 테스트
```python
# backend/tests/integration/test_full_pipeline_file.py

# 테스트 케이스:
1. FileAudioProvider 초기화
   - 파일 존재 확인
   - 파일 로드 성공
   - 메타데이터 올바름

2. AudioInputModule 통합
   - connect() 성공
   - receive_and_parse() 성공
   - 데이터 검증 통과

3. 에러 처리
   - 파일 없음 처리
   - 손상된 파일 처리
   - 타임아웃 처리
```

#### 4. 로깅 설정
```python
# backend/stt_core/config.py 에 로깅 설정 추가

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### 체크리스트
```
✅ AudioInputModule 완성
✅ 입력 검증 로직 구현
✅ 단위 테스트 (커버리지 > 80%)
✅ 통합 테스트 (파일 입력)
✅ 로깅 설정 완료
```

---

## 📅 Day 4 (Thursday) - AudioPreprocessingModule 구현

### 아침 (09:00-12:00)

#### 1. 전처리 모듈 기본 구조
```python
# backend/stt_core/preprocessing/preprocessing_module.py

# 클래스: AudioPreprocessingModule

# 메서드:
- __init__(config: Dict)
- process(audio_data: AudioData) -> np.ndarray
  - 청킹 (CHUNK_SIZE 단위)
  - 각 청크에 대해:
    1. 샘플 레이트 정규화
    2. 고주파 강화
    3. 노이즈 제거
    4. VAD 실행
    5. RMS 정규화

- _normalize_sample_rate(audio, target_sr)
- _emphasize_high_frequency(audio, coef)
- _remove_noise(audio, alpha, beta)
- _voice_activity_detection(audio, threshold)
- _normalize_rms(audio, target_rms)
```

### 오후 (13:00-17:00)

#### 2. 각 전처리 알고리즘 구현
```python
# backend/stt_core/preprocessing/normalization.py
- normalize_sample_rate(audio, sr_original, sr_target)
  librosa.resample() 사용
- normalize_rms(audio, target_rms=0.3)
  RMS 계산: sqrt(mean(x^2))

# backend/stt_core/preprocessing/filters.py
- pre_emphasis_filter(audio, coef=0.97)
  y[n] = x[n] - coef * x[n-1]

# backend/stt_core/preprocessing/noise_removal.py
- spectral_subtraction(audio, sr, alpha=3.0, beta=0.001)
  STFT → 파워 스펙트럼 → 노이즈 추정 → 감산 → ISTFT

# backend/stt_core/preprocessing/voice_activity.py
- detect_voice_activity(audio, threshold=0.05)
  에너지 기반 또는 스펙트럼 기반 VAD
```

#### 3. 속삭임 특화 설정
```python
# backend/config/config.json 업데이트

"preprocessing": {
    "targetSampleRate": 16000,
    "chunkSize_seconds": 2,
    "chunkSize_samples": 32000,  # 2초 * 16kHz
    "chunkOverlap_ms": 500,
    
    # 고주파 강화 (속삭임은 고주파 많음)
    "preEmphasisCoef": 0.97,
    
    # 노이즈 제거
    "noiseRemovalAlpha": 3.0,    # 감산 강도
    "noiseRemovalBeta": 0.001,   # 정규화
    
    # RMS 정규화
    "targetRMS": 0.3,            # 30% 볼륨
    
    # VAD (속삭임용 낮은 임계값)
    "vadThreshold": 0.05,
    
    # 메모리 관리
    "maxAudioDuration_seconds": 300,
    "memoryLimit_MB": 150
}
```

### 체크리스트
```
✅ AudioPreprocessingModule 기본 구조
✅ 샘플 레이트 정규화
✅ 고주파 강화 필터
✅ 스펙트럼 감산 노이즈 제거
✅ VAD (음성 활동 감지)
✅ RMS 정규화
```

---

## 📅 Day 5 (Friday) - 전처리 테스트 & 파일럿 완성

### 아침 (09:00-12:00)

#### 1. 전처리 모듈 테스트
```python
# backend/tests/unit/test_preprocessing_module.py

# 테스트 케이스:
- test_sample_rate_normalization()
  입력: 48kHz → 출력: 16kHz 확인
  
- test_high_frequency_emphasis()
  고주파 성분 증가 확인
  
- test_noise_removal()
  노이즈 감소 확인
  
- test_voice_activity_detection()
  음성 구간 식별
  
- test_rms_normalization()
  RMS 값 확인
  
- test_chunked_processing()
  청킹 처리 (메모리 효율)
  
- test_attribute_error_handling()
  잘못된 입력 처리
```

#### 2. 파이프라인 기본 구조
```python
# backend/stt_core/pipeline/speech_to_text_pipeline.py

class SpeechToTextPipeline:
    def __init__(self, audio_provider, config):
        self.audio_provider = audio_provider
        self.config = config
        self.audio_input_module = AudioInputModule(audio_provider)
        self.preprocessing_module = AudioPreprocessingModule(config)
    
    def process_audio(self, mode="hybrid"):
        """
        Phase 1: 연결
        Phase 2: 입력 처리
        Phase 3: 전처리
        Phase 4: STT (Week 2에서 구현)
        Phase 5: 결과 처리 (Week 2에서 구현)
        """
        # TODO: Week 2에서 구현
```

### 오후 (13:00-17:00)

#### 3. 파일럿 테스트 실행
```bash
# 3-1. 테스트 음성 파일 생성 (존재하지 않으면)
python scripts/generate_test_audio.py

# 3-2. 파일럿 테스트 스크립트 작성
# scripts/run_pilot_test.py 업데이트

# 3-3. 파일럿 테스트 실행
python scripts/run_pilot_test.py
```

#### 4. 성능 측정
```python
# scripts/profile_performance.py 작성

import time
import psutil

def profile_preprocessing(audio_path):
    # 메모리 사용량 측정
    # 처리 시간 측정
    # CPU 사용량 측정
    # 결과 리포트
```

#### 5. Week 1 최종 검증
```bash
# 5-1. 전체 테스트 실행
pytest backend/tests/ -v --cov=backend/stt_core --cov-report=html

# 5-2. 커버리지 확인
# htmlcov/index.html 열기
# 목표: > 80%

# 5-3. 코드 품질 확인
black backend/
flake8 backend/
mypy backend/

# 5-4. Git 커밋
git add .
git commit -m "Week 1: Input and preprocessing modules completed"
git push origin main
```

### 체크리스트
```
✅ 전처리 모듈 테스트 완료 (커버리지 > 80%)
✅ 파이프라인 기본 구조
✅ 파일럿 테스트 성공
✅ 성능 측정 완료
✅ 코드 품질 확인 (black, flake8, mypy)
✅ Git 커밋 완료
✅ Week 1 완료!

📊 Week 1 성과:
   - 라인 수: ~1500
   - 테스트: ~30개
   - 커버리지: >80%
   - 파일럿 테스트: ✅ PASS
```

---

# 📅 WEEK 2: STT 모듈 + 통합 테스트

**목표**: WhisperModel/VoskModel 통합, 전체 파이프라인 동작

**산출물**:
- ✅ ISTTModel 인터페이스
- ✅ WhisperModel 구현
- ✅ VoskModel 구현 (선택)
- ✅ STTModule 구현
- ✅ ResultProcessingModule 구현
- ✅ SpeechToTextPipeline 완성
- ✅ 통합 테스트 통과

---

## 📅 Day 6 (Monday) - STT 모듈 기본 구조

### 아침 (09:00-12:00)

#### 1. STT 모델 인터페이스 정의
```python
# backend/stt_core/stt_models/base.py

from abc import ABC, abstractmethod

class ISTTModel(ABC):
    """모든 STT 모델이 구현해야 할 인터페이스"""
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """모델 로드"""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """모델 언로드 (메모리 해제)"""
        pass
    
    @abstractmethod
    def transcribe(self, audio_data: AudioData) -> Dict:
        """
        음성 인식 수행
        
        Returns:
            {
                "text": "인식된 텍스트",
                "confidence": 0.95,
                "language": "ko",
                "processing_time_ms": 1500
            }
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        pass
```

#### 2. STTModule 구현 (오케스트레이터)
```python
# backend/stt_core/stt_models/stt_module.py

class STTModule:
    def __init__(self, model: ISTTModel):
        self.model = model
    
    def inference(self, audio_data: AudioData) -> Dict:
        """
        STT 추론 실행
        
        Returns:
            {
                "status": "success",
                "text": "결과",
                "confidence": 0.95,
                ...
            }
        """
        try:
            result = self.model.transcribe(audio_data)
            result["status"] = "success"
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
```

#### 3. 모델 팩토리 패턴
```python
# backend/stt_core/stt_models/model_factory.py

class STTModelFactory:
    @staticmethod
    def create(model_type: str, **kwargs) -> ISTTModel:
        """
        모델 생성
        
        Args:
            model_type: "whisper" 또는 "vosk"
            **kwargs: 모델별 파라미터
        
        Returns:
            ISTTModel 구현체
        """
        if model_type.lower() == "whisper":
            return WhisperModel(**kwargs)
        elif model_type.lower() == "vosk":
            return VoskModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# 사용 예시:
# model = STTModelFactory.create("whisper", size="base")
# stt_module = STTModule(model)
```

### 오후 (13:00-17:00)

#### 4. Whisper 모델 통합
```python
# backend/stt_core/stt_models/whisper_model.py

import whisper

class WhisperModel(ISTTModel):
    def __init__(self, model_size: str = "base", language: str = "ko"):
        self.model_size = model_size
        self.language = language
        self.model = None
    
    def load_model(self, model_path: str = None) -> None:
        """
        Whisper 모델 로드
        
        Args:
            model_path: 커스텀 모델 경로 (선택)
                       None이면 기본 모델 다운로드
        """
        if model_path:
            # 커스텀 모델 로드
            self.model = whisper.load_model(model_path)
        else:
            # 기본 모델 다운로드 (tiny/base/small/medium/large)
            self.model = whisper.load_model(self.model_size)
    
    def unload_model(self) -> None:
        """메모리에서 모델 언로드"""
        if self.model:
            del self.model
            self.model = None
    
    def transcribe(self, audio_data: AudioData) -> Dict:
        """
        음성 인식 수행
        
        Args:
            audio_data: AudioData 객체
        
        Returns:
            {
                "text": "인식된 텍스트",
                "confidence": 0.95,
                "language": "ko",
                "processing_time_ms": 1500
            }
        """
        import time
        
        start_time = time.time()
        
        try:
            # Whisper 추론
            result = self.model.transcribe(
                audio_data.audio,
                language=self.language,
                fp16=False  # GPU 없으면 False
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "text": result.get("text", ""),
                "confidence": result.get("segments", [{}])[0].get("confidence", 0.0),
                "language": result.get("language", self.language),
                "processing_time_ms": int(processing_time),
                "raw_result": result
            }
        
        except Exception as e:
            raise Exception(f"Whisper transcription failed: {e}")
    
    def get_model_info(self) -> Dict:
        return {
            "model_type": "whisper",
            "model_size": self.model_size,
            "language": self.language,
            "loaded": self.model is not None
        }
```

### 체크리스트
```
✅ ISTTModel 인터페이스 정의
✅ STTModule 구현
✅ STTModelFactory 구현
✅ WhisperModel 통합
✅ 기본 테스트 작성
```

---

## 📅 Day 7 (Tuesday) - WhisperModel 상세 구현 & 최적화

### 아침 (09:00-12:00)

#### 1. WhisperModel 최적화
```python
# backend/stt_core/stt_models/whisper_model.py 수정

class WhisperModel(ISTTModel):
    def __init__(
        self,
        model_size: str = "base",
        language: str = "ko",
        device: str = "cpu",
        fp16: bool = False
    ):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.fp16 = fp16
        self.model = None
    
    def transcribe(self, audio_data: AudioData) -> Dict:
        """속삭임 특화 옵션들"""
        
        # 속삭임 특화 파라미터
        result = self.model.transcribe(
            audio_data.audio,
            language=self.language,
            fp16=self.fp16,
            verbose=False,
            temperature=0.2,  # 낮은 온도 = 더 확정적
            best_of=5,        # 5개 시도 중 최고 선택
            beam_size=5,      # 빔 서치 크기
            no_speech_threshold=0.6,  # 속삭임용 높은 임계값
        )
        
        return {
            "text": result.get("text", "").strip(),
            "confidence": self._calculate_confidence(result),
            "language": result.get("language", self.language),
            "processing_time_ms": int(processing_time),
            "raw_result": result
        }
    
    def _calculate_confidence(self, result: Dict) -> float:
        """신뢰도 계산"""
        segments = result.get("segments", [])
        if not segments:
            return 0.0
        
        # 평균 신뢰도
        confidences = [seg.get("confidence", 0.0) for seg in segments]
        return sum(confidences) / len(confidences) if confidences else 0.0
```

#### 2. 모델 성능 벤치마크
```python
# scripts/benchmark_stt.py

def benchmark_whisper_models():
    """Whisper 모델 성능 비교"""
    
    model_sizes = ["tiny", "base", "small", "medium"]
    test_audio = "test_audio.wav"
    
    results = {}
    
    for size in model_sizes:
        model = WhisperModel(model_size=size)
        model.load_model()
        
        # 성능 측정
        start = time.time()
        result = model.transcribe(audio_data)
        elapsed = time.time() - start
        
        results[size] = {
            "size_mb": get_model_size(size),
            "time_sec": elapsed,
            "text": result["text"],
            "confidence": result["confidence"]
        }
        
        model.unload_model()
    
    # 결과 비교
    print_benchmark_results(results)
```

### 오후 (13:00-17:00)

#### 3. Vosk 모델 구현 (선택사항)
```python
# backend/stt_core/stt_models/vosk_model.py

from vosk import Model, KaldiRecognizer

class VoskModel(ISTTModel):
    def __init__(self, model_path: str = "models/vosk/korean"):
        self.model_path = model_path
        self.model = None
    
    def load_model(self, model_path: str = None) -> None:
        """Vosk 모델 로드"""
        path = model_path or self.model_path
        self.model = Model(path)
    
    def transcribe(self, audio_data: AudioData) -> Dict:
        """Vosk로 음성 인식"""
        rec = KaldiRecognizer(self.model, audio_data.sample_rate)
        
        # 청크 단위로 처리
        results = []
        for chunk in audio_data.get_chunks(4000):  # 0.25초
            rec.AcceptWaveform(chunk.tobytes())
            result = json.loads(rec.Result())
            if "result" in result:
                results.append(result["result"])
        
        # 최종 결과
        final = json.loads(rec.FinalResult())
        text = " ".join([r.get("conf", 0) for r in results])
        
        return {
            "text": text,
            "confidence": 0.0,  # Vosk는 신뢰도 제공 안 함
            "language": "ko",
            "processing_time_ms": 0
        }
```

#### 4. STT 테스트 작성
```python
# backend/tests/unit/test_stt_models.py

def test_whisper_model_load():
    model = WhisperModel(model_size="base")
    model.load_model()
    assert model.model is not None
    model.unload_model()

def test_whisper_transcription(sample_audio_path):
    model = WhisperModel(model_size="base")
    model.load_model()
    
    audio_data = FileAudioProvider(sample_audio_path).receive_audio()
    result = model.transcribe(audio_data)
    
    assert result["status"] == "success"
    assert "text" in result
    assert 0 <= result["confidence"] <= 1
    
    model.unload_model()

def test_stt_module_with_whisper(sample_audio_path):
    model = WhisperModel(model_size="base")
    stt_module = STTModule(model)
    
    audio_data = FileAudioProvider(sample_audio_path).receive_audio()
    result = stt_module.inference(audio_data)
    
    assert result["status"] == "success"
```

### 체크리스트
```
✅ WhisperModel 최적화
✅ 신뢰도 계산 로직
✅ 성능 벤치마크 스크립트
✅ Vosk 모델 구현 (선택)
✅ STT 모듈 테스트
```

---

## 📅 Day 8 (Wednesday) - ResultProcessingModule 구현

### 아침 (09:00-12:00)

#### 1. 결과 처리 모듈
```python
# backend/stt_core/result_processing/result_processing_module.py

class ResultProcessingModule:
    def __init__(self, config: Dict):
        self.config = config
        self.confidence_threshold = config.get("stt", {}).get("confidenceThreshold", 0.7)
    
    def process(self, stt_result: Dict) -> Dict:
        """
        STT 결과 처리 및 포맷팅
        
        Args:
            stt_result: STTModule.inference()의 결과
        
        Returns:
            {
                "status": "success" | "low_confidence" | "error",
                "text": "최종 텍스트",
                "confidence": 0.95,
                "timestamp": "2026-03-09T10:30:00",
                "processing_time_ms": 1500,
                "message": "오류 메시지 (있으면)"
            }
        """
        
        if stt_result.get("status") != "success":
            return {
                "status": "error",
                "text": "",
                "confidence": 0.0,
                "message": stt_result.get("message", "Unknown error"),
                "timestamp": datetime.now().isoformat()
            }
        
        text = stt_result.get("text", "").strip()
        confidence = stt_result.get("confidence", 0.0)
        
        # 신뢰도 확인
        if confidence < self.confidence_threshold:
            status = "low_confidence"
            text = f"[Low Confidence: {confidence:.2f}] {text}"
        else:
            status = "success"
        
        return {
            "status": status,
            "text": text,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": stt_result.get("processing_time_ms", 0),
            "language": stt_result.get("language", "ko")
        }
    
    def format_for_display(self, result: Dict) -> str:
        """사용자 표시용 포맷팅"""
        
        text = result.get("text", "")
        confidence = result.get("confidence", 0.0)
        
        if result["status"] == "error":
            return f"❌ 오류: {result.get('message', 'Unknown error')}"
        elif result["status"] == "low_confidence":
            return f"⚠️ 낮은 신뢰도 ({confidence:.1%}): {text}"
        else:
            return f"✅ ({confidence:.1%}): {text}"
    
    def format_for_logging(self, result: Dict) -> str:
        """로깅용 포맷팅"""
        return json.dumps(result, ensure_ascii=False, indent=2)
```

### 오후 (13:00-17:00)

#### 2. 신뢰도 계산기
```python
# backend/stt_core/result_processing/confidence_calculator.py

class ConfidenceCalculator:
    """음성 인식 신뢰도 계산"""
    
    @staticmethod
    def calculate_from_whisper(result: Dict) -> float:
        """Whisper 결과에서 신뢰도 추출"""
        segments = result.get("segments", [])
        if not segments:
            return 0.0
        
        # 평균 확률로 신뢰도 계산
        confidences = []
        for segment in segments:
            tokens = segment.get("tokens", [])
            if tokens:
                conf = sum(t.get("probability", 0) for t in tokens) / len(tokens)
                confidences.append(conf)
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    @staticmethod
    def apply_temporal_smoothing(confidences: list) -> float:
        """시간 평활 필터 (현재 + 이전 결과)"""
        alpha = 0.7  # 평활 계수
        
        # 현재 신뢰도에 가중치
        current = confidences[-1] if confidences else 0.0
        previous = sum(confidences[:-1]) / len(confidences[:-1]) if len(confidences) > 1 else 0.0
        
        return alpha * current + (1 - alpha) * previous
```

#### 3. 포맷터
```python
# backend/stt_core/result_processing/formatters.py

class OutputFormatter:
    """다양한 포맷으로 결과 출력"""
    
    @staticmethod
    def to_json(result: Dict) -> str:
        """JSON 형식"""
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @staticmethod
    def to_csv(results: list) -> str:
        """CSV 형식 (여러 결과)"""
        if not results:
            return ""
        
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
        return output.getvalue()
    
    @staticmethod
    def to_human_readable(result: Dict) -> str:
        """사람이 읽기 쉬운 형식"""
        
        status_emoji = {
            "success": "✅",
            "low_confidence": "⚠️",
            "error": "❌"
        }
        
        emoji = status_emoji.get(result["status"], "❓")
        
        text = f"{emoji} {result.get('text', 'N/A')}\n"
        text += f"신뢰도: {result.get('confidence', 0.0):.1%}\n"
        text += f"시간: {result.get('timestamp', 'N/A')}\n"
        text += f"처리시간: {result.get('processing_time_ms', 0)}ms"
        
        return text
```

#### 4. 테스트
```python
# backend/tests/unit/test_result_processing.py

def test_result_formatting():
    module = ResultProcessingModule({
        "stt": {"confidenceThreshold": 0.7}
    })
    
    # 성공 케이스
    result = {
        "status": "success",
        "text": "안녕하세요",
        "confidence": 0.95
    }
    processed = module.process(result)
    assert processed["status"] == "success"
    
    # 신뢰도 낮음 케이스
    result["confidence"] = 0.5
    processed = module.process(result)
    assert processed["status"] == "low_confidence"
```

### 체크리스트
```
✅ ResultProcessingModule 구현
✅ 신뢰도 계산기
✅ 다양한 포맷터
✅ 테스트 작성
```

---

## 📅 Day 9 (Thursday) - 전체 파이프라인 구현

### 아침 (09:00-12:00)

#### 1. SpeechToTextPipeline 완성
```python
# backend/stt_core/pipeline/speech_to_text_pipeline.py

import logging
import time

logger = logging.getLogger(__name__)

class SpeechToTextPipeline:
    """전체 음성 인식 파이프라인 (오케스트레이터)"""
    
    def __init__(
        self,
        audio_provider: IAudioInputProvider,
        stt_model: ISTTModel,
        config: Dict
    ):
        self.audio_provider = audio_provider
        self.stt_model = stt_model
        self.config = config
        
        # 각 모듈 초기화
        self.audio_input_module = AudioInputModule(audio_provider)
        self.preprocessing_module = AudioPreprocessingModule(config)
        self.stt_module = STTModule(stt_model)
        self.result_processing_module = ResultProcessingModule(config)
        
        logger.info("Pipeline initialized")
    
    def process_audio(self, mode: str = "hybrid") -> Dict:
        """
        전체 파이프라인 실행
        
        Args:
            mode: "batch" (한 번에) / "streaming" (실시간) / "hybrid" (혼합)
        
        Returns:
            {
                "status": "success",
                "text": "인식된 텍스트",
                "confidence": 0.95,
                ...
            }
        """
        
        logger.info(f"Starting pipeline (mode: {mode})")
        start_time = time.time()
        
        try:
            # Phase 1: 입력 소스에 연결
            logger.info("Phase 1: Connecting to audio source...")
            if not self.audio_input_module.connect():
                return self._error_result("CONNECTION_FAILED", "Failed to connect to audio source")
            
            # Phase 2: 음성 데이터 수신 및 검증
            logger.info("Phase 2: Receiving audio data...")
            audio_data = self.audio_input_module.receive_and_parse()
            logger.info(f"Received {audio_data.duration_ms:.0f}ms of audio")
            
            # Phase 3: 전처리
            logger.info("Phase 3: Preprocessing audio...")
            processed_audio = self.preprocessing_module.process(audio_data)
            logger.info(f"Preprocessing completed, output shape: {processed_audio.shape}")
            
            # Phase 4: STT 추론
            logger.info("Phase 4: Running STT inference...")
            
            if mode == "batch":
                stt_result = self._batch_mode(processed_audio)
            elif mode == "streaming":
                stt_result = self._streaming_mode(processed_audio)
            else:  # hybrid
                stt_result = self._hybrid_mode(processed_audio)
            
            # Phase 5: 결과 처리
            logger.info("Phase 5: Processing results...")
            final_result = self.result_processing_module.process(stt_result)
            
            # 처리 시간 추가
            final_result["total_processing_time_ms"] = int((time.time() - start_time) * 1000)
            
            logger.info(f"Pipeline completed: {final_result['status']}")
            
            return final_result
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return self._error_result("PIPELINE_ERROR", str(e))
        
        finally:
            # Phase 6: 정리
            logger.info("Disconnecting from audio source...")
            self.audio_input_module.disconnect()
    
    def _batch_mode(self, audio: np.ndarray) -> Dict:
        """배치 모드: 전체 음성을 한 번에 처리"""
        logger.info("Batch mode: Processing entire audio at once")
        
        # AudioData 래핑
        audio_data = AudioData(
            audio=audio,
            sample_rate=self.config["preprocessing"]["targetSampleRate"],
            channels=1,
            bit_depth=32,
            format="PCM"
        )
        
        return self.stt_module.inference(audio_data)
    
    def _streaming_mode(self, audio: np.ndarray) -> Dict:
        """스트리밍 모드: 청크 단위로 처리"""
        logger.info("Streaming mode: Processing audio in chunks")
        
        chunk_size = 16000 * 1  # 1초
        results = []
        
        for i, chunk in enumerate(self._get_chunks(audio, chunk_size)):
            logger.debug(f"Processing chunk {i+1}")
            
            audio_data = AudioData(
                audio=chunk,
                sample_rate=16000,
                channels=1,
                bit_depth=32,
                format="PCM"
            )
            
            result = self.stt_module.inference(audio_data)
            results.append(result.get("text", ""))
        
        # 결과 병합
        combined_text = " ".join(results)
        
        return {
            "status": "success",
            "text": combined_text,
            "confidence": 0.0,  # 평균값 계산 필요
            "processing_time_ms": 0,
            "chunks_processed": len(results)
        }
    
    def _hybrid_mode(self, audio: np.ndarray) -> Dict:
        """하이브리드 모드: 배치 + 스트리밍 혼합"""
        logger.info("Hybrid mode: Using batch mode with streaming backup")
        
        # 먼저 배치 모드 시도
        try:
            audio_data = AudioData(
                audio=audio,
                sample_rate=16000,
                channels=1,
                bit_depth=32,
                format="PCM"
            )
            
            result = self.stt_module.inference(audio_data)
            
            if result.get("confidence", 0) < 0.5:
                logger.warning("Batch mode confidence too low, falling back to streaming")
                return self._streaming_mode(audio)
            
            return result
        
        except Exception as e:
            logger.warning(f"Batch mode failed, falling back to streaming: {e}")
            return self._streaming_mode(audio)
    
    def _get_chunks(self, audio: np.ndarray, chunk_size: int):
        """음성을 청크로 분할"""
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i+chunk_size]
    
    def _error_result(self, error_code: str, message: str) -> Dict:
        """에러 결과 생성"""
        return {
            "status": "error",
            "error_code": error_code,
            "message": message,
            "text": "",
            "confidence": 0.0
        }
```

### 오후 (13:00-17:00)

#### 2. 파이프라인 통합 테스트
```python
# backend/tests/integration/test_stt_integration.py

def test_full_pipeline_with_file_whisper(sample_audio_path):
    """파일 입력 + Whisper 모델 전체 파이프라인"""
    
    # 설정
    config = load_config("backend/config/config.json")
    
    # 입력 소스
    audio_provider = FileAudioProvider(sample_audio_path)
    
    # STT 모델
    stt_model = WhisperModel(model_size="base")
    stt_model.load_model()
    
    # 파이프라인
    pipeline = SpeechToTextPipeline(audio_provider, stt_model, config)
    
    # 실행
    result = pipeline.process_audio(mode="batch")
    
    # 검증
    assert result["status"] in ["success", "low_confidence"]
    assert "text" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    
    stt_model.unload_model()

def test_pipeline_error_handling():
    """에러 처리 테스트"""
    
    # 존재하지 않는 파일
    audio_provider = FileAudioProvider("nonexistent.wav")
    stt_model = WhisperModel(model_size="base")
    config = load_config("backend/config/config.json")
    
    pipeline = SpeechToTextPipeline(audio_provider, stt_model, config)
    result = pipeline.process_audio()
    
    assert result["status"] == "error"
    assert result["error_code"] == "CONNECTION_FAILED"
```

#### 3. 파이프라인 성능 측정
```python
# scripts/benchmark_pipeline.py

def benchmark_pipeline():
    """전체 파이프라인 성능 측정"""
    
    config = load_config("backend/config/config.json")
    test_audio = "test_audio.wav"
    
    # Whisper base 모델
    stt_model = WhisperModel(model_size="base")
    stt_model.load_model()
    
    audio_provider = FileAudioProvider(test_audio)
    pipeline = SpeechToTextPipeline(audio_provider, stt_model, config)
    
    # 5회 실행하여 평균 측정
    times = []
    for i in range(5):
        result = pipeline.process_audio()
        times.append(result["total_processing_time_ms"])
    
    print(f"평균 처리 시간: {sum(times)/len(times):.0f}ms")
    print(f"최소: {min(times)}ms, 최대: {max(times)}ms")
    
    stt_model.unload_model()
```

### 체크리스트
```
✅ SpeechToTextPipeline 완성
✅ 3가지 모드 구현 (batch/streaming/hybrid)
✅ 에러 처리
✅ 로깅
✅ 통합 테스트
✅ 성능 측정
```

---

## 📅 Day 10 (Friday) - Week 2 최종 정리 & 테스트

### 아침 (09:00-12:00)

#### 1. 전체 테스트 실행
```bash
# 1-1. 모든 테스트 실행
pytest backend/tests/ -v --cov=backend/stt_core --cov-report=html

# 1-2. 특정 영역 테스트
pytest backend/tests/unit/ -v                    # 단위 테스트
pytest backend/tests/integration/ -v            # 통합 테스트

# 1-3. 커버리지 확인
open htmlcov/index.html
# 목표: > 85%
```

#### 2. 코드 품질 확인
```bash
# 2-1. 포맷팅
black backend/ --check
black backend/ --diff

# 2-2. Linting
flake8 backend/ --max-line-length=100

# 2-3. 타입 체킹
mypy backend/stt_core/ --ignore-missing-imports
```

#### 3. 성능 벤치마크
```bash
python scripts/benchmark_pipeline.py
python scripts/benchmark_stt.py

# 출력 예상:
# - Whisper base: 1500-2000ms
# - Vosk: 500-1000ms
# - 메모리 사용: 150-300MB
```

### 오후 (13:00-17:00)

#### 4. 문서화
```bash
# 4-1. Week 2 완료 문서
cat > docs/WEEK_2_COMPLETION.md << 'EOF'
# Week 2 완료 보고서

## 구현된 기능
- ISTTModel 인터페이스
- WhisperModel 통합
- VoskModel 구현
- STTModule
- ResultProcessingModule
- SpeechToTextPipeline

## 테스트 결과
- 단위 테스트: 45개 통과
- 통합 테스트: 15개 통과
- 커버리지: 87%

## 성능
- 배치 모드: 1800ms
- 스트리밍 모드: 500-1000ms (청크당)
- 메모리: 280MB

## 향후 개선사항
- GPU 지원
- 스트리밍 개선
- 캐싱 추가
EOF
```

#### 5. 최종 커밋 & 배포
```bash
# 5-1. 모든 변경사항 커밋
git add .
git commit -m "Week 2: STT module and full pipeline completed"
git push origin develop

# 5-2. 태그 추가
git tag -a v0.2.0 -m "Week 2 completion: STT integration"
git push origin v0.2.0

# 5-3. 로그 생성
git log --oneline v0.1.0..v0.2.0 > CHANGELOG.md
```

#### 6. Week 3 준비
```bash
# 6-1. BluetoothAudioProvider 초안 검토
# 6-2. 하드웨어팀과 API 정의 논의
# 6-3. 테스트 계획 수립
```

### 체크리스트
```
✅ 모든 테스트 통과 (커버리지 > 85%)
✅ 코드 품질 확인 (black, flake8, mypy)
✅ 성능 벤치마크 완료
✅ 문서화 완료
✅ Git 커밋 및 태그
✅ Week 2 완료!

📊 Week 2 성과:
   - 라인 수: ~2500 (누적 4000)
   - 테스트: ~60개 (누적 90)
   - 커버리지: >85%
   - 성능: ✅ 목표 달성
```

---

# 📅 WEEK 3: 블루투스 모듈 + 실제 테스트

**목표**: BluetoothAudioProvider 구현, 실제 하드웨어 테스트

**산출물**:
- ✅ BluetoothAudioProvider 구현
- ✅ 블루투스 통신 테스트
- ✅ 파일 vs 블루투스 일관성 검증
- ✅ 실제 음성 인식 테스트

---

## 📅 Day 11 (Monday) - 블루투스 인터페이스 설계

### 아침 (09:00-12:00)

#### 1. 블루투스 통신 프로토콜 정의
```python
# backend/stt_core/input_providers/bluetooth_protocol.py

"""
블루투스 통신 프로토콜

모바일 앱 ←→ 백엔드

메시지 형식: JSON

1. 연결 요청
{
    "type": "CONNECT",
    "device_name": "Whisper Device",
    "timestamp": "2026-03-09T10:00:00"
}

2. 음성 데이터 전송
{
    "type": "AUDIO_CHUNK",
    "data": "base64_encoded_audio",
    "chunk_index": 0,
    "sample_rate": 16000,
    "channels": 1,
    "timestamp": "2026-03-09T10:00:01"
}

3. 음성 전송 완료
{
    "type": "AUDIO_END",
    "total_chunks": 5,
    "duration_ms": 2500,
    "timestamp": "2026-03-09T10:00:02"
}

4. 연결 해제
{
    "type": "DISCONNECT",
    "timestamp": "2026-03-09T10:00:03"
}
"""

class BluetoothMessage:
    def __init__(self, message_type: str, **kwargs):
        self.type = message_type
        self.timestamp = datetime.now().isoformat()
        self.data = kwargs
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "timestamp": self.timestamp,
            **self.data
        })
    
    @staticmethod
    def from_json(json_str: str) -> 'BluetoothMessage':
        data = json.loads(json_str)
        msg_type = data.pop("type")
        return BluetoothMessage(msg_type, **data)
```

#### 2. 블루투스 버퍼 관리
```python
# backend/stt_core/input_providers/bluetooth_buffer.py

import threading
from collections import deque

class BluetoothAudioBuffer:
    """블루투스로 받은 음성 데이터를 버퍼링"""
    
    def __init__(self, max_size_mb: int = 50):
        self.max_size = max_size_mb * 1024 * 1024  # 바이트
        self.buffer = deque()
        self.lock = threading.Lock()
        self.current_size = 0
    
    def add_chunk(self, chunk: np.ndarray) -> bool:
        """청크 추가"""
        chunk_size = chunk.nbytes
        
        with self.lock:
            if self.current_size + chunk_size > self.max_size:
                return False  # 버퍼 가득 참
            
            self.buffer.append(chunk)
            self.current_size += chunk_size
            return True
    
    def get_audio(self) -> np.ndarray:
        """전체 버퍼 데이터 반환"""
        with self.lock:
            if not self.buffer:
                return np.array([])
            
            audio = np.concatenate(list(self.buffer))
            return audio
    
    def clear(self):
        """버퍼 비우기"""
        with self.lock:
            self.buffer.clear()
            self.current_size = 0
    
    def is_empty(self) -> bool:
        """버퍼가 비어있는지"""
        with self.lock:
            return len(self.buffer) == 0
```

### 오후 (13:00-17:00)

#### 3. BluetoothAudioProvider 초안
```python
# backend/stt_core/input_providers/bluetooth_provider.py

import threading
import socket
import base64

class BluetoothAudioProvider(IAudioInputProvider):
    """블루투스 디바이스에서 음성 수신"""
    
    def __init__(self, device_name: str, timeout_ms: int = 5000, port: int = 9000):
        self.device_name = device_name
        self.timeout_ms = timeout_ms
        self.port = port
        self._socket = None
        self._is_connected = False
        self._buffer = BluetoothAudioBuffer()
        self._receive_thread = None
        self._stop_receiving = False
    
    def connect(self) -> bool:
        """블루투스 디바이스에 연결"""
        try:
            logger.info(f"Connecting to Bluetooth device: {self.device_name}")
            
            # 소켓 생성 (TCP for 개발, 실제는 Bluetooth)
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout_ms / 1000)
            
            # 로컬호스트 연결 (테스트용)
            # 실제는: self._socket.connect(bluetooth_address)
            self._socket.connect(("localhost", self.port))
            
            self._is_connected = True
            logger.info(f"✓ Connected to {self.device_name}")
            
            # 수신 스레드 시작
            self._stop_receiving = False
            self._receive_thread = threading.Thread(target=self._receive_loop)
            self._receive_thread.daemon = True
            self._receive_thread.start()
            
            return True
        
        except Exception as e:
            logger.error(f"✗ Bluetooth connection failed: {e}")
            self._is_connected = False
            raise
    
    def disconnect(self) -> None:
        """블루투스 연결 해제"""
        logger.info(f"Disconnecting from {self.device_name}")
        
        self._stop_receiving = True
        
        if self._receive_thread:
            self._receive_thread.join(timeout=1)
        
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
        
        self._is_connected = False
        logger.info("Disconnected")
    
    def is_connected(self) -> bool:
        return self._is_connected
    
    def _receive_loop(self):
        """백그라운드에서 음성 데이터 수신"""
        while not self._stop_receiving and self._is_connected:
            try:
                # JSON 메시지 수신
                message = self._socket.recv(65536)  # 64KB
                
                if not message:
                    break
                
                # 메시지 파싱
                msg_data = json.loads(message.decode('utf-8'))
                msg_type = msg_data.get("type")
                
                if msg_type == "AUDIO_CHUNK":
                    # Base64 디코딩
                    audio_bytes = base64.b64decode(msg_data.get("data", ""))
                    chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # 버퍼에 추가
                    if not self._buffer.add_chunk(chunk):
                        logger.warning("Audio buffer overflow!")
                
                elif msg_type == "AUDIO_END":
                    logger.info("Audio transmission completed")
                    break
            
            except socket.timeout:
                logger.debug("Socket timeout (no data)")
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                break
    
    def receive_audio(self) -> AudioData:
        """블루투스에서 수신한 음성 데이터 반환"""
        if not self._is_connected:
            raise ConnectionError("Bluetooth device not connected")
        
        # 데이터 수신 대기
        while self._buffer.is_empty() and self._is_connected:
            time.sleep(0.1)
        
        if self._buffer.is_empty():
            raise TimeoutError("No audio data received")
        
        # 버퍼에서 데이터 취출
        audio = self._buffer.get_audio()
        self._buffer.clear()
        
        audio_data = AudioData(
            audio=audio,
            sample_rate=16000,
            channels=1,
            bit_depth=32,
            format="PCM",
            source="bluetooth",
            metadata={
                "device_name": self.device_name
            }
        )
        
        logger.info(f"Received audio: {audio_data}")
        
        return audio_data
    
    def get_source_info(self) -> Dict:
        return {
            "type": "bluetooth",
            "device_name": self.device_name,
            "status": "connected" if self._is_connected else "disconnected",
            "sample_rate": 16000,
            "channels": 1,
            "format": "PCM",
            "port": self.port
        }
```

### 체크리스트
```
✅ 블루투스 통신 프로토콜 정의
✅ 블루투스 메시지 클래스
✅ 블루투스 버퍼 관리
✅ BluetoothAudioProvider 기본 구조
```

---

## 📅 Day 12 (Tuesday) - 블루투스 테스트 서버 & 시뮬레이터

### 아침 (09:00-12:00)

#### 1. 블루투스 테스트 서버
```python
# scripts/bluetooth_simulator.py

"""
블루투스 디바이스 시뮬레이터
실제 하드웨어가 없을 때 테스트용
"""

import socket
import threading
import json
import base64
import numpy as np
import time

class BluetoothSimulator:
    """블루투스 하드웨어 시뮬레이터"""
    
    def __init__(self, host: str = "localhost", port: int = 9000):
        self.host = host
        self.port = port
        self.server = None
        self.running = False
    
    def start(self):
        """시뮬레이터 시작"""
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        
        self.running = True
        
        print(f"🎵 Bluetooth simulator started on {self.host}:{self.port}")
        
        # 클라이언트 연결 대기
        try:
            while self.running:
                try:
                    client, addr = self.server.accept()
                    print(f"📱 Client connected: {addr}")
                    
                    # 클라이언트 처리
                    threading.Thread(target=self._handle_client, args=(client, addr)).start()
                
                except KeyboardInterrupt:
                    break
        finally:
            self.stop()
    
    def _handle_client(self, client: socket.socket, addr):
        """클라이언트 처리"""
        try:
            # 테스트 음성 파일 로드
            import librosa
            audio, sr = librosa.load("test_audio.wav", sr=16000)
            audio = audio.astype(np.float32)
            
            # 청크로 분할 (0.5초씩)
            chunk_size = int(sr * 0.5)  # 0.5초
            
            # 청크 전송
            for i, chunk in enumerate(self._get_chunks(audio, chunk_size)):
                if not self.running:
                    break
                
                # Base64 인코딩
                audio_bytes = chunk.tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # 메시지 생성
                message = {
                    "type": "AUDIO_CHUNK",
                    "data": audio_b64,
                    "chunk_index": i,
                    "sample_rate": sr,
                    "channels": 1,
                    "timestamp": time.time()
                }
                
                # 전송
                client.send(json.dumps(message).encode('utf-8'))
                
                # 지연 (실시간 시뮬레이션)
                time.sleep(0.5)
            
            # 전송 완료
            end_message = {
                "type": "AUDIO_END",
                "total_chunks": i + 1,
                "duration_ms": (i + 1) * 500
            }
            client.send(json.dumps(end_message).encode('utf-8'))
            
            print(f"✓ Audio transmitted to {addr}")
        
        except Exception as e:
            print(f"✗ Error handling client: {e}")
        
        finally:
            client.close()
    
    def _get_chunks(self, audio: np.ndarray, chunk_size: int):
        """음성을 청크로 분할"""
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i+chunk_size]
    
    def stop(self):
        """시뮬레이터 중지"""
        self.running = False
        if self.server:
            self.server.close()
        print("🛑 Bluetooth simulator stopped")

if __name__ == "__main__":
    simulator = BluetoothSimulator()
    simulator.start()
```

#### 2. 블루투스 테스트
```python
# backend/tests/integration/test_bluetooth_connection.py

def test_bluetooth_audio_provider_with_simulator():
    """블루투스 시뮬레이터로 테스트"""
    
    # 시뮬레이터 시작 (별도 프로세스)
    import subprocess
    simulator = subprocess.Popen(
        ["python", "scripts/bluetooth_simulator.py"]
    )
    
    time.sleep(1)  # 시뮬레이터 시작 대기
    
    try:
        # BluetoothAudioProvider 생성
        provider = BluetoothAudioProvider(
            device_name="Simulator",
            port=9000
        )
        
        # 연결
        assert provider.connect() == True
        
        # 음성 데이터 수신
        audio_data = provider.receive_audio()
        
        # 검증
        assert audio_data.sample_rate == 16000
        assert audio_data.channels == 1
        assert len(audio_data.audio) > 0
        
        # 연결 해제
        provider.disconnect()
        
    finally:
        simulator.terminate()
```

### 오후 (13:00-17:00)

#### 3. 실제 하드웨어와의 통신 테스트 계획
```markdown
# 실제 하드웨어 테스트 계획

## 전제 조건
1. 하드웨어팀이 블루투스 모듈 완성
2. 하드웨어에서 JSON 메시지 송신 능력
3. 테스트용 Bluetooth 어댑터

## 테스트 절차
1. iOS/Android 앱에서 블루투스 연결
2. 음성 데이터 전송
3. 백엔드에서 수신 및 처리
4. 결과 확인

## 예상 문제 및 해결책
- 신호 손실: 재전송 로직 구현
- 데이터 손상: CRC 체크 추가
- 지연: 버퍼 크기 조정
```

#### 4. 모바일 앱과의 통신 인터페이스 정의
```python
# backend/stt_core/input_providers/mobile_app_protocol.py

"""
모바일 앱 ←→ 백엔드 HTTP 통신

1. 연결 수립 (HTTP POST)
POST /api/v1/audio/start
{
    "device_id": "phone_123",
    "sample_rate": 16000,
    "format": "PCM"
}

Response:
{
    "session_id": "sess_abc123",
    "status": "ready"
}

2. 음성 데이터 전송 (HTTP POST)
POST /api/v1/audio/chunks?session_id=sess_abc123
Body: binary audio data

Response:
{
    "received_bytes": 4096,
    "status": "ok"
}

3. 음성 전송 완료 (HTTP POST)
POST /api/v1/audio/end?session_id=sess_abc123

Response:
{
    "text": "인식된 텍스트",
    "confidence": 0.95,
    "status": "success"
}
"""
```

### 체크리스트
```
✅ 블루투스 통신 프로토콜 구현
✅ 블루투스 시뮬레이터 개발
✅ 블루투스 테스트 작성
✅ 실제 하드웨어 테스트 계획
✅ 모바일 앱 통신 프로토콜 정의
```

---

## 📅 Day 13 (Wednesday) - 파일 vs 블루투스 일관성 검증

### 아침 (09:00-12:00)

#### 1. 일관성 검증 테스트
```python
# backend/tests/integration/test_consistency.py

"""
파일 입력과 블루투스 입력이 같은 결과를 주는지 검증
"""

def test_file_vs_bluetooth_consistency():
    """같은 음성을 파일과 블루투스로 입력받아 결과 비교"""
    
    # 설정
    config = load_config("backend/config/config.json")
    stt_model = WhisperModel(model_size="base")
    stt_model.load_model()
    
    # 1. 파일 입력으로 처리
    file_provider = FileAudioProvider("test_audio.wav")
    pipeline_file = SpeechToTextPipeline(file_provider, stt_model, config)
    result_file = pipeline_file.process_audio()
    
    # 2. 블루투스 입력으로 처리 (시뮬레이터)
    # 시뮬레이터 시작...
    bt_provider = BluetoothAudioProvider("Simulator")
    pipeline_bt = SpeechToTextPipeline(bt_provider, stt_model, config)
    result_bt = pipeline_bt.process_audio()
    
    # 3. 결과 비교
    assert result_file["text"] == result_bt["text"], \
        f"Text mismatch: '{result_file['text']}' vs '{result_bt['text']}'"
    
    assert abs(result_file["confidence"] - result_bt["confidence"]) < 0.05, \
        f"Confidence mismatch: {result_file['confidence']} vs {result_bt['confidence']}"
    
    stt_model.unload_model()

def test_preprocessing_consistency():
    """전처리 결과가 일관성 있는지 검증"""
    
    # 파일에서 음성 로드
    file_provider = FileAudioProvider("test_audio.wav")
    file_provider.connect()
    audio_file = file_provider.receive_audio()
    
    # 전처리 (첫 번째)
    preprocessor = AudioPreprocessingModule(config)
    processed1 = preprocessor.process(audio_file)
    
    # 전처리 (두 번째)
    processed2 = preprocessor.process(audio_file)
    
    # 결과 비교
    assert np.allclose(processed1, processed2, rtol=1e-5), \
        "Preprocessing results are not consistent"
```

### 오후 (13:00-17:00)

#### 2. 성능 비교
```python
# scripts/compare_input_sources.py

def compare_input_sources():
    """파일과 블루투스 입력 성능 비교"""
    
    config = load_config("backend/config/config.json")
    stt_model = WhisperModel(model_size="base")
    stt_model.load_model()
    
    results = {}
    
    # 1. 파일 입력
    print("Testing file input...")
    file_provider = FileAudioProvider("test_audio.wav")
    pipeline_file = SpeechToTextPipeline(file_provider, stt_model, config)
    result_file = pipeline_file.process_audio()
    
    results["file"] = {
        "time_ms": result_file["total_processing_time_ms"],
        "text": result_file["text"],
        "confidence": result_file["confidence"]
    }
    
    # 2. 블루투스 입력 (시뮬레이터)
    print("Testing Bluetooth input...")
    # 시뮬레이터 시작...
    bt_provider = BluetoothAudioProvider("Simulator")
    pipeline_bt = SpeechToTextPipeline(bt_provider, stt_model, config)
    result_bt = pipeline_bt.process_audio()
    
    results["bluetooth"] = {
        "time_ms": result_bt["total_processing_time_ms"],
        "text": result_bt["text"],
        "confidence": result_bt["confidence"]
    }
    
    # 비교 결과 출력
    print("\n=== Input Source Comparison ===")
    print(f"File input:")
    print(f"  - Time: {results['file']['time_ms']}ms")
    print(f"  - Text: {results['file']['text']}")
    print(f"  - Confidence: {results['file']['confidence']:.2%}")
    
    print(f"\nBluetooth input:")
    print(f"  - Time: {results['bluetooth']['time_ms']}ms")
    print(f"  - Text: {results['bluetooth']['text']}")
    print(f"  - Confidence: {results['bluetooth']['confidence']:.2%}")
    
    print(f"\nConsistency:")
    if results['file']['text'] == results['bluetooth']['text']:
        print("✅ Text matches perfectly")
    else:
        print("⚠️ Text differs")
    
    confidence_diff = abs(
        results['file']['confidence'] - results['bluetooth']['confidence']
    )
    print(f"Confidence difference: {confidence_diff:.2%}")
```

#### 3. 에러 처리 테스트
```python
# backend/tests/integration/test_error_handling.py

def test_bluetooth_timeout():
    """블루투스 타임아웃 처리"""
    
    provider = BluetoothAudioProvider(
        device_name="NonExistent",
        timeout_ms=1000
    )
    
    with pytest.raises(Exception):
        provider.connect()

def test_bluetooth_data_corruption():
    """손상된 데이터 처리"""
    
    # 시뮬레이터에서 손상된 데이터 전송
    # → 백엔드가 graceful하게 처리해야 함
    pass

def test_bluetooth_reconnection():
    """블루투스 재연결"""
    
    provider = BluetoothAudioProvider("Simulator")
    
    # 연결
    provider.connect()
    provider.disconnect()
    
    # 재연결
    provider.connect()
    audio = provider.receive_audio()
    
    assert len(audio.audio) > 0
```

### 체크리스트
```
✅ 일관성 검증 테스트
✅ 성능 비교 분석
✅ 에러 처리 테스트
✅ 재연결 로직 검증
```

---

## 📅 Day 14 (Thursday) - 실제 하드웨어 테스트 준비

### 아침 (09:00-12:00)

#### 1. 하드웨어 팀과의 협력 준비
```markdown
# 블루투스 하드웨어 통합 가이드

## 백엔드에서 기대하는 것

### 1. 데이터 포맷
- 샘플 레이트: 16000 Hz
- 채널: 1 (모노)
- 비트 깊이: 16-bit PCM
- 인코딩: WAV 또는 PCM

### 2. 통신 프로토콜
- 전송 방식: TCP/Bluetooth
- 메시지 포맷: JSON
- 데이터 인코딩: Base64 (바이너리)

### 3. 응답 시간
- 연결 대기: 5초 이내
- 청크 전송 간격: 500ms
- 총 전송 시간: 음성 길이 + 30%

## 테스트 항목
1. 연결/해제
2. 음성 데이터 전송
3. 에러 복구
4. 신호 재전송
```

#### 2. 모니터링 및 디버깅 도구
```python
# scripts/bluetooth_monitor.py

"""
블루투스 통신 모니터링 도구
데이터 송수신 상황을 실시간으로 확인
"""

class BluetoothMonitor:
    def __init__(self, log_file: str = "bluetooth_monitor.log"):
        self.log_file = log_file
        self.stats = {
            "messages_received": 0,
            "bytes_received": 0,
            "errors": 0,
            "start_time": time.time()
        }
    
    def log_message(self, msg: Dict):
        """메시지 로깅"""
        self.stats["messages_received"] += 1
        if "data" in msg:
            data = msg.get("data", "")
            self.stats["bytes_received"] += len(base64.b64decode(data))
        
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()}: {json.dumps(msg)}\n")
    
    def log_error(self, error: str):
        """에러 로깅"""
        self.stats["errors"] += 1
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()}: ERROR - {error}\n")
    
    def print_stats(self):
        """통계 출력"""
        elapsed = time.time() - self.stats["start_time"]
        
        print("\n=== Bluetooth Monitor Stats ===")
        print(f"Messages: {self.stats['messages_received']}")
        print(f"Data: {self.stats['bytes_received'] / 1024:.1f} KB")
        print(f"Errors: {self.stats['errors']}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Throughput: {self.stats['bytes_received'] / elapsed / 1024:.1f} KB/s")
```

### 오후 (13:00-17:00)

#### 3. 실제 하드웨어 테스트 체크리스트
```markdown
# 실제 하드웨어 통합 테스트 체크리스트

## Phase 1: 기본 연결 테스트
- [ ] 블루투스 페어링
- [ ] 연결 성공
- [ ] 연결 유지
- [ ] 연결 해제

## Phase 2: 데이터 전송 테스트
- [ ] 청크 수신
- [ ] 데이터 무결성
- [ ] 전송 속도 확인
- [ ] 타임아웃 처리

## Phase 3: 음성 인식 테스트
- [ ] 단문 인식
- [ ] 장문 인식
- [ ] 노이즈 환경 인식
- [ ] 속삭임 인식

## Phase 4: 에러 처리 테스트
- [ ] 연결 끊김 시 재연결
- [ ] 데이터 손실 시 복구
- [ ] 타임아웃 처리
- [ ] 메모리 부족 처리

## Phase 5: 성능 테스트
- [ ] 지연 시간 측정
- [ ] 처리량 측정
- [ ] 메모리 사용량 측정
- [ ] CPU 사용량 측정
```

#### 4. 문제 해결 가이드
```markdown
# 블루투스 통합 시 예상 문제 및 해결책

## 1. 연결 실패
원인:
- 블루투스 어댑터 문제
- 포트 충돌
- 권한 문제

해결책:
- 블루투스 드라이버 업데이트
- 포트 변경 (9000 → 9001 등)
- 관리자 권한 실행

## 2. 데이터 손실
원인:
- 네트워크 지연
- 버퍼 부족
- 처리 속도 저하

해결책:
- 재전송 로직 추가
- 버퍼 크기 증가
- 청킹 크기 최적화

## 3. 인식 정확도 저하
원인:
- 신호 손실로 인한 음질 저하
- 노이즈 증가

해결책:
- FEC (Forward Error Correction) 추가
- 노이즈 제거 강화
- 모델 파인튜닝

## 4. 높은 지연
원인:
- 네트워크 지연
- 처리 시간 부족
- I/O 병목

해결책:
- 스트리밍 모드 사용
- 배치 크기 최적화
- 병렬 처리 추가
```

### 체크리스트
```
✅ 하드웨어 팀과의 협력 가이드 작성
✅ 모니터링 도구 개발
✅ 실제 하드웨어 테스트 계획 수립
✅ 문제 해결 가이드 작성
```

---

## 📅 Day 15 (Friday) - Week 3 완료 & Week 4 준비

### 아침 (09:00-12:00)

#### 1. Week 3 최종 검증
```bash
# 1-1. 모든 테스트 실행
pytest backend/tests/ -v --cov=backend/stt_core

# 1-2. 블루투스 통합 테스트
pytest backend/tests/integration/test_bluetooth*.py -v
pytest backend/tests/integration/test_consistency.py -v

# 1-3. 시뮬레이터 테스트
python scripts/bluetooth_simulator.py &
pytest backend/tests/integration/test_bluetooth_connection.py -v
```

#### 2. 문서화
```bash
# Week 3 완료 보고서
cat > docs/WEEK_3_COMPLETION.md << 'EOF'
# Week 3 완료 보고서

## 구현된 기능
- BluetoothAudioProvider
- 블루투스 통신 프로토콜
- 블루투스 시뮬레이터
- 일관성 검증 테스트

## 테스트 결과
- 파일 입력: ✅ PASS
- 블루투스 입력: ✅ PASS (시뮬레이터)
- 일관성: ✅ 동일한 결과

## 성능
- 지연: 1500-2000ms (배치 모드)
- 처리량: 안정적

## 실제 하드웨어 테스트 준비
- ✅ 프로토콜 정의
- ✅ 모니터링 도구
- ✅ 테스트 계획
- ✅ 문제 해결 가이드
EOF
```

### 오후 (13:00-17:00)

#### 3. Week 4 준비 (모바일 앱 개발)

```markdown
# Week 4-5 개요: 모바일 앱 개발

## iOS 앱 (Swift)
### Architecture
- MVVM 패턴
- Combine 프레임워크

### 주요 모듈
1. AudioInputService
   - 마이크 입력
   - 노이즈 제거

2. BluetoothService
   - 블루투스 통신
   - 데이터 전송

3. STTService
   - 백엔드와 통신
   - 결과 표시

4. ViewModel
   - 상태 관리
   - 비즈니스 로직

### UI
- ContentView
- RecordingView
- ResultsView
- SettingsView

## Android 앱 (Kotlin)
### Architecture
- MVVM + Repository 패턴
- Coroutine 사용

### 주요 모듈
- AudioInputService
- BluetoothService
- STTService
- ResultViewModel

### UI
- MainActivity
- RecordingFragment
- ResultsFragment
- SettingsFragment
```

#### 4. Git 커밋
```bash
git add .
git commit -m "Week 3: Bluetooth integration completed"
git push origin develop
git tag -a v0.3.0 -m "Week 3: Bluetooth module"
```

### 체크리스트
```
✅ Week 3 모든 테스트 통과
✅ 블루투스 시뮬레이터 검증
✅ 파일 vs 블루투스 일관성 확인
✅ 실제 하드웨어 테스트 계획 수립
✅ 문서화 완료
✅ Week 3 완료!

📊 Week 3 성과:
   - 라인 수: ~1500 (누적 5500)
   - 테스트: ~30개 (누적 120)
   - 커버리지: >85%
   - 블루투스 시뮬레이터: ✅ PASS
```

---

# 📅 WEEK 4-5: 모바일 앱 개발

**목표**: iOS & Android 앱 완성, 실제 디바이스 테스트

**Week 4**: iOS 앱 개발  
**Week 5**: Android 앱 개발 + 통합 테스트

## 📅 Day 16-20 (Week 4): iOS 앱 개발

### 주요 작업

1. **Xcode 프로젝트 생성**
   - SwiftUI 프레임워크
   - MVVM 아키텍처

2. **AVFoundation으로 오디오 입력**
   - 마이크에서 실시간 음성 수집
   - PCM 포맷 변환 (16kHz, 1 채널)

3. **CoreBluetooth로 블루투스 통신**
   - 하드웨어 디바이스와 통신
   - JSON 메시지 송수신

4. **백엔드와 HTTP 통신**
   ```swift
   // 음성 데이터를 백엔드로 전송
   URLSession.shared.uploadTask(
       with: url,
       from: audioData
   ) { data, response, error in
       // 결과 처리
   }.resume()
   ```

5. **UI 구성**
   - 녹음 버튼
   - 진행 바
   - 결과 표시
   - 설정 화면

### 산출물
- ✅ 녹음 기능
- ✅ 블루투스 통신
- ✅ 백엔드 통신
- ✅ 결과 표시
- ✅ 테스트 (XCTest)

---

## 📅 Day 21-25 (Week 5): Android 앱 개발 + 통합 테스트

### 주요 작업

1. **Android Studio 프로젝트**
   - Kotlin + Jetpack Compose
   - MVVM 패턴

2. **MediaRecorder로 오디오 입력**
   - 마이크 권한 요청
   - PCM 포맷 변환

3. **BluetoothSocket으로 통신**
   ```kotlin
   // 블루투스 소켓 통신
   bluetoothSocket = device.createRfcommSocketToServiceRecord(UUID)
   ```

4. **Retrofit으로 HTTP 통신**
   ```kotlin
   // 음성 데이터 업로드
   apiService.uploadAudio(audioData)
   ```

5. **UI 구성 (Compose)**
   - 녹음 화면
   - 결과 화면
   - 설정

### 산출물
- ✅ 녹음 기능
- ✅ 블루투스 통신
- ✅ 백엔드 통신
- ✅ 실시간 결과 표시

### 통합 테스트
```bash
# iOS와 Android 동시 테스트
- 동일한 음성 입력
- 동일한 결과 확인
- 성능 비교
```

---

# 📅 WEEK 6: 최적화 및 파인튜닝

**목표**: 성능 최적화, 정확도 향상, 배포 준비

## 📅 Day 26-30

### 1. 성능 최적화

```python
# 모델 양자화 (Quantization)
# 메모리: 350MB → 180MB
# 속도: 2000ms → 1200ms

import torch
quantized_model = torch.quantization.quantize_dynamic(
    whisper_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### 2. 정확도 향상

```python
# 파인튜닝 데이터 수집
# - 속삭임 음성 500개
# - 배경 노이즈 환경 테스트

# 파인튜닝 실행
python scripts/train_finetuned_model.py \
    --data-dir ./data/whisper_samples \
    --model-size base \
    --output-dir ./models/whisper/finetuned
```

### 3. 배포

```bash
# Docker 이미지 빌드
docker build -t whisper-stt:v1 .

# 클라우드에 배포
docker push your-registry/whisper-stt:v1

# Kubernetes 배포
kubectl apply -f deployment.yaml
```

### 4. 모니터링

```python
# 운영 중 성능 모니터링
- 인식 정확도
- 처리 시간
- 에러율
- 사용자 만족도
```

---

## 📊 최종 성과

```
총 개발 기간: 6주 (30일)

최종 라인 수: ~7000
최종 테스트: ~150개
최종 커버리지: >90%

성능 지표:
- 배치 모드: 1800ms
- 스트리밍 모드: 300-500ms/청크
- 정확도: >90% (속삭임)
- 메모리: 180MB (최적화)
- 크래시율: <0.1%

산출물:
✅ 백엔드 (Python): 완성
✅ iOS 앱 (Swift): 완성
✅ Android 앱 (Kotlin): 완성
✅ 문서: 완성
✅ 테스트: 완성
✅ 배포: 준비됨
```

---

## 📋 체크리스트 (전체 6주)

```
WEEK 1: Input & Preprocessing
✅ FileAudioProvider
✅ AudioInputModule
✅ AudioPreprocessingModule
✅ 파일럿 테스트

WEEK 2: STT & Pipeline
✅ ISTTModel
✅ WhisperModel
✅ STTModule
✅ ResultProcessingModule
✅ SpeechToTextPipeline

WEEK 3: Bluetooth
✅ BluetoothAudioProvider
✅ 블루투스 통신
✅ 일관성 검증
✅ 실제 하드웨어 테스트 계획

WEEK 4-5: Mobile Apps
✅ iOS 앱 (Swift)
✅ Android 앱 (Kotlin)
✅ 통합 테스트

WEEK 6: Optimization
✅ 성능 최적화
✅ 정확도 향상
✅ 배포 준비
✅ 모니터링
```

---

**축하합니다! 6주 만에 완성된 속삭임 감지 음성 변환 시스템! 🎉**
