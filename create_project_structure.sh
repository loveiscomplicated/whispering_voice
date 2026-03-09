#!/bin/bash

# 속삭임 감지 음성 변환 시스템 - 프로젝트 초기화 스크립트
# 수정: EOF 문법 오류 해결

set -e

PROJECT_ROOT="${1:-.}"

echo "🚀 속삭임 감지 음성 변환 시스템 프로젝트 구조 생성 시작..."
echo "📁 Root: $PROJECT_ROOT"
echo ""

# 루트 디렉토리 생성
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# ============ BACKEND 디렉토리 구조 ============
echo "📁 backend/ 디렉토리 생성..."

# backend/stt_core 구조
mkdir -p backend/stt_core/{input_providers,audio_input,preprocessing,stt_models,result_processing,pipeline}

# backend/tests 구조
mkdir -p backend/tests/{unit,integration,fixtures}

# backend/config 구조
mkdir -p backend/config

# ============ MOBILE 디렉토리 구조 ============
echo "📱 mobile/ 디렉토리 생성..."

# iOS 기본 구조
mkdir -p mobile/ios/WhisperVoiceApp/{Models,ViewModels,Views,Services,Resources}
mkdir -p mobile/ios/WhisperVoiceAppTests

# Android 기본 구조
mkdir -p mobile/android/app/src/{main/{kotlin/com/whisper/voiceapp/{models,viewmodels,ui,services},res},test/kotlin}

# ============ DOCS 디렉토리 구조 ============
echo "📚 docs/ 디렉토리 생성..."

mkdir -p docs/{api,diagrams}

# ============ RESEARCH 디렉토리 구조 ============
echo "🔬 research/ 디렉토리 생성..."

mkdir -p research/{whisper_experiments,vosk_experiments,preprocessing_experiments,datasets/{whisper_samples,normal_speech_samples,noise_samples}}

# ============ SCRIPTS 디렉토리 구조 ============
echo "🎯 scripts/ 디렉토리 생성..."

mkdir -p scripts

# ============ MODELS 디렉토리 구조 ============
echo "🤖 models/ 디렉토리 생성..."

mkdir -p models/{whisper/{base,small,finetuned},vosk/korean}

# ============ LOGS 디렉토리 구조 ============
echo "📊 logs/ 디렉토리 생성..."

mkdir -p logs

# ============ DOCKER 디렉토리 구조 ============
echo "🐳 docker/ 디렉토리 생성..."

mkdir -p docker

# ============ GITHUB 워크플로우 디렉토리 구조 ============
echo "🔧 .github/ 디렉토리 생성..."

mkdir -p .github/{workflows,ISSUE_TEMPLATE}

# ============ Python 초기화 파일 ============
echo "🐍 Python __init__.py 파일 생성..."

# backend/stt_core
touch backend/stt_core/__init__.py
touch backend/stt_core/input_providers/__init__.py
touch backend/stt_core/audio_input/__init__.py
touch backend/stt_core/preprocessing/__init__.py
touch backend/stt_core/stt_models/__init__.py
touch backend/stt_core/result_processing/__init__.py
touch backend/stt_core/pipeline/__init__.py

# backend/tests
touch backend/tests/__init__.py
touch backend/tests/unit/__init__.py
touch backend/tests/integration/__init__.py
touch backend/tests/fixtures/__init__.py

# ============ .gitignore 생성 ============
echo "📄 .gitignore 생성..."

cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Models (크기가 큼)
models/whisper/base/*
models/whisper/small/*
models/vosk/korean/*
!models/*/.gitkeep

# Logs
logs/
*.log

# Test artifacts
.pytest_cache/
.coverage
htmlcov/
.tox/

# Environment
.env
.env.local
.env.*.local

# Cache
.cache/
*.cache

# OS
.DS_Store
Thumbs.db

# Mobile
mobile/ios/Pods/
mobile/ios/.xcworkspace/
mobile/android/.gradle/
mobile/android/local.properties
GITIGNORE

# ============ README.md 생성 ============
echo "📄 README.md 생성..."

cat > README.md << 'README'
# 속삭임 감지 음성 변환 시스템 (Whisper Voice Conversion System)

블루투스 하드웨어에서 수신한 속삭임 음성을 STT로 텍스트로 변환하는 모바일 애플리케이션

## 📋 프로젝트 개요

- **입력**: 블루투스 디바이스 또는 오디오 파일
- **처리**: 속삭임 특화 전처리 + STT 모델
- **출력**: 인식된 텍스트 + 신뢰도

## 🚀 빠른 시작

### 1. 개발 환경 설정

```bash
bash scripts/setup_dev_env.sh
source venv/bin/activate
```

### 2. 파일럿 테스트 (파일 입력)

```bash
# 테스트 음성 파일 생성
python scripts/generate_test_audio.py

# 파일럿 테스트 실행
python scripts/run_pilot_test.py
```

### 3. 전체 테스트 실행

```bash
bash scripts/run_tests.sh
```

## 📁 디렉토리 구조

```
whisper-voice-conversion/
├── backend/               # 크로스플랫폼 STT 핵심 모듈
│   ├── stt_core/          # 핵심 로직 (입력/전처리/STT/결과)
│   ├── tests/             # 단위 및 통합 테스트
│   └── config/            # 환경별 설정
├── mobile/                # iOS & Android 앱
├── docs/                  # 문서화
├── scripts/               # 유틸리티 스크립트
└── models/                # STT 모델 저장소
```

## 🔧 핵심 기술 스택

- **STT 모델**: Whisper (OpenAI) 또는 Vosk (오픈소스)
- **음성 처리**: librosa, numpy, scipy
- **프로젝트 언어**: Python (백엔드), Swift (iOS), Kotlin (Android)
- **테스트**: pytest

## 📊 개발 단계

- **Phase 1**: 파일 입력 + 전처리 + 파일럿 테스트 (1-2주)
- **Phase 2**: STT 모델 통합 (1-2주)
- **Phase 3**: 블루투스 입력 통합 (1주)
- **Phase 4**: 모바일 앱 개발 (2주)
- **Phase 5**: 최적화 및 파인튜닝 (1주)

## 📚 문서

- [아키텍처](docs/architecture.md)
- [테스트 가이드](docs/testing_guide.md)
- [파인튜닝 가이드](docs/finetuning_guide.md)

## 👥 기여 가이드

1. `develop` 브랜치에서 `feature/*` 브랜치 생성
2. 코드 작성 및 테스트
3. Pull Request 생성
4. 코드 리뷰 후 병합

## 📞 연락처

프로젝트 관련 문의사항은 [이슈](../../issues) 섹션에 등록해주세요.
README

# ============ requirements.txt 생성 ============
echo "📄 requirements.txt 생성..."

cat > requirements.txt << 'REQUIREMENTS'
# 음성 처리
librosa==0.10.0
numpy==1.24.3
scipy==1.11.1
soundfile==0.12.1

# STT 모델
openai-whisper==20230314
vosk==0.3.45

# 테스트
pytest==7.4.0
pytest-cov==4.1.0

# 코드 품질
black==23.7.0
flake8==6.0.0
mypy==1.4.1

# 로깅 및 설정
python-dotenv==1.0.0
pyyaml==6.0

# 유틸리티
tqdm==4.65.0
REQUIREMENTS

# ============ requirements-dev.txt 생성 ============
echo "📄 requirements-dev.txt 생성..."

cat > requirements-dev.txt << 'REQUIREMENTSDEV'
-r requirements.txt

# 추가 개발 도구
ipython==8.14.0
jupyter==1.0.0
matplotlib==3.7.2
REQUIREMENTSDEV

# ============ .env.example 생성 ============
echo "📄 .env.example 생성..."

cat > .env.example << 'ENVEXAMPLE'
# 환경: development, test, production
ENVIRONMENT=development

# STT 모델
STT_MODEL=whisper
STT_MODEL_SIZE=base

# 로깅
LOG_LEVEL=INFO
LOG_FILE=logs/application.log

# 블루투스
BLUETOOTH_DEVICE_NAME=Whisper Device
BLUETOOTH_TIMEOUT_MS=5000

# 파이프라인
PIPELINE_MODE=hybrid
MAX_AUDIO_DURATION_SECONDS=300
ENVEXAMPLE

# ============ setup.py 생성 ============
echo "📄 setup.py 생성..."

cat > setup.py << 'SETUPPY'
from setuptools import setup, find_packages

setup(
    name="whisper-voice-conversion",
    version="0.1.0",
    description="속삭임 감지 음성 변환 시스템",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "openai-whisper>=20230314",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
        ],
    },
    python_requires=">=3.8",
)
SETUPPY

# ============ backend/config/config.json 생성 ============
echo "📄 backend/config/config.json 생성..."

cat > backend/config/config.json << 'CONFIG'
{
  "inputSource": {
    "type": "file",
    "filePath": "test_audio.wav"
  },
  "audioInput": {
    "maxDuration_seconds": 300,
    "supportedFormats": ["PCM", "WAV"],
    "timeout_ms": 5000
  },
  "preprocessing": {
    "targetSampleRate": 16000,
    "chunkSize_seconds": 2,
    "preEmphasisCoef": 0.97,
    "noiseRemovalAlpha": 3.0,
    "targetRMS": 0.3,
    "vadThreshold": 0.05
  },
  "stt": {
    "model": "whisper",
    "modelSize": "base",
    "language": "ko",
    "confidenceThreshold": 0.7
  },
  "pipeline": {
    "processingMode": "hybrid",
    "timeout_ms": 30000
  },
  "logging": {
    "level": "INFO",
    "maxFileSize_MB": 10
  }
}
CONFIG

# ============ scripts/setup_dev_env.sh 생성 ============
echo "📄 scripts/setup_dev_env.sh 생성..."

cat > scripts/setup_dev_env.sh << 'SETUPDEVENV'
#!/bin/bash

echo "🚀 개발 환경 설정 시작..."

# 1. 가상환경 생성
echo "📦 가상환경 생성..."
python3 -m venv venv

# 2. 가상환경 활성화
echo "🔄 가상환경 활성화..."
source venv/bin/activate

# 3. pip 업그레이드
pip install --upgrade pip

# 4. 의존성 설치
echo "📥 의존성 설치..."
pip install -r requirements-dev.txt

# 5. Whisper 기본 모델 다운로드 (선택사항)
echo "🤖 Whisper 기본 모델 다운로드 (이 부분은 시간이 걸릴 수 있습니다)..."
python -c "import whisper; whisper.load_model('base')" 2>/dev/null || echo "⚠ 모델 다운로드 건너뜀 (나중에 수동으로 실행 가능)"

echo ""
echo "✅ 개발 환경 설정 완료!"
echo ""
echo "다음 단계:"
echo "1. 가상환경 활성화: source venv/bin/activate"
echo "2. 파일럿 테스트: python scripts/run_pilot_test.py"
SETUPDEVENV

chmod +x scripts/setup_dev_env.sh

# ============ scripts/run_tests.sh 생성 ============
echo "📄 scripts/run_tests.sh 생성..."

cat > scripts/run_tests.sh << 'RUNTESTS'
#!/bin/bash

echo "🧪 테스트 실행..."

# 전체 테스트
pytest backend/tests/ -v --cov=backend/stt_core --cov-report=html

echo ""
echo "✅ 테스트 완료!"
echo "📊 커버리지 보고서: htmlcov/index.html"
RUNTESTS

chmod +x scripts/run_tests.sh

# ============ scripts/generate_test_audio.py 생성 ============
echo "📄 scripts/generate_test_audio.py 생성..."

cat > scripts/generate_test_audio.py << 'GENERATEAUDIO'
#!/usr/bin/env python3
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
GENERATEAUDIO

chmod +x scripts/generate_test_audio.py

# ============ scripts/run_pilot_test.py 생성 ============
echo "📄 scripts/run_pilot_test.py 생성..."

cat > scripts/run_pilot_test.py << 'PILOTTEST'
#!/usr/bin/env python3
"""
파일럿 테스트 실행 스크립트 (TODO: 구현 예정)
"""

import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
    
    logger.info("\n💡 다음 단계:")
    logger.info("1. module_code_templates.md의 코드를 각 파일에 복사")
    logger.info("2. backend/stt_core 모듈 구현")
    logger.info("3. 이 스크립트 재실행")


if __name__ == "__main__":
    test_audio_path = "test_audio.wav"
    
    if not os.path.exists(test_audio_path):
        logger.warning(f"테스트 파일 없음: {test_audio_path}")
        logger.info("다음을 실행하여 테스트 파일을 생성하세요:")
        logger.info("  python scripts/generate_test_audio.py")
    else:
        run_pilot_test(test_audio_path)
PILOTTEST

chmod +x scripts/run_pilot_test.py

# ============ backend/tests/conftest.py 생성 ============
echo "📄 backend/tests/conftest.py 생성..."

cat > backend/tests/conftest.py << 'CONFTEST'
"""
pytest 설정 및 fixture
"""

import pytest
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_audio_path():
    """테스트용 샘플 음성 파일 경로"""
    return os.path.join(
        os.path.dirname(__file__),
        'fixtures',
        'test_audio.wav'
    )
CONFTEST

# ============ 테스트 파일 생성 ============
echo "✅ 테스트 파일 생성..."

cat > backend/tests/integration/test_full_pipeline_file.py << 'TESTPIPELINE'
"""
파일 입력을 사용한 전체 파이프라인 통합 테스트 (파일럿)
"""

import pytest
import os


def test_file_audio_provider_initialization():
    """FileAudioProvider 초기화 테스트"""
    pytest.skip("구현 예정")


def test_full_pipeline_with_file_input(sample_audio_path):
    """전체 파이프라인 실행 테스트 (파일 입력)"""
    pytest.skip("구현 예정")


def test_pipeline_output_format():
    """파이프라인 출력 포맷 검증"""
    pytest.skip("구현 예정")
TESTPIPELINE

# ============ .gitkeep 파일 추가 ============
echo "🔐 .gitkeep 파일 생성..."

touch backend/stt_core/input_providers/.gitkeep
touch backend/stt_core/audio_input/.gitkeep
touch backend/stt_core/preprocessing/.gitkeep
touch backend/stt_core/stt_models/.gitkeep
touch backend/stt_core/result_processing/.gitkeep
touch backend/tests/fixtures/.gitkeep
touch models/whisper/base/.gitkeep
touch models/whisper/small/.gitkeep
touch models/vosk/korean/.gitkeep
touch logs/.gitkeep

# ============ 완료 메시지 ============
echo ""
echo "════════════════════════════════════════════════════════"
echo "✨ 프로젝트 초기화 완료!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "📁 프로젝트 위치: $(pwd)"
echo ""
echo "🚀 다음 단계:"
echo ""
echo "  1️⃣  개발 환경 설정 (가상환경 + 패키지 설치):"
echo "      bash scripts/setup_dev_env.sh"
echo ""
echo "  2️⃣  가상환경 활성화:"
echo "      source venv/bin/activate"
echo ""
echo "  3️⃣  테스트 음성 파일 생성:"
echo "      python scripts/generate_test_audio.py"
echo ""
echo "  4️⃣  파일럿 테스트 실행 (구현 후):"
echo "      python scripts/run_pilot_test.py"
echo ""
echo "📚 중요 파일:"
echo "  - README.md (프로젝트 개요)"
echo "  - backend/config/config.json (설정)"
echo "  - requirements.txt (Python 패키지)"
echo ""
echo "💡 코드 템플릿:"
echo "  module_code_templates.md 파일의 코드를 각 모듈에 복사하세요."
echo ""