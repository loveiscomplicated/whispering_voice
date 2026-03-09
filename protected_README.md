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
