# 🎯 6주 상세 로드맵 - 빠른 참조

## 📊 전체 일정

```
WEEK 1 (Day 1-5): 입력/전처리 모듈
WEEK 2 (Day 6-10): STT 모듈 + 전체 파이프라인
WEEK 3 (Day 11-15): 블루투스 모듈 + 일관성 검증
WEEK 4 (Day 16-20): iOS 앱 개발
WEEK 5 (Day 21-25): Android 앱 개발 + 통합 테스트
WEEK 6 (Day 26-30): 최적화 및 파인튜닝
```

---

## 🏗️ 각 주차별 핵심 산출물

### 📅 WEEK 1: 입력/전처리 모듈
**목표**: 파일럿 테스트 성공

| 일 | 작업 | 핵심 코드 |
|----|------|---------|
| 1 | 프로젝트 초기화 & 환경 설정 | `python init_project.py` |
| 2 | AudioData & FileAudioProvider | `AudioData`, `FileAudioProvider` |
| 3 | AudioInputModule & 검증 | `AudioInputModule` |
| 4 | 전처리 알고리즘 | `AudioPreprocessingModule` |
| 5 | 파이프라인 기본 & 파일럿 테스트 | `SpeechToTextPipeline` |

**체크 사항**:
```
✅ FileAudioProvider: 음성 파일 로드
✅ AudioInputModule: 데이터 검증
✅ AudioPreprocessingModule: 
    - 샘플 레이트 정규화
    - 고주파 강화
    - 노이즈 제거
    - VAD (음성 활동 감지)
    - RMS 정규화
✅ 파일럿 테스트: 파일 입력 처리 성공
✅ 커버리지: > 80%
```

---

### 📅 WEEK 2: STT 모듈 + 통합 테스트
**목표**: 전체 파이프라인 동작

| 일 | 작업 | 핵심 코드 |
|----|------|---------|
| 6 | STT 모델 인터페이스 | `ISTTModel`, `STTModule`, `STTModelFactory` |
| 7 | Whisper/Vosk 모델 | `WhisperModel`, `VoskModel` |
| 8 | 결과 처리 | `ResultProcessingModule` |
| 9 | 전체 파이프라인 | `SpeechToTextPipeline` (완성) |
| 10 | 최종 테스트 & 커밋 | 통합 테스트, Git 태그 |

**체크 사항**:
```
✅ WhisperModel: 음성 인식
✅ STTModule: 모델 래핑
✅ ResultProcessingModule:
    - 신뢰도 계산
    - 포맷팅
    - 에러 처리
✅ SpeechToTextPipeline:
    - 배치 모드
    - 스트리밍 모드
    - 하이브리드 모드
✅ 통합 테스트: 파일 → 전처리 → STT → 결과
✅ 커버리지: > 85%
```

---

### 📅 WEEK 3: 블루투스 모듈 + 일관성 검증
**목표**: 실제 하드웨어 테스트 준비

| 일 | 작업 | 핵심 코드 |
|----|------|---------|
| 11 | 블루투스 프로토콜 설계 | `BluetoothMessage`, `BluetoothAudioBuffer` |
| 12 | 블루투스 시뮬레이터 | `BluetoothSimulator`, `BluetoothAudioProvider` |
| 13 | 일관성 검증 | `test_consistency.py` |
| 14 | 실제 하드웨어 준비 | 모니터링 도구, 문제 해결 가이드 |
| 15 | Week 3 완료 & 정리 | 최종 테스트, 문서화 |

**체크 사항**:
```
✅ BluetoothAudioProvider: 블루투스 데이터 수신
✅ 블루투스 시뮬레이터: 테스트 환경
✅ 파일 vs 블루투스 일관성: 동일한 결과
✅ 모니터링 도구: 송수신 데이터 추적
✅ 실제 하드웨어 테스트 계획
✅ 문제 해결 가이드 작성
```

---

### 📅 WEEK 4-5: 모바일 앱 개발
**목표**: iOS & Android 앱 완성

#### WEEK 4 (Day 16-20): iOS 앱
```swift
// 주요 구성
- AudioInputService (마이크 입력)
- BluetoothService (블루투스 통신)
- STTService (백엔드 통신)
- ViewModel (상태 관리)
- UI (SwiftUI)

// 핵심 기능
✅ 마이크에서 실시간 음성 수집
✅ 백엔드로 음성 전송
✅ 인식 결과 표시
✅ 속삭임 특화 UI
```

#### WEEK 5 (Day 21-25): Android 앱
```kotlin
// 주요 구성
- AudioInputService (마이크 입력)
- BluetoothService (블루투스 통신)
- STTService (백엔드 통신)
- ViewModel (상태 관리)
- UI (Jetpack Compose)

// 핵심 기능
✅ MediaRecorder로 음성 수집
✅ Retrofit으로 백엔드 통신
✅ 실시간 결과 표시
✅ 안드로이드 권한 관리

// 통합 테스트
✅ iOS와 Android 동일한 결과
✅ 성능 비교
```

---

### 📅 WEEK 6: 최적화 및 파인튜닝
**목표**: 성능 향상 & 배포 준비

```python
# 최적화
- 모델 양자화: 350MB → 180MB
- 처리 속도: 2000ms → 1200ms
- 메모리 사용: 최적화

# 정확도 향상
- 속삭임 음성 데이터 수집 (500개)
- 파인튜닝 모델 학습
- 정확도 >90%

# 배포
- Docker 이미지 빌드
- 클라우드 배포 (AWS/GCP/Azure)
- 모니터링 설정

# 성능 지표
✅ 정확도: >90% (속삭임)
✅ 지연: 1200ms (배치)
✅ 메모리: 180MB
✅ 크래시율: <0.1%
```

---

## 📈 일일 작업 패턴

### 아침 (09:00-12:00): 개발
```
1. 전날 검토
2. 오늘의 목표 확인
3. 코드 작성
4. 단위 테스트 작성
```

### 오후 (13:00-17:00): 테스트 & 정리
```
1. 통합 테스트
2. 코드 리뷰
3. 문서 작성
4. Git 커밋
```

### 금요일 (Day 5, 10, 15, 20, 25, 30): 최종 검증
```
1. 전체 테스트 실행
2. 커버리지 확인
3. 성능 벤치마크
4. 문서화
5. Git 태그
```

---

## 🚀 시작하기

### Step 1: 프로젝트 생성 (Day 1 아침)
```bash
python init_project.py
bash scripts/setup_dev_env.sh
source venv/bin/activate
```

### Step 2: Week 1 구현 (Day 1-5)
```bash
# DETAILED_ROADMAP.md의 Day 1-5 항목 따라 구현
# module_code_templates.md 참고하여 코드 작성
python scripts/generate_test_audio.py
python scripts/run_pilot_test.py
```

### Step 3: 주간별 진행 (Week 2-6)
```bash
# 각 주 마지막 날 (금요일)
pytest backend/tests/ -v --cov=backend/stt_core
git add .
git commit -m "Week X completed"
git push origin develop
git tag -a vX.Y.Z -m "Week X completion"
```

---

## 📚 참고 자료

| 문서 | 내용 | 언제 |
|------|------|------|
| **DETAILED_ROADMAP.md** | 상세 일일 계획 | 매일 참고 |
| **project_directory_structure.md** | 프로젝트 구조 | 처음 한 번 |
| **module_code_templates.md** | 코드 템플릿 | 개발할 때 |
| **INITIALIZATION_GUIDE.md** | 초기화 방법 | Day 1 |

---

## ⏱️ 시간 배분

```
프로젝트 초기화: 2시간 (Day 1 오전)
WEEK 1: 40시간 (일주일)
WEEK 2: 40시간 (일주일)
WEEK 3: 40시간 (일주일)
WEEK 4: 40시간 (iOS 앱, 일주일)
WEEK 5: 40시간 (Android 앱, 일주일)
WEEK 6: 40시간 (최적화, 일주일)

총합: 242시간 (약 6주, 주 40시간 기준)
```

---

## 📊 최종 체크리스트

### WEEK 1 완료 후
```
✅ FileAudioProvider 완성
✅ AudioInputModule 완성
✅ AudioPreprocessingModule 완성
✅ 파일럿 테스트 성공
✅ 커버리지 > 80%
✅ 약 1500 라인의 코드
✅ 약 30개 테스트
```

### WEEK 2 완료 후
```
✅ ISTTModel 인터페이스
✅ WhisperModel/VoskModel
✅ STTModule 완성
✅ ResultProcessingModule
✅ SpeechToTextPipeline 완성
✅ 통합 테스트 성공
✅ 커버리지 > 85%
✅ 추가 2500 라인의 코드
✅ 추가 30개 테스트
```

### WEEK 3 완료 후
```
✅ BluetoothAudioProvider
✅ 블루투스 시뮬레이터
✅ 일관성 검증 완료
✅ 실제 하드웨어 테스트 계획
✅ 추가 1500 라인의 코드
✅ 추가 30개 테스트
```

### WEEK 4-5 완료 후
```
✅ iOS 앱 완성 (SwiftUI)
✅ Android 앱 완성 (Kotlin)
✅ 통합 테스트 성공
✅ 실제 디바이스 테스트 완료
```

### WEEK 6 완료 후
```
✅ 성능 최적화 완료
✅ 정확도 >90% 달성
✅ 배포 준비 완료
✅ 모니터링 설정 완료
```

---

## 🎉 프로젝트 완료!

```
총 개발 시간: 240시간 (6주)
총 라인 수: ~7000
총 테스트: ~150개
최종 커버리지: >90%

산출물:
- ✅ 백엔드 (Python): 완성
- ✅ iOS 앱 (Swift): 완성
- ✅ Android 앱 (Kotlin): 완성
- ✅ 문서: 완성
- ✅ 테스트: 완성
- ✅ 배포: 준비됨

성능:
- 정확도: >90% (속삭임)
- 지연: 1200ms (배치 모드)
- 메모리: 180MB
- 크래시율: <0.1%
```

---

**축하합니다! 이제 시작할 준비가 되었습니다! 🚀**

**다음 단계:**
1. `DETAILED_ROADMAP.md` 읽기
2. `python init_project.py` 실행
3. Day 1 작업 시작!
