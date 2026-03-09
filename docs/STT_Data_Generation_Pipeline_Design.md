# STT 데이터 생성 파이프라인 설계 문서

## 📋 목차
1. [개요](#개요)
2. [핵심 개선 사항](#핵심-개선-사항)
3. [파이프라인 아키텍처](#파이프라인-아키텍처)
4. [메타데이터 구조](#메타데이터-구조)
5. [디렉토리 구조](#디렉토리-구조)
6. [기술 선택](#기술-선택)
7. [주요 고려사항](#주요-고려사항)
8. [장점 요약](#장점-요약)

---

## 개요

본 문서는 STT 모델의 파인튜닝을 위한 **자동화된 데이터 생성 파이프라인**의 설계를 설명합니다.

**핵심 목표:**
- YouTube ASMR 영상에서 음성 추출
- Plain STT 모델로 정답 transcript 자동 생성
- VAD(Voice Activity Detection)로 음성 구간 자동 식별
- 배경소음 합성으로 다양한 SNR 데이터 생성
- 파인튜닝에 필요한 고품질 라벨링 데이터셋 자동 구성

---

## 핵심 개선 사항

### 1. 데이터 소스 다양성 부족

**문제:**
- YouTube ASMR만으로는 속삭임 음성의 다양성이 제한적
- 음성 특성 편향 가능성

**개선안:**
- ASMR 외에도 일상 속삭임 포함 (도서관, 병원, 사무실)
- 다양한 언어/방언의 속삭임 데이터 수집
- 배경 소음 강도별 분류 (조용함/보통/시끄러움)
- 화자 다양성 확보 (성별, 나이, 억양)

---

### 2. Plain STT 모델 실행 및 Transcript 기록

**필요성:**
파인튜닝을 위한 Ground Truth 확보의 필수 단계

**수행 작업:**
```
ASMR 음성 → Plain STT 모델 실행 → Transcript 추출
                                 → Confidence score 기록
```

**얻을 수 있는 이점:**
- 수동 라벨링 비용 제거
- 자동화된 Ground Truth 확보
- Plain 모델과 파인튜닝 모델 성능 비교 가능
- 신뢰도 기반 데이터 품질 판단

---

### 3. VAD (Voice Activity Detection) 구간 기록

**필요성:**
음성과 침묵 구간을 정확히 파악하기 위함

**기록 정보:**
- 음성 구간의 시작/종료 시간
- 각 구간의 confidence score
- 음성/침묵 비율
- 전체 음성 길이

**파인튜닝 활용:**
- 정확한 음성 구간으로 부분 학습 가능
- 침묵 처리 개선
- 속삭임 시작/종료 감지 성능 향상

---

### 4. 라벨링 자동화

**문제:**
- 수동으로 수천 개의 음성 파일을 라벨링하기는 불가능
- 시간과 비용 소모

**해결책:**
- Plain STT 모델의 transcript 자동 추출
- 충분한 confidence score를 가진 결과만 사용
- 배경소음 추가 후에도 같은 transcript 적용 가능

---

### 5. 품질 검증 메커니즘

**검증 항목:**
- 추출된 음성 길이 필터링 (너무 짧거나 긴 것 제외)
- 에너지 레벨 검증 (적절한 음량인지 확인)
- 음성/비음성 비율 확인 (침묵이 너무 많으면 제외)
- STT confidence score 기준 (낮은 신뢰도 제외)

---

### 6. 메모리/저장소 효율성

**문제:**
- YouTube에서 전체 영상을 다운받으면 저장소 폭발

**개선안:**
- 영상이 아닌 음성만 추출 (youtube-dl의 audio-only 모드)
- 스트리밍 처리: 전체 파일을 메모리에 로드하지 말고 청크 단위 처리
- 중복 제거: 같은 영상에서 여러 번 다운받지 않기

---

### 7. 데이터 균형 문제

**문제:**
- ASMR과 배경소음의 비율을 어떻게 맞출 것인가?

**해결책:**
설정 가능한 매개변수:
```json
{
  "snr_range": {
    "min_db": 5,
    "max_db": 25,
    "step": 5
  },
  "noise_type_distribution": {
    "ambient": 0.4,
    "traffic": 0.3,
    "office": 0.3
  },
  "whisper_intensity_levels": ["soft", "normal", "loud"]
}
```

---

### 8. 재현성(Reproducibility) 보장

**문제:**
- 같은 데이터셋을 다시 만들 수 없으면 재실험 불가능

**해결책:**
- 시드(seed) 값으로 랜덤 조합 고정
- 사용된 영상 목록 저장 (변경 추적)
- 생성 로그 기록
- 생성 설정 버전 관리

---

### 9. 에러 처리

**가능한 에러:**
- YouTube 다운로드 실패
- 손상된 오디오 파일
- STT 모델 처리 실패
- VAD 처리 실패

**해결책:**
- 다운로드 실패한 영상 재시도 로직
- 손상된 오디오 파일 자동 스킵
- 부분 실패 시 저장 상태 복구
- 상세한 에러 로깅

---

### 10. 배경소음 합성 고도화

**문제:**
- 단순히 두 음성을 섞기만 하면 부자연스러움

**개선안:**
- **에너지 정규화**: 음성 레벨 동기화
- **시간 정렬**: ASMR과 소음의 타이밍 맞추기
- **페이드 인/아웃**: 부자연스러운 절단 방지
- **주파수 영역 분리**: 고주파/저주파 따로 처리
- **동적 믹싱**: SNR에 따른 음성-소음 비율 자동 조정

---

## 파이프라인 아키텍처

### 전체 데이터 흐름

```
┌─────────────────────────────────────┐
│ 1️⃣ YouTube ASMR 영상 다운로드        │
│    (audio-only, 스트리밍)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2️⃣ 음성 추출 & 전처리               │
│    - 리샘플링                        │
│    - 정규화                          │
│    - 길이 검증                       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3️⃣ 품질 검증 (1차)                  │
│    - 에너지 레벨                     │
│    - 길이 범위                       │
│    - 음성/침묵 비율                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 4️⃣ Plain STT 모델 실행                   │
│    (Whisper-base 또는 선택된 모델)      │
│    ├─ Transcript 추출                   │
│    ├─ Language 감지                     │
│    └─ Confidence score 기록             │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 5️⃣ VAD(Voice Activity Detection) 실행  │
│    (Pyannote 또는 Silero VAD)           │
│    ├─ Voice segments 추출               │
│    ├─ Silence segments 식별             │
│    └─ Confidence score 기록             │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 6️⃣ 메타데이터 생성 & 저장               │
│    {                                     │
│      audio_id,                           │
│      original_transcript,                │
│      stt_confidence,                     │
│      vad_segments,                       │
│      audio_characteristics               │
│    }                                     │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 7️⃣ 품질 검증 (2차)                      │
│    - STT confidence 기준 필터링          │
│    - VAD 결과 유효성 검증               │
│    - 자동 라벨링 품질 확인              │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 8️⃣ 배경소음 추가 & 고도화된 합성       │
│    - 에너지 정규화                       │
│    - 동적 믹싱 (SNR 기반)                │
│    - 여러 SNR 레벨 생성                  │
│      (SNR: 5, 10, 15, 20, 25 dB)        │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 9️⃣ 합성 품질 검증                       │
│    - 신호-잡음비(SNR) 검증              │
│    - 음성 손상도 확인                    │
│    - Metadata 업데이트                  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 🔟 최종 데이터셋 생성                    │
│    파이프라인 완료                       │
│                                          │
│    출력:                                 │
│    ├─ 합성된 오디오 파일                │
│    ├─ 메타데이터 JSON                   │
│    ├─ 데이터셋 매니페스트               │
│    └─ 생성 보고서                       │
└──────────────────────────────────────────┘
```

---

## 메타데이터 구조

### 상세 메타데이터 스키마

```json
{
  "audio_id": "asmr_001",
  "processing_pipeline_version": "1.0",
  
  "📝_plain_stt_result": {
    "transcript": "안녕하세요, 오늘 날씨가 좋네요",
    "language": "ko",
    "confidence_score": 0.96,
    "model_used": "whisper-base",
    "model_version": "v20230314",
    "processing_time_ms": 1200
  },
  
  "🎤_vad_result": {
    "segments": [
      {
        "segment_id": 0,
        "start_ms": 100,
        "end_ms": 2500,
        "duration_ms": 2400,
        "confidence": 0.98,
        "text": "안녕하세요"
      },
      {
        "segment_id": 1,
        "start_ms": 3000,
        "end_ms": 5800,
        "duration_ms": 2800,
        "confidence": 0.95,
        "text": "오늘 날씨가 좋네요"
      }
    ],
    "statistics": {
      "total_segments": 2,
      "total_speech_duration_ms": 5200,
      "total_silence_duration_ms": 2900,
      "speech_ratio": 0.64,
      "average_segment_duration_ms": 2600,
      "average_confidence": 0.965
    },
    "vad_model_used": "pyannote/segmentation",
    "vad_model_version": "2.1"
  },
  
  "🔊_source_information": {
    "youtube": {
      "video_id": "dQw4w9WgXcQ",
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      "title": "ASMR Whisper Roleplay",
      "channel": "ASMR Channel",
      "upload_date": "2024-01-15"
    },
    "download": {
      "timestamp": "2024-03-09T10:30:00Z",
      "download_duration_ms": 45000,
      "source_video_duration_ms": 900000,
      "extracted_segment": {
        "start_ms": 0,
        "end_ms": 8100,
        "duration_ms": 8100
      }
    }
  },
  
  "🎵_audio_characteristics": {
    "format": "wav",
    "sample_rate": 16000,
    "channels": 1,
    "bit_depth": 16,
    "duration_ms": 8100,
    "rms_energy_db": -20.5,
    "peak_amplitude": 0.85,
    "loudness_lufs": -18.2,
    "frequency_characteristics": {
      "frequency_range_hz": [50, 8000],
      "spectral_centroid_hz": 2500,
      "mfcc_coefficients": 13
    }
  },
  
  "✅_quality_validation": {
    "validation_timestamp": "2024-03-09T10:35:00Z",
    "overall_passed": true,
    "validation_checks": {
      "minimum_length": {
        "requirement_ms": 1000,
        "actual_value_ms": 8100,
        "passed": true,
        "message": "OK"
      },
      "maximum_length": {
        "requirement_ms": 30000,
        "actual_value_ms": 8100,
        "passed": true,
        "message": "OK"
      },
      "rms_energy_range": {
        "min_required_db": -30,
        "max_required_db": -10,
        "actual_value_db": -20.5,
        "passed": true,
        "message": "Energy level acceptable"
      },
      "speech_to_silence_ratio": {
        "minimum_required": 0.3,
        "actual_value": 0.64,
        "passed": true,
        "message": "Good speech content"
      },
      "stt_confidence": {
        "minimum_required": 0.7,
        "actual_value": 0.96,
        "passed": true,
        "message": "High confidence transcription"
      },
      "vad_quality": {
        "minimum_avg_confidence": 0.8,
        "actual_avg_confidence": 0.965,
        "passed": true,
        "message": "Excellent VAD quality"
      }
    },
    "rejected_reasons": []
  },
  
  "🔊_synthesized_versions": [
    {
      "synthesis_id": "syn_001_snr15_ambient",
      "original_audio_id": "asmr_001",
      "output_filename": "asmr_001_snr15_ambient.wav",
      "noise_source": {
        "youtube_video_id": "noise_ambient_005",
        "noise_type": "ambient",
        "noise_category": "background_chatter",
        "download_timestamp": "2024-03-09T11:00:00Z"
      },
      "synthesis_parameters": {
        "snr_db": 15,
        "mixing_strategy": "energy_normalized",
        "fade_duration_ms": 100
      },
      "quality_metrics": {
        "signal_to_noise_ratio_db": 15.2,
        "pesq_score": 3.2,
        "overall_quality": "acceptable"
      },
      "metadata": {
        "original_transcript": "안녕하세요, 오늘 날씨가 좋네요",
        "vad_segments": [
          {"start_ms": 100, "end_ms": 2500},
          {"start_ms": 3000, "end_ms": 5800}
        ],
        "synthesis_timestamp": "2024-03-09T11:00:30Z"
      }
    },
    {
      "synthesis_id": "syn_002_snr10_traffic",
      "original_audio_id": "asmr_001",
      "output_filename": "asmr_001_snr10_traffic.wav",
      "noise_source": {
        "youtube_video_id": "noise_traffic_012",
        "noise_type": "traffic",
        "noise_category": "vehicle_sounds",
        "download_timestamp": "2024-03-09T11:00:00Z"
      },
      "synthesis_parameters": {
        "snr_db": 10,
        "mixing_strategy": "energy_normalized",
        "fade_duration_ms": 100
      },
      "quality_metrics": {
        "signal_to_noise_ratio_db": 10.1,
        "pesq_score": 2.8,
        "overall_quality": "acceptable"
      },
      "metadata": {
        "original_transcript": "안녕하세요, 오늘 날씨가 좋네요",
        "vad_segments": [
          {"start_ms": 100, "end_ms": 2500},
          {"start_ms": 3000, "end_ms": 5800}
        ],
        "synthesis_timestamp": "2024-03-09T11:01:00Z"
      }
    }
  ],
  
  "📊_dataset_classification": {
    "split": "train",
    "fold": 0,
    "difficulty_level": "easy",
    "whisper_intensity": "normal",
    "background_complexity": "low",
    "suggested_for_finetuning": true,
    "suggested_for_evaluation": false
  },
  
  "🔄_processing_log": {
    "events": [
      {
        "timestamp": "2024-03-09T10:30:00Z",
        "stage": "download",
        "status": "completed",
        "duration_ms": 45000
      },
      {
        "timestamp": "2024-03-09T10:31:00Z",
        "stage": "extraction",
        "status": "completed",
        "duration_ms": 5000
      },
      {
        "timestamp": "2024-03-09T10:32:00Z",
        "stage": "stt",
        "status": "completed",
        "duration_ms": 1200
      },
      {
        "timestamp": "2024-03-09T10:33:00Z",
        "stage": "vad",
        "status": "completed",
        "duration_ms": 800
      },
      {
        "timestamp": "2024-03-09T10:35:00Z",
        "stage": "validation",
        "status": "completed",
        "duration_ms": 500
      },
      {
        "timestamp": "2024-03-09T11:00:30Z",
        "stage": "synthesis",
        "status": "completed",
        "duration_ms": 30000,
        "variants_created": 2
      }
    ],
    "total_processing_time_ms": 82500
  }
}
```

---

## 디렉토리 구조

### 전체 디렉토리 레이아웃

```
data_generation/
│
├── 📋 README.md
├── 📋 CONFIG.md
│
├── 🎯 config/
│   ├── playlist_config.json        # YouTube 플레이리스트 URL
│   │   └── Example:
│   │       {
│   │         "asmr_playlists": [
│   │           "https://www.youtube.com/playlist?list=...",
│   │           "https://www.youtube.com/playlist?list=..."
│   │         ],
│   │         "noise_playlists": {
│   │           "ambient": "https://www.youtube.com/playlist?list=...",
│   │           "traffic": "https://www.youtube.com/playlist?list=...",
│   │           "office": "https://www.youtube.com/playlist?list=..."
│   │         }
│   │       }
│   │
│   ├── generation_config.json      # 생성 파라미터 설정
│   │   └── Example:
│   │       {
│   │         "stt_model": "whisper-base",
│   │         "vad_model": "pyannote/segmentation",
│   │         "snr_levels": [5, 10, 15, 20, 25],
│   │         "max_samples_per_playlist": 1000,
│   │         "batch_size": 10
│   │       }
│   │
│   ├── quality_config.json         # 품질 검증 기준
│   │   └── Example:
│   │       {
│   │         "min_length_ms": 1000,
│   │         "max_length_ms": 30000,
│   │         "min_speech_ratio": 0.3,
│   │         "min_stt_confidence": 0.7,
│   │         "min_vad_confidence": 0.8
│   │       }
│   │
│   └── seed_config.json            # 재현성 설정
│       └── random_seed: 42
│
├── 📥 raw_downloads/               # 원본 다운로드 데이터
│   │
│   ├── asmr/                       # Plain STT 실행 전 ASMR 음성
│   │   ├── asmr_001.wav
│   │   ├── asmr_002.wav
│   │   ├── asmr_003.wav
│   │   └── ...
│   │   └── manifest.json           # 다운로드된 파일 목록
│   │
│   └── noise/                      # 배경소음 원본 음성
│       ├── ambient/
│       │   ├── ambient_001.wav
│       │   ├── ambient_002.wav
│       │   └── ...
│       ├── traffic/
│       │   ├── traffic_001.wav
│       │   ├── traffic_002.wav
│       │   └── ...
│       ├── office/
│       │   ├── office_001.wav
│       │   ├── office_002.wav
│       │   └── ...
│       └── manifest.json           # 모든 소음 파일 목록
│
├── 📋 stt_and_vad/                 # Plain STT + VAD 처리 결과
│   │
│   ├── metadata/                   # 메타데이터 저장
│   │   ├── asmr_001_metadata.json
│   │   ├── asmr_002_metadata.json
│   │   ├── asmr_003_metadata.json
│   │   └── ...
│   │
│   ├── transcripts/                # Transcript 텍스트 파일
│   │   ├── asmr_001_transcript.txt
│   │   ├── asmr_002_transcript.txt
│   │   └── ...
│   │
│   ├── vad_segments/               # VAD 결과 JSON
│   │   ├── asmr_001_vad.json
│   │   ├── asmr_002_vad.json
│   │   └── ...
│   │
│   ├── manifest.json               # 전체 처리 목록
│   │   └── Example:
│   │       {
│   │         "total_files": 1000,
│   │         "successfully_processed": 950,
│   │         "failed_files": 50,
│   │         "processing_timestamp": "2024-03-09T10:35:00Z"
│   │       }
│   │
│   └── logs/                       # 처리 로그
│       ├── stt_processing.log
│       ├── vad_processing.log
│       └── validation_errors.log
│
├── 🔊 synthesized/                 # 배경소음 추가 후 합성된 데이터
│   │
│   ├── by_snr/                     # SNR별 구조
│   │   ├── snr_05/
│   │   │   ├── asmr_001_snr05_ambient.wav
│   │   │   ├── asmr_001_snr05_traffic.wav
│   │   │   ├── asmr_002_snr05_ambient.wav
│   │   │   └── ...
│   │   │
│   │   ├── snr_10/
│   │   │   └── ...
│   │   │
│   │   ├── snr_15/
│   │   │   └── ...
│   │   │
│   │   ├── snr_20/
│   │   │   └── ...
│   │   │
│   │   └── snr_25/
│   │       └── ...
│   │
│   ├── by_noise_type/              # 노이즈 타입별 구조
│   │   ├── ambient/
│   │   │   ├── snr_05/
│   │   │   ├── snr_10/
│   │   │   └── ...
│   │   │
│   │   ├── traffic/
│   │   │   └── ...
│   │   │
│   │   └── office/
│   │       └── ...
│   │
│   ├── metadata/                   # 합성 메타데이터
│   │   ├── asmr_001_snr05_ambient_metadata.json
│   │   ├── asmr_001_snr10_traffic_metadata.json
│   │   └── ...
│   │
│   ├── manifest.json               # 모든 합성 파일 목록
│   │
│   └── logs/
│       ├── synthesis.log
│       └── quality_metrics.log
│
├── 📊 analysis/                    # 분석 및 통계
│   │
│   ├── quality_report.csv          # 품질 검증 결과 리포트
│   │
│   ├── stt_statistics/
│   │   ├── confidence_distribution.png
│   │   ├── transcript_length_dist.png
│   │   └── language_distribution.png
│   │
│   ├── vad_statistics/
│   │   ├── speech_ratio_distribution.png
│   │   ├── segment_duration_dist.png
│   │   └── vad_confidence_distribution.png
│   │
│   ├── synthesis_validation/
│   │   ├── snr_accuracy_report.csv
│   │   ├── noise_type_distribution.png
│   │   └── quality_by_snr.png
│   │
│   ├── dataset_summary/
│   │   ├── overall_statistics.json
│   │   ├── train_test_split.csv
│   │   └── data_distribution_report.md
│   │
│   └── logs/
│       └── analysis_errors.log
│
├── 🔄 pipeline/                    # 파이프라인 스크립트
│   │
│   ├── __init__.py
│   │
│   ├── 1_download_audio.py         # 단계 1: YouTube 영상 다운로드
│   │   └── Functions:
│   │       - download_from_playlist()
│   │       - validate_downloaded_audio()
│   │       - save_manifest()
│   │
│   ├── 2_validate_quality.py       # 단계 2: 품질 1차 검증
│   │   └── Functions:
│   │       - validate_audio_format()
│   │       - validate_audio_length()
│   │       - validate_energy_level()
│   │       - validate_speech_ratio()
│   │
│   ├── 3_run_stt_and_vad.py        # 단계 3: STT + VAD 실행
│   │   └── Functions:
│   │       - load_stt_model()
│   │       - run_stt()
│   │       - load_vad_model()
│   │       - run_vad()
│   │       - save_metadata()
│   │       - batch_processing()
│   │
│   ├── 4_synthesize_noise.py       # 단계 4: 배경소음 합성
│   │   └── Functions:
│   │       - normalize_energy()
│   │       - mix_audio_by_snr()
│   │       - apply_fade()
│   │       - synthesize_variants()
│   │
│   ├── 5_quality_validation_2.py   # 단계 5: 품질 2차 검증
│   │   └── Functions:
│   │       - validate_stt_confidence()
│   │       - validate_vad_results()
│   │       - validate_synthesis()
│   │
│   ├── 6_generate_finetuning_dataset.py  # 단계 6: 파인튜닝 데이터셋 생성
│   │   └── Functions:
│   │       - create_training_split()
│   │       - create_validation_split()
│   │       - create_test_split()
│   │       - generate_dataset_manifest()
│   │
│   └── data_generation_pipeline.py # 전체 파이프라인 오케스트레이션
│       └── Functions:
│           - run_full_pipeline()
│           - run_pipeline_from_stage()
│           - validate_all_stages()
│           - generate_report()
│
├── 🧪 tests/                       # 단위 및 통합 테스트
│   │
│   ├── __init__.py
│   │
│   ├── unit/
│   │   ├── test_audio_download.py
│   │   ├── test_quality_validation.py
│   │   ├── test_stt_processing.py
│   │   ├── test_vad_processing.py
│   │   ├── test_synthesis.py
│   │   └── test_metadata.py
│   │
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   └── test_data_consistency.py
│   │
│   └── fixtures/
│       ├── sample_asmr.wav
│       ├── sample_noise.wav
│       └── sample_metadata.json
│
├── 📚 docs/                        # 문서
│   │
│   ├── PIPELINE_DESIGN.md          # 파이프라인 설계 (본 문서)
│   ├── SETUP.md                    # 설치 및 설정 가이드
│   ├── USAGE.md                    # 사용 가이드
│   ├── MODEL_SELECTION.md          # STT/VAD 모델 선택 가이드
│   ├── TROUBLESHOOTING.md          # 문제 해결 가이드
│   └── API_REFERENCE.md            # API 참조
│
├── requirements.txt                # Python 의존성
│   └── Example:
│       yt-dlp>=2024.01.01
│       librosa>=0.10.0
│       numpy>=1.24.0
│       scipy>=1.11.0
│       soundfile>=0.12.1
│       openai-whisper>=20230314
│       pyannote.audio>=2.1
│       torch>=2.0.0
│       torchaudio>=2.0.0
│
├── .env.example                    # 환경 변수 템플릿
│   └── Example:
│       YOUTUBE_API_KEY=your_key
│       CUDA_VISIBLE_DEVICES=0
│       MAX_WORKERS=4
│
└── .gitignore                      # Git 무시 파일
    └── Includes:
        /raw_downloads/
        /stt_and_vad/
        /synthesized/
        *.wav
        *.log
        .env
```

---

## 기술 선택

### STT 모델 선택

| 모델 | 장점 | 단점 | 권장도 |
|------|------|------|--------|
| **Whisper (OpenAI)** | 다양한 언어, 배경소음 강함, 속도 | 로컬 모델 필요 | ⭐⭐⭐⭐⭐ |
| **Google Speech-to-Text** | 매우 정확, 많은 언어 | 비용, API 할당량 | ⭐⭐⭐ |
| **KoSpeech** | 한국어 최적화 | 영어 지원 약함 | ⭐⭐⭐⭐ |
| **Azure Speech Services** | 엔터프라이즈 지원 | 비용 | ⭐⭐⭐ |

**최종 선택: Whisper-base**
```
이유:
- 배경소음에 강함 (속삭임 데이터에 적합)
- 무료
- 다양한 언어 지원
- 충분한 정확도
```

---

### VAD 모델 선택

| 모델 | 장점 | 단점 | 권장도 |
|------|------|------|--------|
| **Pyannote** | 높은 정확도, 여러 언어 | 무거움 | ⭐⭐⭐⭐⭐ |
| **Silero VAD** | 가볍고 빠름, 다양한 언어 | 정확도 낮을 수 있음 | ⭐⭐⭐⭐ |
| **WebRTC VAD** | 매우 빠름, 가벼움 | 정확도 낮음 | ⭐⭐⭐ |

**최종 선택: Pyannote v2.1**
```
이유:
- 높은 정확도
- 여러 언어 지원
- Speaker diarization도 가능
```

---

### 오디오 처리 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| **librosa** | 음성 분석, 특징 추출 |
| **soundfile** | WAV 파일 읽기/쓰기 |
| **scipy** | 신호 처리 (필터링 등) |
| **numpy** | 수치 계산 |
| **torchaudio** | PyTorch 오디오 처리 |

---

## 주요 고려사항

### A. Confidence Score 처리

```python
# 옵션 1: 높은 신뢰도만 사용 (권장)
if stt_confidence >= 0.85:
    use_for_finetuning = True
else:
    use_for_evaluation_only = True

# 옵션 2: 신뢰도 기반 가중치
weight = stt_confidence  # 파인튜닝 시 손실 함수에 반영

# 옵션 3: 신뢰도 범위별 분류
if stt_confidence >= 0.9:
    difficulty = "easy"
elif 0.7 <= stt_confidence < 0.9:
    difficulty = "medium"
else:
    difficulty = "hard"
```

---

### B. VAD와 STT 결과 매칭

```python
# VAD 구간과 STT 결과를 텍스트 정렬로 매칭
# 예: 두 문장이 있으면, 첫 번째 VAD 구간에는 첫 번째 문장 할당

def match_vad_to_transcript(vad_segments, transcript_segments):
    """
    VAD 구간과 STT 문장을 매칭
    Returns: 구간별 텍스트 매핑
    """
    mapping = {}
    for i, (vad_seg, trans_seg) in enumerate(zip(vad_segments, transcript_segments)):
        mapping[f"segment_{i}"] = {
            "vad_time": (vad_seg['start_ms'], vad_seg['end_ms']),
            "text": trans_seg,
            "confidence": vad_seg['confidence']
        }
    return mapping
```

---

### C. 저작권 및 라이선스

```
주의사항:
- YouTube 콘텐츠: 개인 연구용만 사용 (공개 금지)
- ASMR 제작자 저작권 존중
- 생성된 데이터는 내부용으로만 사용
- 상업용 이용 시 별도 라이선스 필요
```

---

### D. 성능 및 비용 최적화

```python
# 1. 병렬 처리
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_file, f) for f in files]
    results = [f.result() for f in futures]

# 2. 배치 처리
batch_size = 32
for i in range(0, len(files), batch_size):
    batch = files[i:i+batch_size]
    process_batch(batch)

# 3. GPU 활용
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. 메모리 효율화
# - 스트리밍 오디오 처리
# - 청크 단위 처리
# - 임시 파일 정리
```

---

### E. 에러 복구 및 재시도 전략

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def download_with_retry(video_id):
    """3회 재시도, 지수 백오프"""
    return download_video(video_id)

# 부분 실패 처리
processed_files = load_checkpoint("last_processed.json")
remaining_files = [f for f in all_files if f not in processed_files]
```

---

### F. 데이터 재현성(Reproducibility)

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    """모든 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 사용 시점
set_seed(config['seed'])

# 생성 로그 기록
generation_metadata = {
    "seed": 42,
    "config_version": "1.0",
    "timestamp": "2024-03-09T10:30:00Z",
    "used_playlists": [...],
    "model_versions": {
        "stt": "whisper-base-20230314",
        "vad": "pyannote/segmentation-2.1"
    }
}
```

---

## 장점 요약

### 데이터셋 관점

| 장점 | 설명 |
|------|------|
| **정확한 Ground Truth** | 수동 라벨링 제거, 자동화된 정답 생성 |
| **음성 구간 정보** | VAD로 정확한 음성/침묵 구간 파악 |
| **다양한 SNR** | 같은 transcript로 여러 배경소음 조건 생성 |
| **확장 가능성** | 새로운 ASMR/소음 데이터 지속 추가 가능 |

---

### 개발 관점

| 장점 | 설명 |
|------|------|
| **자동화** | 수작업 최소화, 빠른 데이터셋 생성 |
| **재현성** | 같은 조건으로 데이터셋 재생성 가능 |
| **품질 관리** | 체계적인 검증 프로세스 |
| **추적 가능성** | 모든 처리 과정 로깅 및 메타데이터 기록 |

---

### 파인튜닝 관점

| 장점 | 설명 |
|------|------|
| **성능 평가** | Plain 모델 대비 정량적 성능 비교 가능 |
| **신뢰도 기반 학습** | Confidence score로 가중치 조정 |
| **음성 구간 학습** | VAD 정보로 부분 학습 가능 |
| **실제 상황 대비** | 다양한 SNR로 강건한 모델 학습 |

---

## 실행 순서

### Phase 1: 초기 구성 (1주)
1. 설정 파일 작성 (playlist, generation, quality, seed config)
2. 파이프라인 스크립트 기본 구조 구현
3. 단위 테스트 작성

### Phase 2: 파이프라인 구현 (2주)
1. YouTube 다운로드 모듈 구현 & 테스트
2. 품질 검증 모듈 구현 & 테스트
3. STT + VAD 모듈 구현 & 테스트
4. 배경소음 합성 모듈 구현 & 테스트

### Phase 3: 통합 테스트 (1주)
1. 전체 파이프라인 통합 테스트
2. 메타데이터 검증
3. 데이터셋 생성 및 검증

### Phase 4: 최적화 (1주)
1. 성능 최적화 (병렬 처리, 배치 처리)
2. 에러 처리 및 복구 로직 강화
3. 문서화 및 가이드 작성

---

## 참고 자료

### YouTube 다운로드
- [yt-dlp 문서](https://github.com/yt-dlp/yt-dlp)
- [youtube-dl 대체로 권장](https://github.com/yt-dlp/yt-dlp)

### STT 모델
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Whisper 한국어 가이드](https://github.com/openai/whisper/discussions/categories/korean)

### VAD
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [Silero VAD](https://github.com/snakers4/silero-vad)

### 오디오 처리
- [Librosa 문서](https://librosa.org/)
- [Soundfile 문서](https://python-soundfile.readthedocs.io/)

---

## 변경 로그

| 버전 | 날짜 | 변경 사항 |
|------|------|----------|
| 1.0 | 2024-03-09 | 초안 작성 |

