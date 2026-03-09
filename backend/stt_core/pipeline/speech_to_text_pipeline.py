"""
SpeechToTextPipeline

오디오 입력 → 전처리 → STT 인식 → 결과

AudioPreprocessingModule을 실제 파이프라인에 통합하는 방법을 보여줍니다.
"""

import numpy as np
import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# 데이터 클래스
# ============================================================================


@dataclass
class AudioData:
    """오디오 데이터"""

    audio: np.ndarray
    sample_rate: int
    channels: int
    duration_ms: int


@dataclass
class STTResult:
    """STT 인식 결과"""

    text: str
    confidence: float
    duration_ms: float
    language: str = "ko"


# ============================================================================
# STT 모델 인터페이스
# ============================================================================


class STTModel(ABC):
    """STT 모델 추상 클래스"""

    @abstractmethod
    def recognize(self, audio: np.ndarray) -> STTResult:
        """
        오디오에서 텍스트 인식

        Parameters:
        -----------
        audio : np.ndarray
            16kHz, float32 오디오

        Returns:
        --------
        STTResult
            인식 결과
        """
        pass


class DummySTTModel(STTModel):
    """테스트용 더미 STT 모델"""

    def __init__(self):
        self.test_results = [
            "안녕하세요",
            "반갑습니다",
            "오늘 날씨가 좋네요",
            "음성 인식 테스트입니다",
        ]
        self.call_count = 0

    def recognize(self, audio: np.ndarray) -> STTResult:
        """더미 인식"""
        result_text = self.test_results[self.call_count % len(self.test_results)]
        self.call_count += 1

        # 오디오 길이에 따른 신뢰도
        confidence = min(0.95, 0.5 + len(audio) / 100000)
        duration_ms = len(audio) * 1000 / 16000

        return STTResult(
            text=result_text,
            confidence=confidence,
            duration_ms=duration_ms,
            language="ko",
        )


# ============================================================================
# SpeechToTextPipeline
# ============================================================================


class SpeechToTextPipeline:
    """
    음성 인식 파이프라인

    오디오 입력 → 전처리 → STT 인식 → 결과
    """

    def __init__(
        self,
        preprocessor,  # AudioPreprocessingModule
        stt_model: STTModel,
        config: Optional[Dict] = None,
    ):
        """
        파이프라인 초기화

        Parameters:
        -----------
        preprocessor : AudioPreprocessingModule
            오디오 전처리 모듈
        stt_model : STTModel
            STT 인식 모델
        config : dict, optional
            파이프라인 설정
        """
        self.preprocessor = preprocessor
        self.stt_model = stt_model
        self.config = config or {}

        # 설정
        self.streaming_mode = self.config.get("streaming_mode", False)
        self.save_preprocessed = self.config.get("save_preprocessed", False)
        self.verbose = self.config.get("verbose", True)

        # 통계
        self.total_processed = 0
        self.total_recognized = 0
        self.avg_confidence = 0.0
        self.preprocessed_audios = []  # 전처리된 오디오 저장

        logger.info(
            f"SpeechToTextPipeline initialized\n"
            f"  - Streaming mode: {self.streaming_mode}\n"
            f"  - Save preprocessed: {self.save_preprocessed}"
        )

    # ========================================================================
    # 배치 처리
    # ========================================================================

    def recognize_file(self, audio_data: AudioData) -> STTResult:
        """
        파일 기반 음성 인식 (배치 처리)

        Parameters:
        -----------
        audio_data : AudioData
            오디오 데이터

        Returns:
        --------
        STTResult
            인식 결과

        Example:
        --------
        result = pipeline.recognize_file(audio_data)
        print(f"결과: {result.text} (신뢰도: {result.confidence:.2%})")
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🎵 음성 인식 시작")
            print(
                f"  입력: {audio_data.sample_rate}Hz, {audio_data.channels}ch, "
                f"{audio_data.duration_ms}ms"
            )
            print(f"{'='*70}")

        # Step 1: 전처리
        if self.verbose:
            print(f"\n[Step 1/3] 오디오 전처리 중...")

        preprocessed = self.preprocessor.process(audio_data)

        if self.save_preprocessed:
            self.preprocessed_audios.append(preprocessed)

        if self.verbose:
            print(f"  ✅ 전처리 완료: {len(preprocessed)} 샘플")

        # Step 2: STT 인식
        if self.verbose:
            print(f"\n[Step 2/3] STT 인식 중...")

        result = self.stt_model.recognize(preprocessed)

        if self.verbose:
            print(f"  ✅ 인식 완료")

        # Step 3: 결과 반환
        if self.verbose:
            print(f"\n[Step 3/3] 결과 정리 중...")
            print(f"\n{'─'*70}")
            print(f"📝 인식 결과")
            print(f"{'─'*70}")
            print(f"텍스트:     {result.text}")
            print(f"신뢰도:     {result.confidence:.2%}")
            print(f"처리시간:   {result.duration_ms:.0f}ms")
            print(f"언어:       {result.language}")
            print(f"{'='*70}\n")

        # 통계 업데이트
        self.total_processed += 1
        self.total_recognized += 1

        alpha = 1.0 / self.total_recognized
        self.avg_confidence = (
            1 - alpha
        ) * self.avg_confidence + alpha * result.confidence

        return result

    # ========================================================================
    # 실시간 처리
    # ========================================================================

    def start_streaming(self):
        """실시간 처리 시작"""
        self.preprocessor.reset_streaming_state()
        logger.info("Streaming started, preprocessor state reset")

    def recognize_chunk(self, chunk: np.ndarray) -> Optional[STTResult]:
        """
        청크 단위 음성 인식 (실시간 처리)

        Parameters:
        -----------
        chunk : np.ndarray
            음성 청크 (float32)

        Returns:
        --------
        STTResult or None
            인식 결과 (처리 가능한 경우만)

        Example:
        --------
        pipeline.start_streaming()

        for chunk in audio_stream:
            result = pipeline.recognize_chunk(chunk)
            if result:
                print(f"결과: {result.text}")
        """
        # Step 1: 실시간 전처리
        preprocessed = self.preprocessor.process_streaming_chunk(chunk)

        if self.save_preprocessed:
            self.preprocessed_audios.append(preprocessed)

        # Step 2: STT 인식
        result = self.stt_model.recognize(preprocessed)

        # Step 3: 통계 업데이트
        self.total_processed += 1
        if result.text:
            self.total_recognized += 1

            alpha = 1.0 / self.total_recognized
            self.avg_confidence = (
                1 - alpha
            ) * self.avg_confidence + alpha * result.confidence

        if self.verbose and result.text:
            print(f"🎙️ {result.text} ({result.confidence:.2%})")

        return result if result.text else None

    def stop_streaming(self):
        """실시간 처리 종료"""
        logger.info("Streaming stopped")

    # ========================================================================
    # 통계 및 정보
    # ========================================================================

    def get_statistics(self) -> Dict:
        """
        처리 통계 조회

        Returns:
        --------
        dict
            통계 정보
        """
        return {
            "total_processed": self.total_processed,
            "total_recognized": self.total_recognized,
            "avg_confidence": self.avg_confidence,
            "recognition_rate": (
                self.total_recognized / self.total_processed
                if self.total_processed > 0
                else 0
            ),
        }

    def print_statistics(self):
        """통계 출력"""
        stats = self.get_statistics()

        print(f"\n{'='*70}")
        print(f"📊 처리 통계")
        print(f"{'='*70}")
        print(f"총 처리 건수:      {stats['total_processed']}")
        print(f"인식 성공:        {stats['total_recognized']}")
        print(f"인식률:          {stats['recognition_rate']:.2%}")
        print(f"평균 신뢰도:      {stats['avg_confidence']:.2%}")
        print(f"{'='*70}\n")

    def reset(self):
        """파이프라인 초기화"""
        self.total_processed = 0
        self.total_recognized = 0
        self.avg_confidence = 0.0
        self.preprocessed_audios = []
        self.preprocessor.reset_streaming_state()
        logger.info("Pipeline reset")


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/mnt/user-data/outputs")

    from backend.stt_core.preprocessing.audio_preprocessing_module import (
        AudioPreprocessingModule,
    )

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("SpeechToTextPipeline 사용 예시")
    print("=" * 70)

    # --------
    # 1. 모듈 초기화
    # --------
    print("\n1️⃣ 모듈 초기화")
    print("-" * 70)

    preprocessor = AudioPreprocessingModule()
    stt_model = DummySTTModel()

    pipeline = SpeechToTextPipeline(
        preprocessor=preprocessor,
        stt_model=stt_model,
        config={"streaming_mode": False, "save_preprocessed": True, "verbose": True},
    )

    print("✅ 파이프라인 초기화 완료")

    # --------
    # 2. 배치 처리 테스트
    # --------
    print("\n2️⃣ 배치 처리 테스트")
    print("-" * 70)

    # 테스트 오디오 생성
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.05 * np.sin(2 * np.pi * 4000 * t) + 0.02 * np.random.randn(len(t))

    audio_data = AudioData(
        audio=audio.astype(np.float32),
        sample_rate=sr,
        channels=1,
        duration_ms=int(duration * 1000),
    )

    # 인식
    result = pipeline.recognize_file(audio_data)

    print(f"결과: {result.text}")
    print(f"신뢰도: {result.confidence:.2%}")

    # --------
    # 3. 실시간 처리 테스트
    # --------
    print("\n3️⃣ 실시간 처리 테스트")
    print("-" * 70)

    pipeline.reset()
    pipeline.start_streaming()

    chunk_size = 16000
    num_chunks = 3

    print(f"\n청크 단위 처리 ({num_chunks}개 청크):\n")

    for i in range(num_chunks):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]

        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        result = pipeline.recognize_chunk(chunk)

    pipeline.stop_streaming()

    # --------
    # 4. 전처리된 오디오 확인
    # --------
    print("\n4️⃣ 전처리된 오디오 정보")
    print("-" * 70)

    if pipeline.preprocessed_audios:
        print(f"\n저장된 전처리 오디오: {len(pipeline.preprocessed_audios)}개\n")

        for i, preprocessed in enumerate(pipeline.preprocessed_audios):
            min_val = np.min(preprocessed)
            max_val = np.max(preprocessed)
            rms = np.sqrt(np.mean(preprocessed**2))

            print(f"[배치 {i+1}]")
            print(f"  길이: {len(preprocessed)} 샘플")
            print(f"  범위: [{min_val:.4f}, {max_val:.4f}]")
            print(f"  RMS: {rms:.6f}")
            print()

    # --------
    # 5. 통계 출력
    # --------
    print("\n5️⃣ 처리 통계")
    print("-" * 70)

    pipeline.print_statistics()

    print("\n✅ 모든 테스트 완료!")
