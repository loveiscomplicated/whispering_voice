"""
AudioPreprocessingModule 검증 스크립트

pytest 없이 순수 Python으로 실행 가능한 테스트
"""

import numpy as np
import logging
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, "/mnt/user-data/outputs")

from backend.stt_core.preprocessing.audio_preprocessing_module import (
    AudioPreprocessingModule,
)

logging.basicConfig(level=logging.WARNING)

# ============================================================================
# 테스트 데이터 클래스
# ============================================================================


@dataclass
class AudioData:
    audio: np.ndarray
    sample_rate: int
    channels: int
    duration_ms: int


# ============================================================================
# 테스트 클래스
# ============================================================================


class TestAudioPreprocessingModule:
    """AudioPreprocessingModule 검증 테스트"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests_run = 0

    def _assert_equal(self, actual, expected, tolerance=1e-6, test_name=""):
        """값 비교"""
        if isinstance(expected, (int, float)):
            if abs(actual - expected) <= tolerance:
                return True
            else:
                print(f"  ❌ {test_name}: Expected {expected}, got {actual}")
                return False
        elif isinstance(expected, np.ndarray):
            if np.allclose(actual, expected, atol=tolerance):
                return True
            else:
                print(f"  ❌ {test_name}: Arrays not close")
                return False
        else:
            if actual == expected:
                return True
            else:
                print(f"  ❌ {test_name}: Expected {expected}, got {actual}")
                return False

    def _assert_true(self, condition, test_name=""):
        """조건 확인"""
        if condition:
            return True
        else:
            print(f"  ❌ {test_name}: Condition is False")
            return False

    def _assert_in_range(self, value, min_val, max_val, test_name=""):
        """범위 확인"""
        if min_val <= value <= max_val:
            return True
        else:
            print(f"  ❌ {test_name}: {value} not in [{min_val}, {max_val}]")
            return False

    def run_test(self, test_func, test_name):
        """테스트 실행"""
        self.tests_run += 1

        try:
            result = test_func()

            if result:
                self.passed += 1
                print(f"✅ {test_name}")
                return True
            else:
                self.failed += 1
                print(f"❌ {test_name}")
                return False

        except Exception as e:
            self.failed += 1
            print(f"❌ {test_name}: {type(e).__name__}: {e}")
            return False

    # ========================================================================
    # 테스트 케이스들
    # ========================================================================

    def test_initialization(self):
        """Test 1: 모듈 초기화"""
        preprocessor = AudioPreprocessingModule()

        return (
            self._assert_equal(
                preprocessor.target_sample_rate, 16000, test_name="target_sample_rate"
            )
            and self._assert_equal(
                preprocessor.chunk_size, 16000, test_name="chunk_size"
            )
            and self._assert_equal(
                preprocessor.high_freq_coef, 0.97, test_name="high_freq_coef"
            )
            and self._assert_equal(
                preprocessor.noise_alpha, 2.0, test_name="noise_alpha"
            )
            and self._assert_equal(
                preprocessor.vad_threshold, 0.02, test_name="vad_threshold"
            )
        )

    def test_custom_config(self):
        """Test 2: 커스텀 설정"""
        config = {
            "target_sample_rate": 8000,
            "high_freq_coef": 0.95,
            "noise_alpha": 3.0,
        }

        preprocessor = AudioPreprocessingModule(config)

        return (
            self._assert_equal(
                preprocessor.target_sample_rate, 8000, test_name="custom_sr"
            )
            and self._assert_equal(
                preprocessor.high_freq_coef, 0.95, test_name="custom_coef"
            )
            and self._assert_equal(
                preprocessor.noise_alpha, 3.0, test_name="custom_alpha"
            )
        )

    def test_empty_input(self):
        """Test 3: 빈 입력 처리"""
        preprocessor = AudioPreprocessingModule()

        audio_data = AudioData(
            audio=np.array([]), sample_rate=16000, channels=1, duration_ms=0
        )

        try:
            preprocessor.process(audio_data)
            return False  # 예외 발생해야 함
        except ValueError:
            return True

    def test_mono_audio_processing(self):
        """Test 4: 모노 오디오 처리"""
        preprocessor = AudioPreprocessingModule()

        # 테스트 오디오 생성
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.1 * np.sin(2 * np.pi * 1000 * t)

        audio_data = AudioData(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            channels=1,
            duration_ms=int(duration * 1000),
        )

        result = preprocessor.process(audio_data)

        return (
            self._assert_true(isinstance(result, np.ndarray), test_name="output_type")
            and self._assert_true(
                result.dtype in [np.float32, np.float64], test_name="output_dtype"
            )
            and self._assert_true(len(result) > 0, test_name="output_not_empty")
            and self._assert_true(
                np.min(result) >= -1.0 and np.max(result) <= 1.0,
                test_name="output_range",
            )
        )

    def test_stereo_to_mono(self):
        """Test 5: 스테레오 → 모노 변환"""
        preprocessor = AudioPreprocessingModule()

        # 스테레오 오디오 생성
        sr = 16000
        duration = 0.5
        n_samples = int(sr * duration)

        left = 0.1 * np.sin(2 * np.pi * 1000 * np.linspace(0, duration, n_samples))
        right = 0.1 * np.cos(2 * np.pi * 1000 * np.linspace(0, duration, n_samples))

        stereo_audio = np.vstack([left, right])

        audio_data = AudioData(
            audio=stereo_audio.astype(np.float32),
            sample_rate=sr,
            channels=2,
            duration_ms=int(duration * 1000),
        )

        result = preprocessor.process(audio_data)

        return self._assert_true(
            result.ndim == 1, test_name="output_is_mono"
        ) and self._assert_true(len(result) > 0, test_name="output_not_empty")

    def test_sample_rate_normalization(self):
        """Test 6: 샘플 레이트 정규화"""
        preprocessor = AudioPreprocessingModule()

        # 22.05kHz 오디오
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.1 * np.sin(2 * np.pi * 1000 * t)

        audio_data = AudioData(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            channels=1,
            duration_ms=int(duration * 1000),
        )

        result = preprocessor.process(audio_data)

        # 16kHz로 변환되므로 샘플 수 감소
        expected_samples = int(len(audio) * 16000 / sr)

        return self._assert_true(
            abs(len(result) - expected_samples) <= 2,  # 작은 오차 허용
            test_name="sample_count_matches",
        )

    def test_clipping(self):
        """Test 7: 클리핑 (범위 제한)"""
        preprocessor = AudioPreprocessingModule()

        # 매우 큰 진폭의 신호
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio = 10.0 * np.sin(2 * np.pi * 1000 * t)  # 범위 초과

        audio_data = AudioData(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            channels=1,
            duration_ms=int(duration * 1000),
        )

        result = preprocessor.process(audio_data)

        return self._assert_true(
            np.min(result) >= -1.0 and np.max(result) <= 1.0, test_name="output_clipped"
        )

    def test_streaming_initialization(self):
        """Test 8: 실시간 처리 상태 초기화"""
        preprocessor = AudioPreprocessingModule()

        preprocessor.reset_streaming_state()

        state = preprocessor.get_streaming_state()

        return self._assert_true(
            state["has_noise_profile"] == False, test_name="noise_profile_none"
        )

    def test_streaming_chunk_processing(self):
        """Test 9: 실시간 청크 처리"""
        preprocessor = AudioPreprocessingModule()
        preprocessor.reset_streaming_state()

        # 청크 생성
        chunk_size = 16000
        chunk = 0.1 * np.random.randn(chunk_size).astype(np.float32)

        result = preprocessor.process_streaming_chunk(chunk)

        return (
            self._assert_true(isinstance(result, np.ndarray), test_name="result_type")
            and self._assert_equal(len(result), chunk_size, test_name="result_length")
            and self._assert_equal(result.dtype, np.float32, test_name="result_dtype")
        )

    def test_streaming_state_tracking(self):
        """Test 10: 실시간 노이즈 프로필 추적"""
        preprocessor = AudioPreprocessingModule()
        preprocessor.reset_streaming_state()

        # 첫 청크 (프로필 초기화)
        chunk1 = 0.1 * np.random.randn(16000).astype(np.float32)
        preprocessor.process_streaming_chunk(chunk1)

        state1 = preprocessor.get_streaming_state()

        # 두 번째 청크 (프로필 유지)
        chunk2 = 0.1 * np.random.randn(16000).astype(np.float32)
        preprocessor.process_streaming_chunk(chunk2)

        state2 = preprocessor.get_streaming_state()

        return self._assert_true(
            state1["has_noise_profile"], test_name="profile_after_chunk1"
        ) and self._assert_true(
            state2["has_noise_profile"], test_name="profile_after_chunk2"
        )

    def test_multiple_files_processing(self):
        """Test 11: 여러 파일 순차 처리"""
        preprocessor = AudioPreprocessingModule()

        results = []

        for _ in range(3):
            sr = 16000
            duration = 0.5
            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.1 * np.sin(2 * np.pi * 1000 * t)

            audio_data = AudioData(
                audio=audio.astype(np.float32),
                sample_rate=sr,
                channels=1,
                duration_ms=int(duration * 1000),
            )

            result = preprocessor.process(audio_data)
            results.append(result)

        return self._assert_equal(
            len(results), 3, test_name="num_results"
        ) and self._assert_true(
            all(isinstance(r, np.ndarray) for r in results),
            test_name="all_results_valid",
        )

    def test_rms_normalization(self):
        """Test 12: RMS 정규화"""
        preprocessor = AudioPreprocessingModule()

        # 충분한 크기의 신호 (VAD가 0으로 만들지 않도록)
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.15 * np.sin(2 * np.pi * 1000 * t)  # 충분히 큰 진폭

        audio_data = AudioData(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            channels=1,
            duration_ms=int(duration * 1000),
        )

        result = preprocessor.process(audio_data)

        # RMS가 정규화되었는지 확인
        rms = np.sqrt(np.mean(result**2))

        # 충분한 신호가 있으면 RMS가 0이 아님
        return self._assert_true(
            rms > 0.05, test_name="rms_normalization"  # 정규화되었다면 충분한 RMS
        )

    def test_high_frequency_emphasis(self):
        """Test 13: 고주파 강화"""
        preprocessor = AudioPreprocessingModule()

        # 원본 vs 강화된 신호
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))

        # 저주파 신호
        audio = 0.1 * np.sin(2 * np.pi * 1000 * t)

        audio_data = AudioData(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            channels=1,
            duration_ms=int(duration * 1000),
        )

        result = preprocessor.process(audio_data)

        # 고주파 강화는 처리 결과에 영향을 줌
        return self._assert_true(
            isinstance(result, np.ndarray), test_name="result_valid"
        ) and self._assert_true(len(result) > 0, test_name="result_not_empty")

    def test_noise_removal(self):
        """Test 14: 노이즈 제거"""
        preprocessor = AudioPreprocessingModule()

        # 신호 + 노이즈
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))

        signal = 0.1 * np.sin(2 * np.pi * 1000 * t)
        noise = 0.05 * np.random.randn(len(signal))
        noisy_audio = signal + noise

        audio_data = AudioData(
            audio=noisy_audio.astype(np.float32),
            sample_rate=sr,
            channels=1,
            duration_ms=int(duration * 1000),
        )

        result = preprocessor.process(audio_data)

        return self._assert_true(
            isinstance(result, np.ndarray), test_name="result_valid"
        ) and self._assert_true(len(result) > 0, test_name="result_not_empty")

    def test_vad_operation(self):
        """Test 15: VAD (음성 활동 감지)"""
        preprocessor = AudioPreprocessingModule()

        # 음성 + 침묵
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # 처음 0.5초: 신호, 나머지: 침묵
        audio = np.zeros(int(sr * duration))
        audio[: int(sr * 0.5)] = 0.1 * np.sin(2 * np.pi * 1000 * t[: int(sr * 0.5)])

        audio_data = AudioData(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            channels=1,
            duration_ms=int(duration * 1000),
        )

        result = preprocessor.process(audio_data)

        # 침묵 부분에 0이 있는지 확인
        return self._assert_true(np.any(result == 0), test_name="silence_zeroed")

    def test_performance_batch(self):
        """Test 16: 성능 테스트 (배치)"""
        preprocessor = AudioPreprocessingModule()

        # 2초 오디오
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.1 * np.sin(2 * np.pi * 1000 * t)

        audio_data = AudioData(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            channels=1,
            duration_ms=int(duration * 1000),
        )

        start = time.time()
        result = preprocessor.process(audio_data)
        elapsed = time.time() - start

        # 2초 오디오는 500ms 이내에 처리되어야 함
        return self._assert_true(
            elapsed < 0.5, test_name=f"processing_time ({elapsed*1000:.1f}ms)"
        )

    def test_performance_streaming(self):
        """Test 17: 성능 테스트 (실시간)"""
        preprocessor = AudioPreprocessingModule()
        preprocessor.reset_streaming_state()

        chunk_size = 16000
        num_chunks = 3

        start = time.time()

        for _ in range(num_chunks):
            chunk = 0.1 * np.random.randn(chunk_size).astype(np.float32)
            preprocessor.process_streaming_chunk(chunk)

        elapsed = time.time() - start

        # 3개 청크(3초)는 500ms 이내에 처리되어야 함
        return self._assert_true(
            elapsed < 0.5, test_name=f"streaming_time ({elapsed*1000:.1f}ms)"
        )

    def test_dtype_consistency(self):
        """Test 18: 데이터 타입 일관성"""
        preprocessor = AudioPreprocessingModule()

        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))

        # 여러 입력 타입 테스트
        for dtype in [np.float32, np.float64]:
            audio = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(dtype)

            audio_data = AudioData(
                audio=audio,
                sample_rate=sr,
                channels=1,
                duration_ms=int(duration * 1000),
            )

            result = preprocessor.process(audio_data)

            if result.dtype != np.float32:
                return False

        return True

    def test_edge_case_very_short_audio(self):
        """Test 19: 엣지 케이스 - 매우 짧은 오디오"""
        preprocessor = AudioPreprocessingModule()

        # 100ms 오디오
        sr = 16000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.1 * np.sin(2 * np.pi * 1000 * t)

        audio_data = AudioData(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            channels=1,
            duration_ms=int(duration * 1000),
        )

        result = preprocessor.process(audio_data)

        return self._assert_true(
            len(result) > 0, test_name="output_not_empty"
        ) and self._assert_equal(result.dtype, np.float32, test_name="output_dtype")

    def test_edge_case_very_long_audio(self):
        """Test 20: 엣지 케이스 - 긴 오디오"""
        preprocessor = AudioPreprocessingModule()

        # 10초 오디오
        sr = 16000
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.1 * np.sin(2 * np.pi * 1000 * t)

        audio_data = AudioData(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            channels=1,
            duration_ms=int(duration * 1000),
        )

        result = preprocessor.process(audio_data)

        return self._assert_true(len(result) > 0, test_name="output_not_empty")

    # ========================================================================
    # 테스트 실행
    # ========================================================================

    def run_all_tests(self):
        """모든 테스트 실행"""
        print("=" * 70)
        print("AudioPreprocessingModule 검증 테스트")
        print("=" * 70)
        print()

        tests = [
            (self.test_initialization, "Test 1: 모듈 초기화"),
            (self.test_custom_config, "Test 2: 커스텀 설정"),
            (self.test_empty_input, "Test 3: 빈 입력 처리"),
            (self.test_mono_audio_processing, "Test 4: 모노 오디오 처리"),
            (self.test_stereo_to_mono, "Test 5: 스테레오 → 모노 변환"),
            (self.test_sample_rate_normalization, "Test 6: 샘플 레이트 정규화"),
            (self.test_clipping, "Test 7: 클리핑"),
            (self.test_streaming_initialization, "Test 8: 실시간 상태 초기화"),
            (self.test_streaming_chunk_processing, "Test 9: 실시간 청크 처리"),
            (self.test_streaming_state_tracking, "Test 10: 노이즈 프로필 추적"),
            (self.test_multiple_files_processing, "Test 11: 여러 파일 처리"),
            (self.test_rms_normalization, "Test 12: RMS 정규화"),
            (self.test_high_frequency_emphasis, "Test 13: 고주파 강화"),
            (self.test_noise_removal, "Test 14: 노이즈 제거"),
            (self.test_vad_operation, "Test 15: VAD 동작"),
            (self.test_performance_batch, "Test 16: 성능 - 배치"),
            (self.test_performance_streaming, "Test 17: 성능 - 실시간"),
            (self.test_dtype_consistency, "Test 18: 데이터 타입 일관성"),
            (
                self.test_edge_case_very_short_audio,
                "Test 19: 엣지 케이스 - 짧은 오디오",
            ),
            (self.test_edge_case_very_long_audio, "Test 20: 엣지 케이스 - 긴 오디오"),
        ]

        for test_func, test_name in tests:
            self.run_test(test_func, test_name)

        # 결과 출력
        print()
        print("=" * 70)
        print("📊 테스트 결과 요약")
        print("=" * 70)
        print(f"총 테스트:     {self.tests_run}")
        print(f"성공:          {self.passed}")
        print(f"실패:          {self.failed}")

        if self.failed == 0:
            print(f"\n✅ 모든 테스트 통과! ({self.passed}/{self.tests_run})")
        else:
            print(f"\n❌ {self.failed}개 테스트 실패")

        print("=" * 70)
        print()

        return self.failed == 0


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    tester = TestAudioPreprocessingModule()
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)
