"""
AdvancedAudioPreprocessingModule - 실시간 처리 지원 버전

기존 AudioPreprocessingModule을 상속받아
실시간 스트리밍 처리 기능을 추가합니다.

사용:
  배치 처리: module.process(audio_data)
  실시간:   module.process_streaming_chunk(chunk)
"""

import numpy as np
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class AudioPreprocessingModule:
    """
    배치 + 실시간 스트리밍 모두 지원하는 전처리 모듈

    주요 기능:
    ✅ 배치 처리: 파일 전체 처리
    ✅ 실시간: 청크 단위 실시간 처리
    ✅ 상태 관리: 노이즈 프로필 유지
    ✅ 음성 연속성: 청크 경계에서 부드러운 전환
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        모듈 초기화

        Parameters:
        -----------
        config : dict, optional
            설정 딕셔너리
            - target_sample_rate: 목표 샘플 레이트 (기본: 16000)
            - chunk_size: 청크 크기 (기본: 16000)
            - high_freq_coef: 고주파 강화 계수 (기본: 0.97)
            - noise_alpha: 노이즈 제거 강도 (기본: 2.0)
            - vad_threshold: VAD 임계값 (기본: 0.02)
            - target_rms: RMS 정규화 목표값 (기본: 0.1)
            - noise_update_rate: 노이즈 프로필 업데이트율 (기본: 0.05)
        """
        self.config = config or {}

        # 기본 설정
        self.target_sample_rate = self.config.get("target_sample_rate", 16000)
        self.chunk_size = self.config.get("chunk_size", 16000)
        self.high_freq_coef = self.config.get("high_freq_coef", 0.97)
        self.noise_alpha = self.config.get("noise_alpha", 2.0)
        self.noise_beta = self.config.get("noise_beta", 0.1)
        self.vad_threshold = self.config.get("vad_threshold", 0.02)
        self.target_rms = self.config.get("target_rms", 0.1)

        # 실시간 처리용 상태
        self.streaming_mode = False
        self.noise_profile = None  # 이전 청크의 노이즈 프로필
        self.noise_update_rate = self.config.get("noise_update_rate", 0.05)
        self.previous_chunk = None  # 이전 청크 (오버랩 처리용)

        logger.info(
            f"AudioPreprocessingModule initialized\n"
            f"  - Target SR: {self.target_sample_rate}Hz\n"
            f"  - Chunk size: {self.chunk_size}\n"
            f"  - Noise update rate: {self.noise_update_rate*100:.1f}%"
        )

    # ========================================================================
    # 배치 처리 메서드
    # ========================================================================

    def process(self, audio_data) -> np.ndarray:
        """
        배치 처리 (전체 파일)

        Parameters:
        -----------
        audio_data : AudioData
            입력 오디오 데이터

        Returns:
        --------
        np.ndarray
            처리된 오디오
        """
        if audio_data.audio is None or len(audio_data.audio) == 0:
            raise ValueError("Input audio is empty")

        audio = audio_data.audio.copy().astype(np.float32)

        logger.debug(
            f"Processing batch audio: "
            f"sr={audio_data.sample_rate}, "
            f"channels={audio_data.channels}, "
            f"duration={audio_data.duration_ms}ms"
        )

        # 배치 모드로 설정 (노이즈 프로필 초기화)
        self.streaming_mode = False
        self.noise_profile = None
        self.previous_chunk = None

        # 스테레오 → 모노
        if audio_data.channels > 1:
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            logger.debug("Converted stereo to mono")

        # 샘플 레이트 정규화
        if audio_data.sample_rate != self.target_sample_rate:
            audio = self._normalize_sample_rate(
                audio, audio_data.sample_rate, self.target_sample_rate
            )
            logger.debug(f"Resampled to {self.target_sample_rate}Hz")

        # 클리핑
        audio = np.clip(audio, -1.0, 1.0)

        # 청킹 처리
        processed_audio = self._process_by_chunks(audio)

        logger.debug(f"Preprocessing complete. Output shape: {processed_audio.shape}")

        return processed_audio

    def _normalize_sample_rate(self, audio, original_sr, target_sr):
        """샘플 레이트 정규화"""
        if original_sr == target_sr:
            return audio

        ratio = target_sr / original_sr
        new_length = int(len(audio) * ratio)
        old_indices = np.arange(len(audio))
        new_indices = np.arange(new_length) / ratio
        resampled = np.interp(new_indices, old_indices, audio)

        return resampled.astype(np.float32)

    def _process_by_chunks(self, audio: np.ndarray) -> np.ndarray:
        """청크 단위 배치 처리"""
        num_samples = len(audio)
        hop_size = self.chunk_size // 2
        processed_chunks = []

        for start in range(0, num_samples, hop_size):
            end = min(start + self.chunk_size, num_samples)
            chunk = audio[start:end]

            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))

            processed_chunk = self._process_chunk(chunk)
            processed_chunks.append(processed_chunk)

        if len(processed_chunks) == 1:
            return processed_chunks[0]

        # 오버랩 병합
        output = np.zeros(num_samples)
        window_count = np.zeros(num_samples)

        for i, chunk in enumerate(processed_chunks):
            start = i * hop_size
            end = min(start + self.chunk_size, num_samples)
            actual_len = end - start

            output[start:end] += chunk[:actual_len]
            window_count[start:end] += 1.0

        output = np.divide(output, window_count, where=window_count > 0, out=output)

        return output

    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """단일 청크 처리"""
        # 고주파 강화
        chunk = self._emphasize_high_frequency(chunk, self.high_freq_coef)

        # 노이즈 제거 (모드에 따라)
        if self.streaming_mode:
            chunk = self._remove_noise_streaming(chunk)
        else:
            chunk = self._remove_noise(chunk)

        # VAD
        chunk = self._voice_activity_detection(chunk, self.vad_threshold)

        # RMS 정규화
        chunk = self._normalize_rms(chunk, self.target_rms)

        return chunk

    def _emphasize_high_frequency(self, audio: np.ndarray, coef: float) -> np.ndarray:
        """고주파 강화 (Pre-emphasis Filter)"""
        emphasized = np.zeros_like(audio)
        emphasized[0] = audio[0]

        for n in range(1, len(audio)):
            emphasized[n] = audio[n] - coef * audio[n - 1]

        return emphasized

    def _remove_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        노이즈 제거 - 배치 처리용
        (기존 방식: 처음 10%를 노이즈로 가정)
        """
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        phase = np.angle(fft)

        # 처음 10%를 노이즈 프로필로 가정
        noise_profile = np.mean(
            magnitude[: len(magnitude) // 10], axis=0, keepdims=True
        )

        # 스펙트럼 감산
        subtracted = magnitude - self.noise_alpha * noise_profile
        subtracted = np.maximum(subtracted, self.noise_beta * magnitude)

        # IFFT
        reconstructed = subtracted * np.exp(1j * phase)
        output = np.fft.ifft(reconstructed).real

        return output.astype(np.float32)

    # ========================================================================
    # 실시간 처리 메서드 (신규) ⭐
    # ========================================================================

    def process_streaming_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        실시간 청크 처리

        스트리밍 입력(블루투스, 마이크 등)에서 청크를 받아서
        즉시 처리하여 반환합니다.

        Parameters:
        -----------
        chunk : np.ndarray
            입력 청크 (float32, 범위: -1.0 ~ 1.0)

        Returns:
        --------
        np.ndarray
            처리된 청크 (float32)

        Example:
        --------
        # 블루투스에서 오디오 수신
        preprocessor = AudioPreprocessingModule()

        for audio_chunk in bluetooth_stream:
            processed = preprocessor.process_streaming_chunk(audio_chunk)
            stt_model.recognize(processed)  # STT에 바로 전달
        """
        # 실시간 모드 활성화
        self.streaming_mode = True

        # 입력 검증
        if chunk is None or len(chunk) == 0:
            raise ValueError("Input chunk is empty")

        chunk = chunk.copy().astype(np.float32)
        chunk = np.clip(chunk, -1.0, 1.0)

        # 청크 처리
        result = self._process_chunk(chunk)

        return result

    def _remove_noise_streaming(self, chunk: np.ndarray) -> np.ndarray:
        """
        노이즈 제거 - 실시간 처리용 ⭐

        이전 청크의 노이즈 프로필을 유지하면서 처리하여
        음성의 연속성을 보장합니다.

        Parameters:
        -----------
        chunk : np.ndarray
            현재 청크

        Returns:
        --------
        np.ndarray
            노이즈 제거된 청크
        """
        # FFT 계산
        fft = np.fft.fft(chunk)
        magnitude = np.abs(fft)
        phase = np.angle(fft)

        # 첫 청크라면 노이즈 프로필 초기화
        if self.noise_profile is None:
            # 첫 청크의 처음 10%를 노이즈로 가정
            self.noise_profile = np.mean(
                magnitude[: len(magnitude) // 10], axis=0, keepdims=True
            )
            logger.debug("Initialized noise profile from first streaming chunk")

        # 스펙트럼 감산 (저장된 노이즈 프로필 사용)
        subtracted = magnitude - self.noise_alpha * self.noise_profile
        subtracted = np.maximum(subtracted, self.noise_beta * magnitude)

        # 현재 청크의 주파수 정보로 노이즈 프로필 점진적 업데이트
        # → 음성이 바뀔 때 자동으로 프로필 조정
        current_noise = np.mean(
            magnitude[: len(magnitude) // 10], axis=0, keepdims=True
        )

        self.noise_profile = (
            1 - self.noise_update_rate
        ) * self.noise_profile + self.noise_update_rate * current_noise

        # IFFT
        reconstructed = subtracted * np.exp(1j * phase)
        output = np.fft.ifft(reconstructed).real

        return output.astype(np.float32)

    # ========================================================================
    # 공통 처리 메서드
    # ========================================================================

    def _voice_activity_detection(
        self, audio: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Voice Activity Detection (VAD)"""
        frame_size = int(0.025 * self.target_sample_rate)
        padded = np.pad(audio, (frame_size // 2, frame_size // 2), mode="reflect")

        output = np.zeros_like(audio)

        for i in range(len(audio)):
            frame_start = i
            frame_end = i + frame_size
            frame = padded[frame_start:frame_end]

            energy = np.sqrt(np.mean(frame**2))

            if energy > threshold:
                output[i] = audio[i]

        return output

    def _normalize_rms(self, audio: np.ndarray, target_rms: float) -> np.ndarray:
        """RMS 정규화"""
        current_rms = np.sqrt(np.mean(audio**2))

        if current_rms < 1e-10:
            logger.warning("Current RMS is near zero, skipping normalization")
            return audio

        scaling_factor = target_rms / current_rms
        normalized = audio * scaling_factor
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized.astype(np.float32)

    # ========================================================================
    # 실시간 상태 관리 메서드
    # ========================================================================

    def reset_streaming_state(self):
        """
        실시간 처리 상태 초기화

        새로운 스트림 시작 시 호출하세요.
        """
        self.streaming_mode = False
        self.noise_profile = None
        self.previous_chunk = None
        logger.debug("Streaming state reset")

    def get_streaming_state(self) -> dict:
        """
        현재 실시간 처리 상태 조회

        Returns:
        --------
        dict
            현재 상태 정보
        """
        return {
            "streaming_mode": self.streaming_mode,
            "has_noise_profile": self.noise_profile is not None,
            "noise_profile_shape": (
                self.noise_profile.shape if self.noise_profile is not None else None
            ),
        }


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("AudioPreprocessingModule 사용 예시")
    print("=" * 70)

    # 모듈 초기화
    preprocessor = AudioPreprocessingModule()

    # --------
    # 예시 1: 배치 처리 (파일)
    # --------
    print("\n1️⃣ 배치 처리 (파일)")
    print("-" * 70)

    from dataclasses import dataclass

    @dataclass
    class AudioData:
        audio: np.ndarray
        sample_rate: int
        channels: int
        duration_ms: int

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

    start = time.time()
    result = preprocessor.process(audio_data)
    elapsed = time.time() - start

    print(f"✅ 배치 처리 완료")
    print(f"   입력: {len(audio)} 샘플 ({sr}Hz)")
    print(f"   출력: {len(result)} 샘플 ({preprocessor.target_sample_rate}Hz)")
    print(f"   처리 시간: {elapsed*1000:.2f}ms")

    # --------
    # 예시 2: 실시간 처리 (스트리밍)
    # --------
    print("\n2️⃣ 실시간 처리 (스트리밍)")
    print("-" * 70)

    # 상태 초기화
    preprocessor.reset_streaming_state()

    # 1초 청크를 3번 처리
    chunk_size = 16000  # 1초
    results = []

    for i in range(3):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]

        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        start = time.time()
        result_chunk = preprocessor.process_streaming_chunk(chunk)
        elapsed = time.time() - start

        results.append(result_chunk)
        state = preprocessor.get_streaming_state()

        print(
            f"   청크 {i+1}: {len(chunk)} → {len(result_chunk)} 샘플 "
            f"({elapsed*1000:.2f}ms, "
            f"노이즈프로필: {'✅' if state['has_noise_profile'] else '❌'})"
        )

    print(f"✅ 실시간 처리 완료")

    # --------
    # 예시 3: 성능 비교
    # --------
    print("\n3️⃣ 성능 비교")
    print("-" * 70)

    # 배치 처리
    start = time.time()
    result_batch = preprocessor.process(audio_data)
    time_batch = time.time() - start

    # 실시간 처리
    preprocessor.reset_streaming_state()
    start = time.time()
    for i in range(3):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        preprocessor.process_streaming_chunk(chunk)
    time_streaming = time.time() - start

    print(f"배치 처리:   {time_batch*1000:.2f}ms")
    print(f"실시간 처리: {time_streaming*1000:.2f}ms")
    print(f"차이:       {abs(time_batch-time_streaming)*1000:.2f}ms")
