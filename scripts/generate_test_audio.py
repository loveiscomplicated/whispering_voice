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
