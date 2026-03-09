#!/usr/bin/env python
"""
파일럿 테스트 실행 스크립트
"""

import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.stt_core.input_providers.file_provider import FileAudioProvider
from backend.stt_core.audio_input.audio_input_module import AudioInputModule
from backend.stt_core.pipeline.config import Config


# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

    try:
        # 1. 설정 로드
        logger.info("\\n[1/5] 설정 로드 중...")
        config = Config()
        logger.info(f"✓ 설정 로드 완료")

        # 2. 입력 소스 생성 (파일)
        logger.info("\\n[2/5] 파일 입력 소스 생성 중...")
        if not os.path.exists(audio_file_path):
            logger.error(f"✗ 파일을 찾을 수 없음: {audio_file_path}")
            logger.info("💡 테스트 파일을 생성하려면:")
            logger.info("   python scripts/generate_test_audio.py")
            return

        file_provider = FileAudioProvider(audio_file_path)
        logger.info(f"✓ 파일 입력 소스 생성 완료")

        # 3. 입력 모듈 생성
        logger.info("\\n[3/5] 입력 모듈 초기화 중...")
        audio_input_module = AudioInputModule(file_provider)
        logger.info(f"✓ 입력 모듈 초기화 완료")

        # 4. 음성 수신
        logger.info("\\n[4/5] 음성 파일 읽기 중...")
        audio_input_module.connect()
        audio_data = audio_input_module.receive_and_parse()
        logger.info(f"✓ 음성 데이터 수신 완료:")
        logger.info(f"  - 길이: {audio_data.duration_ms:.2f}ms")
        logger.info(f"  - 샘플 레이트: {audio_data.sample_rate}Hz")
        logger.info(f"  - 채널: {audio_data.channels}")
        logger.info(f"  - 포맷: {audio_data.format}")

        # 5. 메타데이터 출력
        logger.info("\\n[5/5] 수신 정보:")
        source_info = audio_input_module.get_source_info()
        for key, value in source_info.items():
            logger.info(f"  - {key}: {value}")

        logger.info("\\n" + "=" * 60)
        logger.info("✅ 파일럿 테스트 성공!")
        logger.info("=" * 60)

        audio_input_module.disconnect()

        return audio_data

    except Exception as e:
        logger.error(f"\\n✗ 파일럿 테스트 실패: {e}", exc_info=True)
        logger.info("\\n🔍 문제 해결:")
        logger.info("1. 파일 경로 확인")
        logger.info("2. 파일이 올바른 오디오 포맷인지 확인 (WAV, MP3 등)")
        logger.info("3. 의존성 설치 확인: pip install librosa")
        return None


if __name__ == "__main__":
    # 테스트 음성 파일 경로
    test_audio_path = "test_audio.wav"

    # 파일 없으면 기본 경로 사용
    if not os.path.exists(test_audio_path):
        logger.warning(f"테스트 파일 없음: {test_audio_path}")
        logger.info("다음을 실행하여 테스트 파일을 생성하세요:")
        logger.info("  python scripts/generate_test_audio.py")
    else:
        run_pilot_test(test_audio_path)
