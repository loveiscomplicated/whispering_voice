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
