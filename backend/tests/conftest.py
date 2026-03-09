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
