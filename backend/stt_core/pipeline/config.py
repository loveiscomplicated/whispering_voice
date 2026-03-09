"""
파이프라인 설정 관리
"""

import json
import logging
from typing import Dict, Any
from pathlib import Path


logger = logging.getLogger(__name__)


class Config:
    """애플리케이션 설정 관리 클래스"""

    def __init__(self, config_path: str = "backend/config/config.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                logger.info(f"✓ Config loaded from {self.config_path}")
                return config
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return self._get_default_config()

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "inputSource": {"type": "file", "filePath": "test_audio.wav"},
            "audioInput": {"maxDuration_seconds": 300, "timeout_ms": 5000},
            "preprocessing": {"targetSampleRate": 16000, "chunkSize_seconds": 2},
            "stt": {"model": "whisper", "modelSize": "base"},
            "pipeline": {"processingMode": "hybrid"},
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정 값 조회

        Args:
            key: 설정 키 (점으로 구분, 예: "stt.model")
            default: 기본값

        Returns:
            설정 값
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def __getitem__(self, key: str) -> Dict:
        """딕셔너리처럼 접근"""
        return self.config.get(key, {})
