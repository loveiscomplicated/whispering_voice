"""
Config 클래스 단위 테스트
"""

import json
import pytest
from pathlib import Path

from backend.stt_core.pipeline.config import Config


# ---------------------------------------------------------------------------
# 기본 설정 (파일 없는 경우)
# ---------------------------------------------------------------------------


def test_default_config_returned_when_file_missing(tmp_path):
    """존재하지 않는 경로 → 기본 설정 반환"""
    config = Config(str(tmp_path / "nonexistent.json"))

    assert config["stt"]["model"] == "whisper"
    assert config["pipeline"]["processingMode"] == "hybrid"
    assert config["stt"]["modelSize"] == "base"


def test_default_config_has_all_expected_keys(tmp_path):
    """기본 설정에 필수 키 존재"""
    config = Config(str(tmp_path / "missing.json"))

    for key in ("inputSource", "audioInput", "preprocessing", "stt", "pipeline"):
        assert key in config.config


# ---------------------------------------------------------------------------
# 파일에서 로드
# ---------------------------------------------------------------------------


def test_loads_valid_json_file(tmp_path):
    """정상 JSON 파일 로드"""
    data = {"stt": {"model": "custom_model"}, "pipeline": {"processingMode": "batch"}}
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data), encoding="utf-8")

    config = Config(str(cfg_file))

    assert config["stt"]["model"] == "custom_model"
    assert config["pipeline"]["processingMode"] == "batch"


def test_invalid_json_falls_back_to_default(tmp_path):
    """손상된 JSON → 기본 설정"""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{ invalid json !!!", encoding="utf-8")

    config = Config(str(cfg_file))

    assert config["stt"]["model"] == "whisper"


# ---------------------------------------------------------------------------
# get() — 점 표기법 조회
# ---------------------------------------------------------------------------


def test_get_nested_key(tmp_path):
    """점으로 구분된 중첩 키 조회"""
    data = {"stt": {"model": "whisper", "modelSize": "large"}}
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data))

    config = Config(str(cfg_file))

    assert config.get("stt.model") == "whisper"
    assert config.get("stt.modelSize") == "large"


def test_get_top_level_key(tmp_path):
    """단일 레벨 키 조회"""
    data = {"topLevel": "value"}
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data))

    config = Config(str(cfg_file))

    assert config.get("topLevel") == "value"


def test_get_missing_key_returns_none(tmp_path):
    """없는 키 → None"""
    config = Config(str(tmp_path / "missing.json"))

    assert config.get("not.exist") is None


def test_get_missing_key_returns_default(tmp_path):
    """없는 키에 default 지정"""
    config = Config(str(tmp_path / "missing.json"))

    assert config.get("not.exist", "fallback") == "fallback"


def test_get_non_dict_intermediate_returns_default(tmp_path):
    """중간 경로가 dict가 아닐 때 default 반환"""
    data = {"stt": "not_a_dict"}
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data))

    config = Config(str(cfg_file))

    assert config.get("stt.model", "fallback") == "fallback"


def test_get_default_config_nested(tmp_path):
    """기본 설정의 중첩 키 조회"""
    config = Config(str(tmp_path / "missing.json"))

    assert config.get("stt.model") == "whisper"
    assert config.get("preprocessing.targetSampleRate") == 16000


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


def test_getitem_existing_key(tmp_path):
    """딕셔너리처럼 접근"""
    config = Config(str(tmp_path / "missing.json"))

    stt = config["stt"]

    assert isinstance(stt, dict)
    assert "model" in stt


def test_getitem_missing_key_returns_empty_dict(tmp_path):
    """존재하지 않는 키 → 빈 딕셔너리"""
    config = Config(str(tmp_path / "missing.json"))

    result = config["nonexistent"]

    assert result == {}
