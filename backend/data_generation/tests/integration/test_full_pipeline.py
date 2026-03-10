"""Integration tests for the full data-generation pipeline.

All external ML libraries (whisper, pyannote, yt-dlp) are mocked.
Synthetic audio files are generated with numpy to exercise real I/O paths.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import yaml


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, duration_s: float = 2.0, sr: int = 16_000,
               amplitude: float = 0.3) -> Path:
    """Write a sine-wave WAV file of the given duration."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    audio = amplitude * np.sin(2 * np.pi * 440 * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return path


def _write_config(path: Path, raw_dir: str, stt_vad_dir: str,
                  synth_dir: str, dataset_dir: str, logs_dir: str) -> Path:
    cfg = {
        "youtube": {"playlist_ids": ["PLtest"], "max_videos_per_playlist": 2},
        "quality_validation": {
            "min_audio_length_ms": 500,
            "max_audio_length_ms": 60_000,
            "min_energy_db": -60.0,
            "max_energy_db": 0.0,
        },
        "stt": {"model": "whisper-base", "language": "ko",
                "min_confidence": 0.85, "device": "cpu"},
        "vad": {"model": "pyannote/segmentation",
                "threshold": 0.5, "min_speech_duration_ms": 200},
        "synthesis": {
            "snr_levels_db": [10],
            "noise_types": ["ambient"],
            "noise_distribution": {"ambient": 1.0},
        },
        "output_dirs": {
            "raw_downloads": raw_dir,
            "stt_and_vad": stt_vad_dir,
            "synthesized": synth_dir,
            "dataset": dataset_dir,
            "logs": logs_dir,
        },
        "reproducibility": {"seed": 0, "version": "1.0"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return path


# ---------------------------------------------------------------------------
# Shared fixture: tmp-dir workspace
# ---------------------------------------------------------------------------

@pytest.fixture()
def workspace(tmp_path) -> dict[str, Path]:
    """Create a minimal pipeline workspace with two sample WAV files."""
    raw = tmp_path / "raw_downloads"
    stt_vad = tmp_path / "stt_and_vad"
    synth = tmp_path / "synthesized"
    dataset = tmp_path / "dataset"
    logs = tmp_path / "logs"
    cfg_dir = tmp_path / "config"

    for d in (raw, stt_vad, synth, dataset, logs, cfg_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Two ASMR WAVs
    _write_wav(raw / "vid_aaa.wav")
    _write_wav(raw / "vid_bbb.wav")

    # Noise file
    noise_dir = raw / "noise" / "ambient"
    noise_dir.mkdir(parents=True, exist_ok=True)
    _write_wav(noise_dir / "amb_001.wav", amplitude=0.1)

    cfg_path = _write_config(
        cfg_dir / "generation.yaml",
        raw_dir=str(raw),
        stt_vad_dir=str(stt_vad),
        synth_dir=str(synth),
        dataset_dir=str(dataset),
        logs_dir=str(logs),
    )

    return {
        "root": tmp_path,
        "raw": raw,
        "stt_vad": stt_vad,
        "synth": synth,
        "dataset": dataset,
        "logs": logs,
        "config": cfg_path,
    }


# ---------------------------------------------------------------------------
# Stage 2: Quality Validation 1
# ---------------------------------------------------------------------------

class TestStage2QualityValidation1:
    def test_valid_files_pass(self, workspace):
        from src._2_quality_validation_1 import QualityValidator1  # noqa: PLC0415
        import logging

        cfg_path = str(workspace["config"])
        from src.utils.config import load_config
        config = load_config(cfg_path)

        validator = QualityValidator1(config=config, logger=logging.getLogger("t"))
        df = validator.validate_batch(str(workspace["raw"]))

        # Only WAVs in raw/ (noise sub-dir is excluded from direct listing)
        wav_count = sum(1 for p in workspace["raw"].iterdir()
                        if p.is_file() and p.suffix == ".wav")
        assert len(df) == wav_count
        assert df["passed"].all(), df[~df["passed"]][["file_path", "failure_reasons"]]

    def test_report_files_created(self, workspace, tmp_path):
        from src._2_quality_validation_1 import QualityValidator1
        import logging
        from src.utils.config import load_config

        config = load_config(str(workspace["config"]))
        validator = QualityValidator1(config=config, logger=logging.getLogger("t"))
        validator.validate_batch(str(workspace["raw"]))

        report_path = str(workspace["logs"] / "qv1_report.csv")
        validator.generate_report(report_path)

        assert Path(report_path).exists()
        passed_json = workspace["logs"] / "qv1_report_passed.json"
        assert passed_json.exists()


# ---------------------------------------------------------------------------
# Stage 4: Noise Synthesis
# ---------------------------------------------------------------------------

class TestStage4NoiseSynthesis:
    def test_wav_files_created(self, workspace):
        from src._4_synthesize_noise import NoiseSynthesizer
        import logging
        from src.utils.config import load_config

        config = load_config(str(workspace["config"]))
        synth = NoiseSynthesizer(config=config, logger=logging.getLogger("t"))

        results = synth.synthesize_batch(
            asmr_dir=str(workspace["raw"]),
            noise_dir=str(workspace["raw"] / "noise"),
            snr_levels=[10.0],
            output_dir=str(workspace["synth"]),
        )

        assert len(results) > 0
        for paths in results.values():
            for p in paths:
                assert Path(p).exists()

    def test_json_sidecars_created(self, workspace):
        from src._4_synthesize_noise import NoiseSynthesizer
        import logging
        from src.utils.config import load_config

        config = load_config(str(workspace["config"]))
        synth = NoiseSynthesizer(config=config, logger=logging.getLogger("t"))
        synth.synthesize_batch(
            asmr_dir=str(workspace["raw"]),
            noise_dir=str(workspace["raw"] / "noise"),
            snr_levels=[10.0],
            output_dir=str(workspace["synth"]),
        )

        json_files = list(workspace["synth"].rglob("*.json"))
        assert len(json_files) > 0


# ---------------------------------------------------------------------------
# Stage 5: Quality Validation 2
# ---------------------------------------------------------------------------

class TestStage5QualityValidation2:
    def _prepare_synth_files(self, workspace):
        """Pre-populate synthesized/ with WAV + JSON sidecar pairs."""
        snr_dir = workspace["synth"] / "snr_10" / "ambient"
        snr_dir.mkdir(parents=True, exist_ok=True)

        for name in ("vid_aaa_snr10_ambient", "vid_bbb_snr10_ambient"):
            wav_path = snr_dir / f"{name}.wav"
            _write_wav(wav_path)
            meta = {
                "audio_id": name.split("_snr")[0],
                "source_asmr": str(workspace["raw"] / (name.split("_snr")[0] + ".wav")),
                "source_noise": str(
                    workspace["raw"] / "noise" / "ambient" / "amb_001.wav"
                ),
                "noise_type": "ambient",
                "target_snr_db": 10.0,
                "fade_duration_ms": 50,
                "audio_characteristics": {
                    "format": "wav", "sample_rate": 16_000,
                    "channels": 1, "duration_ms": 2000.0,
                    "rms_energy_db": -20.0, "peak_amplitude": 0.9,
                },
            }
            with open(wav_path.with_suffix(".json"), "w") as f:
                json.dump(meta, f)

    def test_synthesized_files_pass(self, workspace):
        from src._5_quality_validation_2 import QualityValidator2
        import logging
        from src.utils.config import load_config

        self._prepare_synth_files(workspace)
        config = load_config(str(workspace["config"]))
        validator = QualityValidator2(config=config, logger=logging.getLogger("t"))
        df = validator.validate_batch(str(workspace["synth"]))

        assert not df.empty
        assert df["format_ok"].all()
        assert df["file_intact"].all()


# ---------------------------------------------------------------------------
# Stage 6: Dataset Generation
# ---------------------------------------------------------------------------

class TestStage6DatasetGeneration:
    def _populate_stage3_metadata(self, workspace):
        """Create fake Stage 3 metadata JSONs alongside the raw WAVs."""
        meta_dir = workspace["stt_vad"] / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)

        for audio_id in ("vid_aaa", "vid_bbb"):
            meta = {
                "audio_id": audio_id,
                "processing_pipeline_version": "1.0",
                "stt_result": {
                    "transcript": "테스트 문장입니다",
                    "language": "ko",
                    "confidence_score": 0.92,
                    "model_used": "whisper-base",
                    "model_version": "v20230314",
                },
                "vad_result": {
                    "segments": [],
                    "total_speech_duration_ms": 1500.0,
                    "total_silence_duration_ms": 500.0,
                    "speech_ratio": 0.75,
                },
                "audio_characteristics": {
                    "format": "wav", "sample_rate": 16_000,
                    "channels": 1, "duration_ms": 2000.0,
                    "rms_energy_db": -18.0, "peak_amplitude": 0.85,
                },
            }
            with open(meta_dir / f"{audio_id}_metadata.json", "w") as f:
                json.dump(meta, f)

            # Copy WAV to stt_vad dir so it is discoverable
            src = workspace["raw"] / f"{audio_id}.wav"
            if src.exists():
                shutil.copy(src, workspace["stt_vad"] / f"{audio_id}.wav")

    def _populate_synth_files(self, workspace):
        snr_dir = workspace["synth"] / "snr_10" / "ambient"
        snr_dir.mkdir(parents=True, exist_ok=True)
        for audio_id in ("vid_aaa", "vid_bbb"):
            name = f"{audio_id}_snr10_ambient"
            wav = snr_dir / f"{name}.wav"
            _write_wav(wav)
            meta = {
                "audio_id": audio_id,
                "source_asmr": str(workspace["raw"] / f"{audio_id}.wav"),
                "source_noise": str(workspace["raw"] / "noise" / "ambient" / "amb_001.wav"),
                "noise_type": "ambient",
                "target_snr_db": 10.0,
                "fade_duration_ms": 50,
                "audio_characteristics": {
                    "format": "wav", "sample_rate": 16_000, "channels": 1,
                    "duration_ms": 2000.0, "rms_energy_db": -20.0, "peak_amplitude": 0.9,
                },
            }
            with open(wav.with_suffix(".json"), "w") as f:
                json.dump(meta, f)

    def test_dataset_structure_created(self, workspace):
        from src._6_generate_finetuning_dataset import FinetuningDatasetGenerator
        import logging
        from src.utils.config import load_config

        self._populate_stage3_metadata(workspace)
        self._populate_synth_files(workspace)

        config = load_config(str(workspace["config"]))
        gen = FinetuningDatasetGenerator(config=config, logger=logging.getLogger("t"))
        manifest = gen.create_dataset(
            stt_and_vad_dir=str(workspace["stt_vad"]),
            synthesized_dir=str(workspace["synth"]),
            output_dir=str(workspace["dataset"]),
        )

        assert manifest
        assert manifest["total"] > 0
        assert (workspace["dataset"] / "manifest.json").exists()

    def test_split_dirs_contain_jsonl(self, workspace):
        from src._6_generate_finetuning_dataset import FinetuningDatasetGenerator
        import logging
        from src.utils.config import load_config

        self._populate_stage3_metadata(workspace)
        self._populate_synth_files(workspace)

        config = load_config(str(workspace["config"]))
        gen = FinetuningDatasetGenerator(config=config, logger=logging.getLogger("t"))
        gen.create_dataset(
            stt_and_vad_dir=str(workspace["stt_vad"]),
            synthesized_dir=str(workspace["synth"]),
            output_dir=str(workspace["dataset"]),
        )

        all_splits = list(workspace["dataset"].iterdir())
        jsonl_count = sum(
            1 for p in workspace["dataset"].rglob("metadata.jsonl")
        )
        assert jsonl_count > 0, "No metadata.jsonl files were created"

    def test_jsonl_entries_have_required_fields(self, workspace):
        from src._6_generate_finetuning_dataset import FinetuningDatasetGenerator, _read_jsonl
        import logging
        from src.utils.config import load_config

        self._populate_stage3_metadata(workspace)
        self._populate_synth_files(workspace)

        config = load_config(str(workspace["config"]))
        gen = FinetuningDatasetGenerator(config=config, logger=logging.getLogger("t"))
        gen.create_dataset(
            stt_and_vad_dir=str(workspace["stt_vad"]),
            synthesized_dir=str(workspace["synth"]),
            output_dir=str(workspace["dataset"]),
        )

        required = {"audio_id", "audio_path", "transcript", "split", "sample_type"}
        for jsonl in workspace["dataset"].rglob("metadata.jsonl"):
            for entry in _read_jsonl(jsonl):
                for key in required:
                    assert key in entry, f"Missing field '{key}' in {jsonl}"


# ---------------------------------------------------------------------------
# Pipeline Checkpoint
# ---------------------------------------------------------------------------

class TestPipelineCheckpoint:
    def test_save_and_load_roundtrip(self, workspace):
        from src.data_generation_pipeline import DataGenerationPipeline

        pipeline = DataGenerationPipeline(str(workspace["config"]))
        pipeline._save_checkpoint(
            stage=2,
            status="completed",
            details={"passed": 10, "total": 10},
        )

        checkpoint = pipeline._load_checkpoint()
        assert checkpoint["last_completed_stage"] == 2
        assert checkpoint["stages"]["2"]["status"] == "completed"
        assert checkpoint["stages"]["2"]["passed"] == 10

    def test_failed_stage_not_updated_as_last_completed(self, workspace):
        from src.data_generation_pipeline import DataGenerationPipeline

        pipeline = DataGenerationPipeline(str(workspace["config"]))
        pipeline._save_checkpoint(2, "completed", {})
        pipeline._save_checkpoint(3, "failed", {"error": "oops"})

        checkpoint = pipeline._load_checkpoint()
        # last_completed_stage should still be 2, not 3
        assert checkpoint["last_completed_stage"] == 2

    def test_resume_raises_without_checkpoint(self, workspace):
        from src.data_generation_pipeline import DataGenerationPipeline

        pipeline = DataGenerationPipeline(str(workspace["config"]))
        with pytest.raises(FileNotFoundError):
            pipeline.resume_from_checkpoint()

    def test_start_stage_validation(self, workspace):
        from src.data_generation_pipeline import DataGenerationPipeline

        pipeline = DataGenerationPipeline(str(workspace["config"]))
        with pytest.raises(ValueError):
            pipeline.run(start_stage=0)
        with pytest.raises(ValueError):
            pipeline.run(start_stage=7)
