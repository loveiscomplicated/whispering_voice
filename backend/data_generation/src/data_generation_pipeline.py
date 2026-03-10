"""Full data-generation pipeline orchestrator.

Connects all six stages into a single, resumable pipeline.  After each stage
completes, a JSON checkpoint is written so that the pipeline can be restarted
from any point without re-doing earlier work.

Typical usage::

    # Run from the beginning
    python src/data_generation_pipeline.py --config config/generation.yaml

    # Skip completed stages and start at stage 3
    python src/data_generation_pipeline.py --config config/generation.yaml --start-stage 3

    # Auto-resume from the last saved checkpoint
    python src/data_generation_pipeline.py --config config/generation.yaml --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# NOTE: is this necessary?
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.audio_processor import set_seed  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

# Stage imports — done lazily inside each runner to avoid importing heavy
# ML libraries (whisper, pyannote) until they are actually needed.

_CHECKPOINT_FILENAME = "pipeline_checkpoint.json"
_TOTAL_STAGES = 6


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DataGenerationPipeline:
    """Orchestrate the six-stage STT data-generation pipeline.

    Args:
        config_path: Path to ``generation.yaml``.
    """

    def __init__(self, config_path: str) -> None:
        self._config_path = config_path
        self._config = load_config(config_path)

        logs_dir = self._config.get("output_dirs", {}).get(
            "logs",
            "./backend/data_generation/logs",  # Assuming the script is run from root.
        )
        self._log = setup_logger(
            "pipeline",
            log_file=str(Path(logs_dir) / "pipeline.log"),
        )

        # Reproducibility
        seed = self._config.get("reproducibility", {}).get("seed", 42)
        set_seed(seed)
        self._log.info(f"Random seed set to {seed}")

        out_dirs = self._config.get("output_dirs", {})
        self._raw_dir: str = out_dirs.get(
            "raw_downloads",
            "./backend/data_generation/raw_downloads",  # Assuming the script is run from root.
        )
        self._stt_vad_dir: str = out_dirs.get(
            "stt_and_vad",
            "./backend/data_generation/stt_and_vad",  # Assuming the script is run from root.
        )
        self._synth_dir: str = out_dirs.get(
            "synthesized",
            "./backend/data_generation/synthesized",  # Assuming the script is run from root.
        )
        self._dataset_dir: str = out_dirs.get(
            "dataset",
            "./backend/data_generation/dataset",  # Assuming the script is run from root.
        )
        self._logs_dir: str = logs_dir

        self._checkpoint_path = Path(self._logs_dir) / _CHECKPOINT_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, start_stage: int = 1) -> bool:
        """Execute pipeline stages from *start_stage* to 6.

        Args:
            start_stage: First stage to execute (1-6).

        Returns:
            ``True`` if every stage from *start_stage* onward succeeded,
            ``False`` if any stage raised an exception.
        """
        if not 1 <= start_stage <= _TOTAL_STAGES:
            raise ValueError(
                f"start_stage must be between 1 and {_TOTAL_STAGES}, "
                f"got {start_stage}"
            )

        stages = self._build_stage_definitions()
        overall_ok = True

        for stage_def in stages:
            n = stage_def["number"]
            if n < start_stage:
                self._log.info(f"Skipping stage {n} ({stage_def['name']})")
                continue

            self._log.info(f"{'='*60}")
            self._log.info(f"Starting stage {n}: {stage_def['name']}")
            self._log.info(f"{'='*60}")

            t_start = datetime.now(tz=timezone.utc)
            try:
                details = stage_def["runner"]()
                status = "completed"
                self._log.info(f"Stage {n} completed successfully.")
            except Exception as exc:
                status = "failed"
                details = {"error": str(exc)}
                self._log.exception(f"Stage {n} FAILED: {exc}")
                overall_ok = False

            self._save_checkpoint(
                stage=n,
                status=status,
                details={
                    **details,
                    "started_at": t_start.isoformat(),
                    "finished_at": datetime.now(tz=timezone.utc).isoformat(),
                },
            )

            if status == "failed":
                self._log.error(
                    f"Pipeline halted at stage {n}. "
                    "Fix the error and re-run with --start-stage "
                    f"{n} or --resume."
                )
                return False

        self._log.info("Pipeline finished successfully.")
        return overall_ok

    def resume_from_checkpoint(self) -> bool:
        """Resume the pipeline from the stage after the last checkpoint.

        Returns:
            Result of :meth:`run` starting at the appropriate stage.

        Raises:
            FileNotFoundError: If no checkpoint file exists.
        """
        checkpoint = self._load_checkpoint()
        if not checkpoint:
            raise FileNotFoundError(
                f"No checkpoint found at {self._checkpoint_path}. "
                "Run without --resume to start from the beginning."
            )

        last_completed = checkpoint.get("last_completed_stage", 0)
        resume_from = last_completed + 1

        if resume_from > _TOTAL_STAGES:
            self._log.info("All stages already completed. Nothing to resume.")
            return True

        self._log.info(
            f"Resuming from stage {resume_from} "
            f"(last completed: stage {last_completed})"
        )
        return self.run(start_stage=resume_from)

    def _save_checkpoint(
        self,
        stage: int,
        status: str,
        details: dict[str, Any],
    ) -> None:
        """Persist pipeline progress to a JSON checkpoint file.

        Args:
            stage: Stage number just executed.
            status: ``"completed"`` or ``"failed"``.
            details: Arbitrary stage-output metadata.
        """
        checkpoint = self._load_checkpoint() or {"stages": {}}

        checkpoint["last_completed_stage"] = (
            stage
            if status == "completed"
            else (checkpoint.get("last_completed_stage", 0))
        )
        checkpoint["stages"][str(stage)] = {
            "status": status,
            **details,
        }
        checkpoint["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        self._log.debug(f"Checkpoint saved: stage={stage} status={status}")

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load the JSON checkpoint file.

        Returns:
            Parsed checkpoint dictionary, or an empty dict if absent.
        """
        if not self._checkpoint_path.exists():
            return {}
        try:
            with open(self._checkpoint_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self._log.warning(f"Could not read checkpoint: {exc}")
            return {}

    # ------------------------------------------------------------------
    # Stage definitions
    # ------------------------------------------------------------------

    def _build_stage_definitions(self) -> list[dict[str, Any]]:
        """Return an ordered list of stage descriptors.

        Each descriptor has:
        - ``number`` (int): Stage index.
        - ``name`` (str): Human-readable label.
        - ``runner`` (callable): Zero-arg function that executes the stage
          and returns a details dict.

        Returns:
            Ordered list of stage descriptors (1 through 6).
        """
        return [
            {"number": 1, "name": "YouTube Download", "runner": self._run_stage_1},
            {"number": 2, "name": "Quality Validation 1", "runner": self._run_stage_2},
            {"number": 3, "name": "STT + VAD Processing", "runner": self._run_stage_3},
            {"number": 4, "name": "Noise Synthesis", "runner": self._run_stage_4},
            {"number": 5, "name": "Quality Validation 2", "runner": self._run_stage_5},
            {"number": 6, "name": "Dataset Generation", "runner": self._run_stage_6},
        ]

    # Stage runners ---------------------------------------------------

    def _run_stage_1(self) -> dict[str, Any]:
        from backend.data_generation.src._1_download_youtube import (
            YouTubeDownloader,
        )  # noqa: PLC0415

        downloader = YouTubeDownloader(
            output_dir=self._raw_dir,
            config=self._config,
            logger=self._log,
        )
        playlist_ids: list[str] = self._config.get("youtube", {}).get(
            "playlist_ids", []
        )
        all_paths: list[str] = []
        for pid in playlist_ids:
            paths = downloader.download_playlist(pid)
            all_paths.extend(paths)

        return {"downloaded_files": len(all_paths), "output_dir": self._raw_dir}

    def _run_stage_2(self) -> dict[str, Any]:
        from backend.data_generation.src._2_quality_validation_1 import (
            QualityValidator1,
        )  # noqa: PLC0415

        validator = QualityValidator1(config=self._config, logger=self._log)
        df = validator.validate_batch(self._raw_dir)

        report_path = str(Path(self._logs_dir) / "quality_validation_1_report.csv")
        validator.generate_report(report_path)

        n_pass = int(df["passed"].sum()) if not df.empty else 0
        return {
            "total": len(df),
            "passed": n_pass,
            "report": report_path,
        }

    def _run_stage_3(self) -> dict[str, Any]:
        from backend.data_generation.src._3_run_stt_and_vad import (
            STTAndVADProcessor,
        )  # noqa: PLC0415

        processor = STTAndVADProcessor(config=self._config, logger=self._log)

        # Prefer the list of files that passed Stage 2
        passed_json = str(
            Path(self._logs_dir) / "quality_validation_1_report_passed.json"
        )
        metadata_dir = Path(self._stt_vad_dir) / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        if Path(passed_json).exists():
            with open(passed_json, encoding="utf-8") as f:
                data = json.load(f)
            audio_files = [Path(p) for p in data.get("files", [])]
        else:
            audio_files = sorted(
                p for p in Path(self._raw_dir).iterdir() if p.suffix.lower() == ".wav"
            )

        results: list[dict[str, Any]] = []
        for af in audio_files:
            try:
                meta = processor.process_audio(str(af))
                out = metadata_dir / f"{af.stem}_metadata.json"
                processor.save_metadata(meta, str(out))
                results.append(meta)
            except Exception as exc:
                self._log.error(f"Stage 3 failed for {af.name}: {exc}")

        return {
            "processed": len(results),
            "metadata_dir": str(metadata_dir),
        }

    def _run_stage_4(self) -> dict[str, Any]:
        from backend.data_generation.src._4_synthesize_noise import (
            NoiseSynthesizer,
        )  # noqa: PLC0415

        synthesizer = NoiseSynthesizer(config=self._config, logger=self._log)
        noise_dir = str(Path(self._raw_dir) / "noise")
        results = synthesizer.synthesize_batch(
            asmr_dir=self._stt_vad_dir,
            noise_dir=noise_dir,
            output_dir=self._synth_dir,
        )
        total = sum(len(v) for v in results.values())
        return {"synthesized_files": total, "output_dir": self._synth_dir}

    def _run_stage_5(self) -> dict[str, Any]:
        from backend.data_generation.src._5_quality_validation_2 import (
            QualityValidator2,
        )  # noqa: PLC0415

        validator = QualityValidator2(config=self._config, logger=self._log)
        df = validator.validate_batch(self._synth_dir)

        report_path = str(Path(self._logs_dir) / "quality_validation_2_report.csv")
        if not df.empty:
            validator.generate_report(report_path)

        n_pass = int(df["passed"].sum()) if not df.empty else 0
        return {
            "total": len(df),
            "passed": n_pass,
            "report": report_path,
        }

    def _run_stage_6(self) -> dict[str, Any]:
        from backend.data_generation.src._6_generate_finetuning_dataset import (
            FinetuningDatasetGenerator,
        )  # noqa: PLC0415

        generator = FinetuningDatasetGenerator(config=self._config, logger=self._log)
        manifest = generator.create_dataset(
            stt_and_vad_dir=self._stt_vad_dir,
            synthesized_dir=self._synth_dir,
            output_dir=self._dataset_dir,
        )
        return {
            "total_samples": manifest.get("total", 0),
            "dataset_dir": self._dataset_dir,
        }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full STT data-generation pipeline.",
    )
    parser.add_argument(
        "--config", required=True, help="Path to generation.yaml config file."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--start-stage",
        type=int,
        default=1,
        metavar="N",
        help="Stage to start from (1–6). Default: 1.",
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from the last saved checkpoint.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the orchestrator.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    args = _parse_args(argv)
    pipeline = DataGenerationPipeline(config_path=args.config)

    if args.resume:
        ok = pipeline.resume_from_checkpoint()
    else:
        ok = pipeline.run(start_stage=args.start_stage)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
