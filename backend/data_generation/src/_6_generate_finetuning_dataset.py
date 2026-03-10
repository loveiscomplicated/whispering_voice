"""Fine-tuning dataset generator — Stage 6.

Collects all validated audio samples (clean ASMR + noise-synthesized variants),
applies an 8:1:1 train/val/test split grouped by source audio ID, copies audio
files into the final dataset directory, and writes per-split ``metadata.jsonl``
files and a top-level ``manifest.json``.

Typical usage::

    python src/6_generate_finetuning_dataset.py --config config/generation.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


# ---------------------------------------------------------------------------
# Sample collection helpers
# ---------------------------------------------------------------------------


def _collect_clean_samples(stt_and_vad_dir: str) -> list[dict[str, Any]]:
    """Collect clean ASMR samples from Stage 3 output.

    Each WAV file in *stt_and_vad_dir* is paired with its ``*_metadata.json``
    sidecar to extract transcript and audio characteristics.

    Args:
        stt_and_vad_dir: Root directory produced by Stage 3.

    Returns:
        List of sample dictionaries with at minimum ``audio_id``,
        ``audio_path``, ``transcript``, ``sample_type``.
    """
    root = Path(stt_and_vad_dir)
    metadata_dir = root / "metadata"

    samples: list[dict[str, Any]] = []

    # Discover WAV files at the root level (Stage 3 doesn't copy audio, so
    # we look for the original validated audio alongside metadata JSONs)
    meta_files = (
        sorted(metadata_dir.glob("*_metadata.json")) if metadata_dir.is_dir() else []
    )

    for meta_path in meta_files:
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        audio_id: str = meta.get("audio_id", meta_path.stem.replace("_metadata", ""))
        stt = meta.get("stt_result", {})
        ac = meta.get("audio_characteristics", {})

        # Attempt to locate the actual WAV — check common locations
        candidate_paths = [
            root / f"{audio_id}.wav",
            root / "audio" / f"{audio_id}.wav",
            Path(ac.get("format", "")),  # may be empty / non-path
        ]
        wav_path: Path | None = next((p for p in candidate_paths if p.is_file()), None)

        samples.append(
            {
                "audio_id": audio_id,
                "audio_path": str(wav_path) if wav_path else None,
                "transcript": stt.get("transcript", ""),
                "confidence": stt.get("confidence_score"),
                "language": stt.get("language", "ko"),
                "sample_type": "clean",
                "noise_type": None,
                "target_snr_db": None,
                "duration_ms": ac.get("duration_ms"),
                "rms_energy_db": ac.get("rms_energy_db"),
                "source_metadata": str(meta_path),
            }
        )

    return samples


def _collect_synthesized_samples(synthesized_dir: str) -> list[dict[str, Any]]:
    """Collect noise-synthesized samples from Stage 4 output.

    Each WAV file is paired with its JSON sidecar written by Stage 4.

    Args:
        synthesized_dir: Root directory produced by Stage 4.

    Returns:
        List of sample dictionaries.
    """
    root = Path(synthesized_dir)
    samples: list[dict[str, Any]] = []

    wav_files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    for wav_path in wav_files:
        sidecar = wav_path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if sidecar.exists():
            try:
                with open(sidecar, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        audio_id: str = meta.get("audio_id", wav_path.stem)
        ac = meta.get("audio_characteristics", {})

        samples.append(
            {
                "audio_id": audio_id,
                "audio_path": str(wav_path),
                "transcript": "",  # filled in later if clean meta available
                "confidence": None,
                "language": "ko",
                "sample_type": "synthesized",
                "noise_type": meta.get("noise_type"),
                "target_snr_db": meta.get("target_snr_db"),
                "duration_ms": ac.get("duration_ms"),
                "rms_energy_db": ac.get("rms_energy_db"),
                "source_metadata": str(sidecar) if sidecar.exists() else None,
            }
        )

    return samples


def _enrich_transcripts(
    synthesized: list[dict[str, Any]],
    clean_map: dict[str, str],
) -> None:
    """Fill missing transcripts in synthesized samples from the clean map.

    Args:
        synthesized: List of synthesized sample dicts (mutated in-place).
        clean_map: Mapping from ``audio_id`` to transcript string.
    """
    for sample in synthesized:
        if not sample["transcript"]:
            sample["transcript"] = clean_map.get(sample["audio_id"], "")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class FinetuningDatasetGenerator:
    """Build the final fine-tuning dataset from Stage 3 and Stage 4 outputs.

    Args:
        config: Pipeline configuration dictionary (from ``load_config``).
        logger: Logger instance.
    """

    def __init__(self, config: dict[str, Any], logger: Any) -> None:
        self._config = config
        self._logger = logger
        self._seed: int = config.get("reproducibility", {}).get("seed", 42)
        self._pipeline_version: str = config.get("reproducibility", {}).get(
            "version", "1.0"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        stt_and_vad_dir: str,
        synthesized_dir: str,
        output_dir: str,
    ) -> dict[str, Any]:
        """Assemble and split the full dataset.

        Steps:

        1. Collect clean and synthesized samples.
        2. Enrich synthesized samples with transcripts from clean metadata.
        3. Group samples by source ``audio_id``.
        4. Split groups 8:1:1 (train/val/test).
        5. Copy audio files into ``output_dir/<split>/audio/``.
        6. Write ``metadata.jsonl`` per split.
        7. Write ``manifest.json``.

        Args:
            stt_and_vad_dir: Stage 3 output directory.
            synthesized_dir: Stage 4 output directory.
            output_dir: Destination root (``dataset/``).

        Returns:
            Manifest dictionary (also written to ``manifest.json``).
        """
        random.seed(self._seed)
        out_root = Path(output_dir)

        self._logger.info("Collecting clean samples …")
        clean_samples = _collect_clean_samples(stt_and_vad_dir)
        self._logger.info(f"  Clean samples: {len(clean_samples)}")

        self._logger.info("Collecting synthesized samples …")
        synth_samples = _collect_synthesized_samples(synthesized_dir)
        self._logger.info(f"  Synthesized samples: {len(synth_samples)}")

        # Build transcript lookup from clean samples
        clean_map: dict[str, str] = {
            s["audio_id"]: s["transcript"] for s in clean_samples if s["transcript"]
        }
        _enrich_transcripts(synth_samples, clean_map)

        all_samples = [
            s
            for s in clean_samples + synth_samples
            if s.get("audio_path") and Path(s["audio_path"]).exists()
        ]

        if not all_samples:
            self._logger.error("No valid samples found. Aborting dataset generation.")
            return {}

        self._logger.info(f"Total valid samples with audio: {len(all_samples)}")

        split_map = self._split_by_audio_id(all_samples)
        manifest = self._write_splits(split_map, out_root)

        manifest_path = out_root / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        self._logger.info(f"Manifest written: {manifest_path}")

        return manifest

    def generate_manifest(self, dataset_dir: str) -> dict[str, Any]:
        """Build a manifest from an existing dataset directory.

        Useful for re-indexing without regenerating the dataset.

        Args:
            dataset_dir: Root directory of an existing dataset.

        Returns:
            Manifest dictionary.
        """
        root = Path(dataset_dir)
        manifest: dict[str, Any] = {"splits": {}}

        for split in ("train", "val", "test"):
            jsonl_path = root / split / "metadata.jsonl"
            if not jsonl_path.exists():
                continue
            entries = _read_jsonl(jsonl_path)
            manifest["splits"][split] = {
                "count": len(entries),
                "metadata_jsonl": str(jsonl_path),
            }

        manifest["total"] = sum(v["count"] for v in manifest["splits"].values())
        return manifest

    def generate_statistics(self, dataset_dir: str) -> dict[str, Any]:
        """Compute descriptive statistics over the assembled dataset.

        Args:
            dataset_dir: Root directory of an existing dataset.

        Returns:
            Statistics dictionary with counts, duration, noise-type and SNR
            breakdowns per split.
        """
        root = Path(dataset_dir)
        stats: dict[str, Any] = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "splits": {},
        }

        all_entries: list[dict[str, Any]] = []
        for split in ("train", "val", "test"):
            jsonl_path = root / split / "metadata.jsonl"
            if not jsonl_path.exists():
                continue
            entries = _read_jsonl(jsonl_path)
            all_entries.extend(entries)

            durations = [e["duration_ms"] for e in entries if e.get("duration_ms")]
            snr_by_type: dict[str, list[float]] = defaultdict(list)
            for e in entries:
                if e.get("noise_type") and e.get("target_snr_db") is not None:
                    snr_by_type[e["noise_type"]].append(e["target_snr_db"])

            stats["splits"][split] = {
                "total_samples": len(entries),
                "clean_samples": sum(
                    1 for e in entries if e.get("sample_type") == "clean"
                ),
                "synthesized_samples": sum(
                    1 for e in entries if e.get("sample_type") == "synthesized"
                ),
                "total_duration_s": (
                    round(sum(durations) / 1_000, 2) if durations else 0
                ),
                "mean_duration_ms": (
                    round(sum(durations) / len(durations), 2) if durations else 0
                ),
                "noise_type_counts": {
                    nt: len(files) for nt, files in snr_by_type.items()
                },
            }

        # Overall
        all_durations = [e["duration_ms"] for e in all_entries if e.get("duration_ms")]
        stats["total_samples"] = len(all_entries)
        stats["total_duration_s"] = (
            round(sum(all_durations) / 1_000, 2) if all_durations else 0
        )

        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_by_audio_id(
        self, samples: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group samples by audio_id and assign each group to a split.

        Grouping by source ID ensures that all SNR variants of the same
        recording end up in the same split (no data leakage).

        Args:
            samples: All valid sample dictionaries.

        Returns:
            Dictionary mapping split names to lists of samples.
        """
        # Group by audio_id
        id_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in samples:
            id_groups[s["audio_id"]].append(s)

        audio_ids = list(id_groups.keys())
        random.shuffle(audio_ids)

        n = len(audio_ids)
        n_train = max(1, int(n * _SPLIT_RATIOS["train"]))
        n_val = max(1, int(n * _SPLIT_RATIOS["val"]))

        train_ids = set(audio_ids[:n_train])
        val_ids = set(audio_ids[n_train : n_train + n_val])
        # Remaining IDs go to test
        test_ids = set(audio_ids[n_train + n_val :])

        split_map: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        for aid, group in id_groups.items():
            if aid in train_ids:
                split_map["train"].extend(group)
            elif aid in val_ids:
                split_map["val"].extend(group)
            else:
                split_map["test"].extend(group)

        for split, items in split_map.items():
            self._logger.info(f"  {split}: {len(items)} samples")

        return split_map

    def _write_splits(
        self,
        split_map: dict[str, list[dict[str, Any]]],
        out_root: Path,
    ) -> dict[str, Any]:
        """Copy audio files and write metadata.jsonl for each split.

        Args:
            split_map: Output of :meth:`_split_by_audio_id`.
            out_root: Root output directory.

        Returns:
            Manifest dictionary.
        """
        manifest: dict[str, Any] = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "pipeline_version": self._pipeline_version,
            "seed": self._seed,
            "splits": {},
            "total": 0,
        }

        for split, samples in split_map.items():
            audio_out = out_root / split / "audio"
            audio_out.mkdir(parents=True, exist_ok=True)
            jsonl_path = out_root / split / "metadata.jsonl"

            written = 0
            with open(jsonl_path, "w", encoding="utf-8") as jf:
                for sample in tqdm(samples, desc=f"Writing {split}", unit="file"):
                    src = Path(sample["audio_path"])
                    dest = audio_out / src.name

                    # Copy audio (skip if already there)
                    if not dest.exists():
                        try:
                            shutil.copy2(src, dest)
                        except Exception as exc:
                            self._logger.error(f"Failed to copy {src.name}: {exc}")
                            continue

                    record = {
                        "audio_id": sample["audio_id"],
                        "audio_path": str(dest.relative_to(out_root)),
                        "transcript": sample["transcript"],
                        "confidence": sample.get("confidence"),
                        "language": sample.get("language", "ko"),
                        "sample_type": sample.get("sample_type"),
                        "noise_type": sample.get("noise_type"),
                        "target_snr_db": sample.get("target_snr_db"),
                        "duration_ms": sample.get("duration_ms"),
                        "rms_energy_db": sample.get("rms_energy_db"),
                        "split": split,
                        "source_metadata": sample.get("source_metadata"),
                    }
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

            manifest["splits"][split] = {
                "count": written,
                "audio_dir": str((out_root / split / "audio").relative_to(out_root)),
                "metadata_jsonl": str(jsonl_path.relative_to(out_root)),
            }
            self._logger.info(f"  {split}: {written} samples written → {jsonl_path}")

        manifest["total"] = sum(v["count"] for v in manifest["splits"].values())
        return manifest


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-Lines file into a list of dictionaries.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of parsed JSON objects; malformed lines are silently skipped.
    """
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning dataset from pipeline outputs (Stage 6).",
    )
    parser.add_argument(
        "--config", required=True, help="Path to generation.yaml config file."
    )
    parser.add_argument(
        "--stt-vad-dir",
        default=None,
        help="Stage 3 output directory (default: stt_and_vad/ from config).",
    )
    parser.add_argument(
        "--synthesized-dir",
        default=None,
        help="Stage 4 output directory (default: synthesized/ from config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dataset output directory (default: dataset/ from config).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print statistics for an existing dataset without regenerating.",
    )
    parser.add_argument("--log-file", default=None, help="Path to log file (optional).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for Stage 6.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    args = _parse_args(argv)
    config = load_config(args.config)

    logs_dir: str = config.get("output_dirs", {}).get("logs", "./logs")
    log_file = args.log_file or str(
        Path(logs_dir) / "6_generate_finetuning_dataset.log"
    )
    log = setup_logger("finetuning_dataset", log_file=log_file)

    out_dirs = config.get("output_dirs", {})
    stt_vad_dir = args.stt_vad_dir or out_dirs.get("stt_and_vad", "./stt_and_vad")
    synthesized_dir = args.synthesized_dir or out_dirs.get(
        "synthesized", "./synthesized"
    )
    output_dir = args.output_dir or out_dirs.get("dataset", "./dataset")

    generator = FinetuningDatasetGenerator(config=config, logger=log)

    if args.stats_only:
        stats = generator.generate_statistics(output_dir)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    manifest = generator.create_dataset(
        stt_and_vad_dir=stt_vad_dir,
        synthesized_dir=synthesized_dir,
        output_dir=output_dir,
    )

    if not manifest:
        sys.exit(1)

    stats = generator.generate_statistics(output_dir)
    stats_path = Path(output_dir) / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nDataset written to : {output_dir}")
    print(f"Total samples      : {manifest.get('total', 0)}")
    for split, info in manifest.get("splits", {}).items():
        print(f"  {split:6s}: {info['count']} samples")
    print(f"Statistics         : {stats_path}")


if __name__ == "__main__":
    main()
