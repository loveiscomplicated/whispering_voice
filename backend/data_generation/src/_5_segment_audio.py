"""Audio segmentation — Stage 6.

Reads per-file STT+VAD metadata produced by Stage 5 and physically slices
each source audio file into individual speech segments based on the Pyannote
VAD output.  Each segment is saved as a 16 kHz mono WAV together with a
per-segment metadata JSON whose schema is compatible with the downstream
noise-synthesis and dataset-generation stages.

Text alignment uses Whisper's segment timestamps stored as
``stt_result.whisper_segments`` by Stage 5.  When that field is absent (e.g.
for metadata written by an older pipeline run) the full transcript is
assigned to every segment as a fallback.

Typical usage::

    python src/_5_segment_audio.py \\
        --stt-vad-dir backend/data_generation/stt_and_vad \\
        --output-dir  backend/data_generation/segments \\
        --config      config/generation.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.audio_processor import get_audio_info, load_audio, save_audio  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

_PIPELINE_VERSION = "1.0"
_SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Text alignment helper
# ---------------------------------------------------------------------------


def _build_exclusive_alignment(
    vad_segments: list[dict[str, Any]],
    whisper_segments: list[dict[str, Any]],
) -> dict[int, list[str]]:
    """Assign each Whisper segment exclusively to the best-matching VAD segment.

    For each Whisper segment the VAD segment with the largest time overlap
    wins the text.  This prevents the same sentence from being duplicated
    across multiple short VAD segments that all fall inside one long Whisper
    segment.

    Args:
        vad_segments: Ordered list of VAD segment dicts (must have
            ``start_ms`` and ``end_ms``).
        whisper_segments: List of ``{"start_ms", "end_ms", "text"}`` dicts.

    Returns:
        Mapping from VAD segment list-index → list of assigned text strings.
    """
    assignments: dict[int, list[str]] = {i: [] for i in range(len(vad_segments))}

    for ws in whisper_segments:
        ws_start = ws.get("start_ms", 0.0)
        ws_end = ws.get("end_ms", 0.0)
        text = ws.get("text", "").strip()
        if not text:
            continue

        best_idx = -1
        best_overlap = 0.0

        for i, vad in enumerate(vad_segments):
            overlap = max(
                0.0,
                min(ws_end, vad["end_ms"]) - max(ws_start, vad["start_ms"]),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        if best_idx >= 0:
            assignments[best_idx].append(text)

    return assignments


# ---------------------------------------------------------------------------
# Segmentor
# ---------------------------------------------------------------------------


class AudioSegmentor:
    """Slice audio files into per-VAD-segment WAV clips.

    Configuration is read from ``config["segmentation"]``.  Sensible
    defaults are provided for every parameter.

    Args:
        config: Pipeline configuration dictionary (from ``load_config``).
        logger: Logger instance for diagnostic messages.
    """

    def __init__(self, config: dict[str, Any], logger: Any) -> None:
        self._config = config
        self._logger = logger

        seg_cfg = config.get("segmentation", {})
        self._min_duration_ms: float = seg_cfg.get("min_segment_duration_ms", 500.0)
        self._max_duration_ms: float = seg_cfg.get("max_segment_duration_ms", 30_000.0)
        self._padding_ms: float = seg_cfg.get("padding_ms", 50.0)
        self._skip_low_confidence: bool = seg_cfg.get("skip_low_confidence", False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(
        self,
        audio_path: str,
        metadata_path: str,
        output_dir: str,
    ) -> list[dict[str, Any]]:
        """Slice one audio file into VAD-based segments.

        Args:
            audio_path: Path to the source WAV file (from Stage 5 output).
            metadata_path: Path to the ``*_metadata.json`` produced by Stage 5.
            output_dir: Root output directory (``segments/``).

        Returns:
            List of per-segment metadata dictionaries.  Empty when no VAD
            segments pass the duration filter.

        Raises:
            RuntimeError: If the audio file cannot be loaded.
        """
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)

        if self._skip_low_confidence and meta.get("low_confidence", False):
            self._logger.info(
                f"Skipping low-confidence file: {Path(audio_path).stem}"
            )
            return []

        audio_id: str = meta["audio_id"]
        stt = meta.get("stt_result", {})
        full_transcript: str = stt.get("transcript", "")
        whisper_segments: list[dict[str, Any]] = stt.get("whisper_segments", [])

        vad_segments: list[dict[str, Any]] = (
            meta.get("vad_result", {}).get("segments", [])
        )
        if not vad_segments:
            self._logger.warning(f"No VAD segments in metadata for {audio_id}")
            return []

        try:
            audio, sr = load_audio(audio_path, sr=_SAMPLE_RATE)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load audio '{audio_path}': {exc}"
            ) from exc

        total_samples = len(audio)
        audio_out = Path(output_dir) / "audio"
        meta_out_dir = Path(output_dir) / "metadata"
        audio_out.mkdir(parents=True, exist_ok=True)
        meta_out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute exclusive alignment over all VAD segments so that each
        # Whisper segment's text is assigned to exactly one VAD segment (the
        # one with the greatest time overlap).  This prevents the same
        # sentence appearing in multiple short consecutive segments.
        text_assignments = _build_exclusive_alignment(vad_segments, whisper_segments)

        results: list[dict[str, Any]] = []
        seg_counter = 0  # index among segments that actually pass filters

        for vad_idx, vad_seg in enumerate(vad_segments):
            start_ms: float = vad_seg["start_ms"]
            end_ms: float = vad_seg["end_ms"]
            duration_ms: float = end_ms - start_ms

            if duration_ms < self._min_duration_ms:
                self._logger.debug(
                    f"  Skip {audio_id} seg {vad_seg['segment_id']}: "
                    f"{duration_ms:.0f}ms < min {self._min_duration_ms:.0f}ms"
                )
                continue

            if duration_ms > self._max_duration_ms:
                self._logger.debug(
                    f"  Skip {audio_id} seg {vad_seg['segment_id']}: "
                    f"{duration_ms:.0f}ms > max {self._max_duration_ms:.0f}ms"
                )
                continue

            # Apply symmetric padding while staying within bounds
            padded_start_ms = max(0.0, start_ms - self._padding_ms)
            padded_end_ms = min(
                total_samples / sr * 1_000, end_ms + self._padding_ms
            )

            start_sample = int(padded_start_ms / 1_000 * sr)
            end_sample = int(padded_end_ms / 1_000 * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) == 0:
                continue

            # Use exclusively assigned text; fall back to full transcript only
            # when no Whisper segment had its best overlap here.
            assigned = text_assignments.get(vad_idx, [])
            text = " ".join(assigned).strip() if assigned else full_transcript

            seg_id = f"{audio_id}_seg{seg_counter:03d}"
            seg_wav_path = audio_out / f"{seg_id}.wav"

            save_audio(segment_audio, str(seg_wav_path), sr=sr)

            info = get_audio_info(segment_audio, sr)
            seg_meta: dict[str, Any] = {
                "audio_id": seg_id,
                "source_audio_id": audio_id,
                "segment_index": seg_counter,
                "processing_pipeline_version": _PIPELINE_VERSION,
                "stt_result": {
                    "transcript": text,
                    "language": stt.get("language", "ko"),
                    "confidence_score": stt.get("confidence_score"),
                    "model_used": stt.get("model_used"),
                    "model_version": stt.get("model_version"),
                },
                "vad_segment": {
                    "segment_id": vad_seg["segment_id"],
                    "start_ms": vad_seg["start_ms"],
                    "end_ms": vad_seg["end_ms"],
                    "duration_ms": vad_seg["duration_ms"],
                    "confidence": vad_seg["confidence"],
                },
                "audio_characteristics": {
                    "format": "wav",
                    "sample_rate": sr,
                    "channels": 1,
                    "duration_ms": round(info["duration_ms"], 2),
                    "rms_energy_db": round(info["rms_energy_db"], 2),
                    "peak_amplitude": round(info["peak_amplitude"], 4),
                },
            }

            seg_meta_path = meta_out_dir / f"{seg_id}_metadata.json"
            with open(seg_meta_path, "w", encoding="utf-8") as f:
                json.dump(seg_meta, f, ensure_ascii=False, indent=2)

            self._logger.info(
                f"  {seg_id}: {info['duration_ms']:.0f}ms  "
                f"text={repr(text[:50])}"
            )
            results.append(seg_meta)
            seg_counter += 1

        return results

    def process_batch(
        self,
        stt_vad_dir: str,
        output_dir: str,
    ) -> list[dict[str, Any]]:
        """Process all metadata files in a Stage 5 output directory.

        For each ``*_metadata.json`` the corresponding source WAV is located
        in *stt_vad_dir* (the same directory where Stage 5 copies audio).

        Args:
            stt_vad_dir: Root directory produced by Stage 5 (contains WAV
                files and a ``metadata/`` sub-directory).
            output_dir: Root output directory for segments.

        Returns:
            Flat list of all per-segment metadata dictionaries produced.
        """
        # Always create output directories so downstream stages (noise
        # synthesis, dataset generation) can safely reference them even
        # when 0 segments are produced.
        audio_out = Path(output_dir) / "audio"
        meta_out_dir = Path(output_dir) / "metadata"
        audio_out.mkdir(parents=True, exist_ok=True)
        meta_out_dir.mkdir(parents=True, exist_ok=True)

        meta_dir = Path(stt_vad_dir) / "metadata"
        if not meta_dir.is_dir():
            raise NotADirectoryError(
                f"Stage 5 metadata directory not found: {meta_dir}. "
                "Ensure Stage 5 (STT + VAD) has been run and completed successfully."
            )

        meta_files = sorted(meta_dir.glob("*_metadata.json"))
        if not meta_files:
            raise RuntimeError(
                f"No metadata files found in: {meta_dir}. "
                "Stage 5 (STT + VAD) completed but wrote no metadata — "
                "check that at least one audio file passed Stage 4 validation "
                "and that STT/VAD processing did not fail for all files."
            )

        self._logger.info(
            f"Segmenting {len(meta_files)} file(s) from: {stt_vad_dir}"
        )

        all_segments: list[dict[str, Any]] = []
        for meta_path in meta_files:
            audio_id = meta_path.stem.replace("_metadata", "")
            audio_path = Path(stt_vad_dir) / f"{audio_id}.wav"

            if not audio_path.exists():
                self._logger.warning(
                    f"Source WAV not found, skipping: {audio_path}"
                )
                continue

            try:
                segs = self.process_file(
                    str(audio_path), str(meta_path), output_dir
                )
                all_segments.extend(segs)
                self._logger.info(
                    f"{audio_id}: {len(segs)} segment(s) written"
                )
            except Exception as exc:
                self._logger.error(f"Failed to segment {audio_id}: {exc}")

        return all_segments

    def save_manifest(
        self, segments: list[dict[str, Any]], output_dir: str
    ) -> str:
        """Write a ``manifest.json`` summarising all produced segments.

        Args:
            segments: List of per-segment metadata dicts returned by
                :meth:`process_batch`.
            output_dir: Root output directory (``segments/``).

        Returns:
            Absolute path to the written manifest file.
        """
        manifest = {
            "total_segments": len(segments),
            "total_speech_duration_ms": round(
                sum(
                    s.get("audio_characteristics", {}).get("duration_ms", 0.0)
                    for s in segments
                ),
                2,
            ),
            "segments": [
                {
                    "audio_id": s["audio_id"],
                    "source_audio_id": s["source_audio_id"],
                    "duration_ms": s.get("audio_characteristics", {}).get(
                        "duration_ms"
                    ),
                    "transcript": s.get("stt_result", {}).get("transcript", ""),
                    "audio_path": str(
                        Path(output_dir) / "audio" / f"{s['audio_id']}.wav"
                    ),
                    "metadata_path": str(
                        Path(output_dir)
                        / "metadata"
                        / f"{s['audio_id']}_metadata.json"
                    ),
                }
                for s in segments
            ],
        }

        manifest_path = Path(output_dir) / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self._logger.info(f"Manifest saved: {manifest_path}")
        return str(manifest_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice audio into VAD segments (Stage 6).",
    )
    parser.add_argument(
        "--config", required=True, help="Path to generation.yaml config file."
    )
    parser.add_argument(
        "--stt-vad-dir",
        default=None,
        help="Stage 5 output directory (default: stt_and_vad/ from config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for segments (default: segments/ from config).",
    )
    parser.add_argument(
        "--log-file", default=None, help="Path to log file (optional)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the segmentation stage.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    args = _parse_args(argv)
    config = load_config(args.config)

    logs_dir: str = config.get("output_dirs", {}).get("logs", "./logs")
    log_file = args.log_file or str(Path(logs_dir) / "6_segment_audio.log")
    log = setup_logger("segment_audio", log_file=log_file)

    out_dirs = config.get("output_dirs", {})
    stt_vad_dir = args.stt_vad_dir or out_dirs.get(
        "stt_and_vad", "./backend/data_generation/stt_and_vad"
    )
    output_dir = args.output_dir or out_dirs.get(
        "segments", "./backend/data_generation/segments"
    )

    segmentor = AudioSegmentor(config=config, logger=log)
    all_segments = segmentor.process_batch(stt_vad_dir, output_dir)
    manifest_path = segmentor.save_manifest(all_segments, output_dir)

    total_ms = sum(
        s.get("audio_characteristics", {}).get("duration_ms", 0.0)
        for s in all_segments
    )

    print(f"\nSegments produced : {len(all_segments)}")
    print(f"Total speech      : {total_ms / 1_000:.1f}s")
    print(f"Output directory  : {output_dir}")
    print(f"Manifest          : {manifest_path}")


if __name__ == "__main__":
    main()
