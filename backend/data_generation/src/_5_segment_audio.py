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
        self._min_duration_ms: float = seg_cfg.get("min_segment_duration_ms", 5_000.0)
        self._max_duration_ms: float = seg_cfg.get("max_segment_duration_ms", 20_000.0)
        self._merge_max_gap_ms: float = seg_cfg.get("merge_max_gap_ms", 1_000.0)
        self._padding_ms: float = seg_cfg.get("padding_ms", 50.0)
        self._skip_low_confidence: bool = seg_cfg.get("skip_low_confidence", False)
        # When Pyannote speech_ratio falls below this threshold (e.g. whispering
        # content), use subtitle/Whisper timestamps for slicing instead.
        self._use_subtitle_fallback: bool = seg_cfg.get(
            "use_subtitle_segments_fallback", True
        )
        self._vad_fallback_threshold: float = seg_cfg.get(
            "vad_fallback_threshold", 0.3
        )

    # ------------------------------------------------------------------
    # VAD segment merging
    # ------------------------------------------------------------------

    def _collapse_group(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        """Collapse a list of consecutive VAD segments into one merged segment.

        The merged segment spans from ``group[0]["start_ms"]`` to
        ``group[-1]["end_ms"]``.  Confidence is the minimum across the group
        (conservative estimate).

        Args:
            group: Non-empty list of VAD segment dicts.

        Returns:
            A single merged VAD segment dict.
        """
        return {
            "segment_id": group[0]["segment_id"],
            "start_ms": group[0]["start_ms"],
            "end_ms": group[-1]["end_ms"],
            "duration_ms": round(group[-1]["end_ms"] - group[0]["start_ms"], 2),
            "confidence": round(min(s["confidence"] for s in group), 4),
        }

    def _merge_vad_segments(
        self, vad_segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge short consecutive VAD segments into longer clips.

        Adjacent segments are merged when:

        1. The silence gap between them is at most ``merge_max_gap_ms``, **and**
        2. The resulting merged duration would not exceed ``max_segment_duration_ms``.

        When either condition is violated the current group is finalised and a
        new group begins with the next segment.

        Args:
            vad_segments: Raw VAD segments from Pyannote (ordered by start time).

        Returns:
            List of merged segment dicts ready for slicing.
        """
        if not vad_segments:
            return []

        merged: list[dict[str, Any]] = []
        group: list[dict[str, Any]] = [vad_segments[0]]

        for seg in vad_segments[1:]:
            last = group[-1]
            gap_ms = seg["start_ms"] - last["end_ms"]
            new_duration_ms = seg["end_ms"] - group[0]["start_ms"]

            if gap_ms <= self._merge_max_gap_ms and new_duration_ms <= self._max_duration_ms:
                group.append(seg)
            else:
                merged.append(self._collapse_group(group))
                group = [seg]

        merged.append(self._collapse_group(group))
        return merged

    def _merge_whisper_segments(
        self, whisper_segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge subtitle/Whisper segments into longer sliceable chunks.

        Used as a fallback when Pyannote VAD speech ratio is too low to be
        reliable (e.g. whispering/ASMR content).  Unlike VAD merging, each
        input segment already carries its transcript text, so the merged
        output includes a ``text`` field with the concatenated lines.

        Args:
            whisper_segments: List of ``{"start_ms", "end_ms", "text"}`` dicts
                from ``stt_result.whisper_segments``.

        Returns:
            List of merged chunk dicts with keys ``start_ms``, ``end_ms``,
            ``duration_ms``, and ``text``.
        """
        segs = [s for s in whisper_segments if s.get("text", "").strip()]
        if not segs:
            return []

        merged: list[dict[str, Any]] = []
        group = [segs[0]]
        group_texts: list[str] = [segs[0]["text"].strip()]

        for seg in segs[1:]:
            last = group[-1]
            gap_ms = seg["start_ms"] - last["end_ms"]
            new_duration_ms = seg["end_ms"] - group[0]["start_ms"]

            if gap_ms <= self._merge_max_gap_ms and new_duration_ms <= self._max_duration_ms:
                group.append(seg)
                text = seg.get("text", "").strip()
                if text:
                    group_texts.append(text)
            else:
                merged.append({
                    "start_ms": group[0]["start_ms"],
                    "end_ms": group[-1]["end_ms"],
                    "duration_ms": round(group[-1]["end_ms"] - group[0]["start_ms"], 2),
                    "text": " ".join(group_texts).strip(),
                })
                group = [seg]
                group_texts = [seg.get("text", "").strip()]

        merged.append({
            "start_ms": group[0]["start_ms"],
            "end_ms": group[-1]["end_ms"],
            "duration_ms": round(group[-1]["end_ms"] - group[0]["start_ms"], 2),
            "text": " ".join(group_texts).strip(),
        })
        return merged

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

        vad_result = meta.get("vad_result", {})
        raw_vad_segments: list[dict[str, Any]] = vad_result.get("segments", [])
        vad_speech_ratio: float = vad_result.get("speech_ratio", 1.0)

        # ------------------------------------------------------------------
        # Decide which source to use for slicing boundaries.
        #
        # Pyannote VAD is trained on normal-volume speech.  For ASMR /
        # whispering content its speech_ratio is often < 10% even when the
        # speaker never stops talking.  When this happens, fall back to the
        # subtitle / Whisper timestamps which were produced independently and
        # are accurate regardless of energy level.
        # ------------------------------------------------------------------
        use_subtitle = (
            self._use_subtitle_fallback
            and vad_speech_ratio < self._vad_fallback_threshold
            and bool(whisper_segments)
        )

        if use_subtitle:
            self._logger.warning(
                f"  {audio_id}: VAD speech_ratio={vad_speech_ratio:.1%} < "
                f"threshold={self._vad_fallback_threshold:.0%} — "
                "falling back to subtitle/Whisper segments for slicing"
            )
            # Each element: {start_ms, end_ms, duration_ms, text}
            sliceable = self._merge_whisper_segments(whisper_segments)
            segment_source = "subtitle"
            self._logger.info(
                f"  {audio_id}: merged {len(whisper_segments)} subtitle segments "
                f"→ {len(sliceable)} chunks"
            )
        else:
            if not raw_vad_segments:
                self._logger.warning(f"No VAD segments in metadata for {audio_id}")
                return []
            merged_vad = self._merge_vad_segments(raw_vad_segments)
            self._logger.info(
                f"  {audio_id}: merged {len(raw_vad_segments)} VAD segments "
                f"→ {len(merged_vad)} chunks "
                f"(gap≤{self._merge_max_gap_ms:.0f}ms, max≤{self._max_duration_ms/1000:.0f}s)"
            )
            text_assignments = _build_exclusive_alignment(merged_vad, whisper_segments)
            # Convert to unified format with pre-resolved text
            sliceable = [
                {
                    "start_ms": seg["start_ms"],
                    "end_ms": seg["end_ms"],
                    "duration_ms": seg["duration_ms"],
                    "text": (
                        " ".join(text_assignments.get(i, [])).strip()
                        or full_transcript
                    ),
                    "_vad_seg": seg,
                }
                for i, seg in enumerate(merged_vad)
            ]
            segment_source = "vad"

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

        results: list[dict[str, Any]] = []
        seg_counter = 0

        for chunk in sliceable:
            start_ms: float = chunk["start_ms"]
            end_ms: float = chunk["end_ms"]
            duration_ms: float = end_ms - start_ms

            if duration_ms < self._min_duration_ms:
                self._logger.debug(
                    f"  Skip {audio_id} chunk @{start_ms:.0f}ms: "
                    f"{duration_ms:.0f}ms < min {self._min_duration_ms:.0f}ms"
                )
                continue

            if duration_ms > self._max_duration_ms:
                self._logger.debug(
                    f"  Skip {audio_id} chunk @{start_ms:.0f}ms: "
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

            text = chunk["text"]

            seg_id = f"{audio_id}_seg{seg_counter:03d}"
            seg_wav_path = audio_out / f"{seg_id}.wav"

            save_audio(segment_audio, str(seg_wav_path), sr=sr)

            info = get_audio_info(segment_audio, sr)

            # Build vad_segment metadata field — use original VAD info when
            # available, otherwise synthesise it from the chunk boundaries.
            vad_seg_ref = chunk.get("_vad_seg")
            vad_segment_meta: dict[str, Any] = (
                {
                    "segment_id": vad_seg_ref["segment_id"],
                    "start_ms": vad_seg_ref["start_ms"],
                    "end_ms": vad_seg_ref["end_ms"],
                    "duration_ms": vad_seg_ref["duration_ms"],
                    "confidence": vad_seg_ref["confidence"],
                }
                if vad_seg_ref is not None
                else {
                    "segment_id": seg_counter,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": round(duration_ms, 2),
                    "confidence": 1.0,
                }
            )

            seg_meta: dict[str, Any] = {
                "audio_id": seg_id,
                "source_audio_id": audio_id,
                "segment_index": seg_counter,
                "segment_source": segment_source,
                "processing_pipeline_version": _PIPELINE_VERSION,
                "stt_result": {
                    "transcript": text,
                    "language": stt.get("language", "ko"),
                    "confidence_score": stt.get("confidence_score"),
                    "model_used": stt.get("model_used"),
                    "model_version": stt.get("model_version"),
                },
                "vad_segment": vad_segment_meta,
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
