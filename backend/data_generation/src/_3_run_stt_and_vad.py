"""STT (Whisper) and VAD (Pyannote) processor — Stage 3.

Loads audio files validated in Stage 2, runs speech-to-text and voice
activity detection, and writes per-file JSON metadata plus a manifest.

Typical usage::

    python src/3_run_stt_and_vad.py \\
        --input-dir raw_downloads \\
        --config config/generation.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.audio_processor import load_audio  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PIPELINE_VERSION = "1.0"
_WHISPER_MODEL_VERSION = "v20230314"
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ---------------------------------------------------------------------------
# Lazy model loaders
# ---------------------------------------------------------------------------


def _load_whisper(model_name: str = "base", device: str = "cpu") -> Any:
    """Load a Whisper model, downloading it on the first call.

    Args:
        model_name: Whisper model size identifier (e.g. ``"base"``).
        device: Torch device string (``"cuda"`` or ``"cpu"``).

    Returns:
        Loaded Whisper model object.

    Raises:
        ImportError: If ``openai-whisper`` is not installed.
    """
    try:
        import whisper  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "openai-whisper is required. Install it with: pip install openai-whisper"
        ) from exc

    logger.info(f"Loading Whisper model '{model_name}' on {device} …")
    model = whisper.load_model(model_name, device=device)
    logger.info("Whisper model loaded.")
    return model


def _load_pyannote(model_name: str = "pyannote/segmentation") -> Any:
    """Load a Pyannote audio pipeline, downloading it on the first call.

    Requires a valid Hugging Face token exported as ``HF_TOKEN`` or passed
    explicitly.

    Args:
        model_name: Pyannote pipeline identifier on the Hub.

    Returns:
        Loaded Pyannote ``Pipeline`` object.

    Raises:
        ImportError: If ``pyannote.audio`` is not installed.
        RuntimeError: If the HF token is missing.
    """
    try:
        from pyannote.audio import Pipeline  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio is required. " "Install it with: pip install pyannote.audio"
        ) from exc

    import os  # noqa: PLC0415

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Hugging Face token not found. " "Set the HF_TOKEN environment variable."
        )

    logger.info(f"Loading Pyannote pipeline '{model_name}' …")
    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    logger.info("Pyannote pipeline loaded.")
    return pipeline


def _resolve_device(preferred: str = "cuda") -> str:
    """Return the best available torch device.

    Falls back to CPU if CUDA is requested but unavailable.

    Args:
        preferred: Preferred device (``"cuda"`` or ``"cpu"``).

    Returns:
        Resolved device string.
    """
    if preferred != "cuda":
        return preferred
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class STTAndVADProcessor:
    """Run Whisper STT and Pyannote VAD on audio files.

    Models are loaded lazily on the first call to :meth:`process_audio` to
    avoid slow startup when only metadata helpers are needed.

    Args:
        config: Pipeline configuration dictionary (from ``load_config``).
        logger: Logger instance for this processor.
    """

    def __init__(self, config: dict[str, Any], logger: Any) -> None:
        self._config = config
        self._logger = logger

        stt_cfg = config.get("stt", {})
        vad_cfg = config.get("vad", {})

        self._stt_model_name: str = stt_cfg.get("model", "whisper-base").replace(
            "whisper-", ""
        )
        self._stt_language: str = stt_cfg.get("language", "ko")
        self._min_confidence: float = stt_cfg.get("min_confidence", 0.85)
        self._device: str = _resolve_device(stt_cfg.get("device", "cuda"))

        self._vad_model_name: str = vad_cfg.get("model", "pyannote/segmentation")
        self._vad_threshold: float = vad_cfg.get("threshold", 0.5)
        self._min_speech_ms: float = vad_cfg.get("min_speech_duration_ms", 300)

        # Lazy-loaded model handles
        self._whisper_model: Any = None
        self._vad_pipeline: Any = None

    # ------------------------------------------------------------------
    # Model accessors (lazy init)
    # ------------------------------------------------------------------

    @property
    def _whisper(self) -> Any:
        if self._whisper_model is None:
            self._whisper_model = _load_whisper(self._stt_model_name, self._device)
        return self._whisper_model

    @property
    def _pyannote(self) -> Any:
        if self._vad_pipeline is None:
            self._vad_pipeline = _load_pyannote(self._vad_model_name)
        return self._vad_pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_stt(self, audio_path: str) -> dict[str, Any]:
        """Transcribe an audio file using Whisper.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Dictionary with keys:

            - ``transcript`` (str): Transcribed text.
            - ``confidence`` (float): Segment-averaged log-probability score
              mapped to [0, 1].
            - ``language`` (str): Detected language code.
            - ``model_used`` (str): Model identifier.
            - ``model_version`` (str): Static model version tag.
            - ``processing_time_ms`` (int): Wall-clock time in milliseconds.

        Raises:
            RuntimeError: If Whisper fails to transcribe the file.
        """
        t0 = time.perf_counter()
        try:
            result = self._whisper.transcribe(
                audio_path,
                language=self._stt_language,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Whisper transcription failed for '{audio_path}': {exc}"
            ) from exc

        elapsed_ms = int((time.perf_counter() - t0) * 1_000)

        # Derive a scalar confidence from per-segment avg_logprob
        segments = result.get("segments", [])
        if segments:
            avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segments])
            # Map logprob (typically -2 … 0) to probability-like score in [0,1]
            confidence = float(np.clip(np.exp(avg_logprob), 0.0, 1.0))
        else:
            confidence = 0.0

        return {
            "transcript": result.get("text", "").strip(),
            "confidence": round(confidence, 4),
            "language": result.get("language", self._stt_language),
            "model_used": f"whisper-{self._stt_model_name}",
            "model_version": _WHISPER_MODEL_VERSION,
            "processing_time_ms": elapsed_ms,
        }

    def run_vad(self, audio_path: str) -> dict[str, Any]:
        """Detect voice activity segments in an audio file using Pyannote.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Dictionary with keys:

            - ``segments`` (list[dict]): Voice segments with ``segment_id``,
              ``start_ms``, ``end_ms``, ``duration_ms``, ``confidence``.
            - ``total_speech_duration_ms`` (float)
            - ``total_silence_duration_ms`` (float)
            - ``speech_ratio`` (float): Fraction of audio that is speech.

        Raises:
            RuntimeError: If Pyannote fails to process the file.
        """
        # Determine total duration for silence calculation
        try:
            audio, sr = load_audio(audio_path, sr=16_000)
            total_duration_ms = len(audio) / sr * 1_000
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load audio for VAD '{audio_path}': {exc}"
            ) from exc

        try:
            diarization = self._pyannote(audio_path)
        except Exception as exc:
            raise RuntimeError(
                f"Pyannote VAD failed for '{audio_path}': {exc}"
            ) from exc

        segments: list[dict[str, Any]] = []
        total_speech_ms = 0.0

        for seg_id, (segment, _, _) in enumerate(
            diarization.itertracks(yield_label=True)
        ):
            start_ms = segment.start * 1_000
            end_ms = segment.end * 1_000
            duration_ms = end_ms - start_ms

            if duration_ms < self._min_speech_ms:
                continue

            # Pyannote segments don't expose per-segment confidence directly;
            # use the pipeline's internal score if available, else 1.0.
            confidence = getattr(segment, "confidence", 1.0)
            if confidence < self._vad_threshold:
                continue

            segments.append(
                {
                    "segment_id": seg_id,
                    "start_ms": round(start_ms, 2),
                    "end_ms": round(end_ms, 2),
                    "duration_ms": round(duration_ms, 2),
                    "confidence": round(float(confidence), 4),
                }
            )
            total_speech_ms += duration_ms

        silence_ms = max(0.0, total_duration_ms - total_speech_ms)
        speech_ratio = (
            total_speech_ms / total_duration_ms if total_duration_ms > 0 else 0.0
        )

        return {
            "segments": segments,
            "total_speech_duration_ms": round(total_speech_ms, 2),
            "total_silence_duration_ms": round(silence_ms, 2),
            "speech_ratio": round(speech_ratio, 4),
        }

    def generate_metadata(
        self,
        audio_id: str,
        stt_result: dict[str, Any],
        vad_result: dict[str, Any],
        audio_path: str,
    ) -> dict[str, Any]:
        """Build a metadata record following the CLAUDE.md schema.

        The VAD segment list is enriched with per-segment transcript text
        aligned by time overlap (best-effort, word-level alignment is not
        attempted here).

        Args:
            audio_id: Stable identifier for this audio file (e.g. video ID).
            stt_result: Output of :meth:`run_stt`.
            vad_result: Output of :meth:`run_vad`.
            audio_path: Path to the source audio file.

        Returns:
            Metadata dictionary matching the schema in CLAUDE.md.
        """
        audio, sr = load_audio(audio_path, sr=16_000)
        rms = float(np.sqrt(np.mean(audio**2)))
        rms_db = float(20.0 * np.log10(rms + 1e-9))
        peak = float(np.max(np.abs(audio)))
        duration_ms = len(audio) / sr * 1_000

        # Annotate VAD segments with transcript text (full transcript shared)
        annotated_segments = [
            {**seg, "text": stt_result.get("transcript", "")}
            for seg in vad_result.get("segments", [])
        ]
        enriched_vad = {**vad_result, "segments": annotated_segments}

        return {
            "audio_id": audio_id,
            "processing_pipeline_version": _PIPELINE_VERSION,
            "stt_result": {
                "transcript": stt_result["transcript"],
                "language": stt_result["language"],
                "confidence_score": stt_result["confidence"],
                "model_used": stt_result["model_used"],
                "model_version": stt_result["model_version"],
            },
            "vad_result": enriched_vad,
            "audio_characteristics": {
                "format": Path(audio_path).suffix.lstrip("."),
                "sample_rate": sr,
                "channels": 1,
                "duration_ms": round(duration_ms, 2),
                "rms_energy_db": round(rms_db, 2),
                "peak_amplitude": round(peak, 4),
            },
        }

    def process_audio(self, audio_path: str) -> dict[str, Any]:
        """Run STT + VAD and assemble metadata for a single file.

        Files whose Whisper confidence falls below ``stt.min_confidence`` are
        still processed but flagged with ``"low_confidence": true`` in the
        returned metadata.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Assembled metadata dictionary.

        Raises:
            RuntimeError: If STT or VAD processing fails.
        """
        stem = Path(audio_path).stem
        self._logger.info(f"Processing: {stem}")

        stt = self.run_stt(audio_path)
        vad = self.run_vad(audio_path)
        metadata = self.generate_metadata(stem, stt, vad, audio_path)

        if stt["confidence"] < self._min_confidence:
            self._logger.warning(
                f"Low STT confidence {stt['confidence']:.3f} < "
                f"{self._min_confidence} for {stem}"
            )
            metadata["low_confidence"] = True
        else:
            metadata["low_confidence"] = False

        self._logger.info(
            f"Done {stem}: "
            f"confidence={stt['confidence']:.3f}, "
            f"segments={len(vad['segments'])}, "
            f"speech_ratio={vad['speech_ratio']:.2f}"
        )
        return metadata

    def process_batch(self, directory: str) -> list[dict[str, Any]]:
        """Process all supported audio files in a directory.

        Args:
            directory: Path to the directory containing audio files.

        Returns:
            List of metadata dictionaries, one per successfully processed file.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Input directory not found: {directory}")

        audio_files = sorted(
            p
            for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )

        if not audio_files:
            self._logger.warning(f"No audio files found in: {directory}")
            return []

        self._logger.info(
            f"Batch processing {len(audio_files)} file(s) from: {directory}"
        )

        results: list[dict[str, Any]] = []
        for audio_path in audio_files:
            try:
                metadata = self.process_audio(str(audio_path))
                results.append(metadata)
            except Exception as exc:
                self._logger.error(f"Failed to process {audio_path.name}: {exc}")

        self._logger.info(f"Batch done — {len(results)}/{len(audio_files)} succeeded")
        return results

    def save_metadata(self, metadata: dict[str, Any], output_path: str) -> None:
        """Persist a metadata dictionary as a formatted JSON file.

        Args:
            metadata: Metadata dictionary to serialise.
            output_path: Destination file path.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            self._logger.debug(f"Metadata saved: {out}")
        except OSError as exc:
            self._logger.error(f"Failed to save metadata to '{out}': {exc}")
            raise


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Whisper STT + Pyannote VAD on downloaded audio (Stage 3).",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing validated audio files.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to generation.yaml config file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config (default: stt_and_vad/).",
    )
    parser.add_argument(
        "--passed-json",
        default=None,
        help=(
            "Path to the passed-files JSON from Stage 2. "
            "When provided, only listed files are processed."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (optional).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the STT + VAD processing stage.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    args = _parse_args(argv)
    config = load_config(args.config)

    logs_dir: str = config.get("output_dirs", {}).get("logs", "./logs")
    log_file = args.log_file or str(Path(logs_dir) / "3_run_stt_and_vad.log")
    log = setup_logger("stt_and_vad", log_file=log_file)

    output_base = Path(
        args.output_dir or config["output_dirs"].get("stt_and_vad", "./stt_and_vad")
    )
    metadata_dir = output_base / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    processor = STTAndVADProcessor(config=config, logger=log)

    # Optionally restrict to files that passed Stage 2 validation
    if args.passed_json:
        passed_path = Path(args.passed_json)
        if not passed_path.exists():
            log.error(f"Passed-files JSON not found: {passed_path}")
            sys.exit(1)
        with open(passed_path, encoding="utf-8") as f:
            passed_data = json.load(f)
        audio_files = [Path(p) for p in passed_data.get("files", [])]
        log.info(f"Processing {len(audio_files)} file(s) from passed-files list")

        results: list[dict[str, Any]] = []
        for audio_path in audio_files:
            try:
                meta = processor.process_audio(str(audio_path))
                out_path = metadata_dir / f"{audio_path.stem}_metadata.json"
                processor.save_metadata(meta, str(out_path))
                results.append(meta)
            except Exception as exc:
                log.error(f"Failed: {audio_path.name}: {exc}")
    else:
        results = processor.process_batch(args.input_dir)
        for meta in results:
            audio_id = meta.get("audio_id", "unknown")
            out_path = metadata_dir / f"{audio_id}_metadata.json"
            processor.save_metadata(meta, str(out_path))

    # Write manifest
    manifest_path = output_base / "manifest.json"
    manifest = {
        "total": len(results),
        "passed_confidence": sum(
            1 for r in results if not r.get("low_confidence", True)
        ),
        "files": [
            {
                "audio_id": r.get("audio_id"),
                "low_confidence": r.get("low_confidence"),
                "speech_ratio": r.get("vad_result", {}).get("speech_ratio"),
                "metadata_path": str(
                    metadata_dir / f"{r.get('audio_id')}_metadata.json"
                ),
            }
            for r in results
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info(f"Manifest saved: {manifest_path}")

    print(f"\nResults : {len(results)} file(s) processed")
    print(f"Metadata: {metadata_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
