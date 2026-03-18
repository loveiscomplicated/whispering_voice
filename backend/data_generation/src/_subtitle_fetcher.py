"""YouTube subtitle fetcher — drop-in STT replacement.

Downloads and parses YouTube captions using yt-dlp and returns a result
dictionary that is structurally identical to the one produced by
``STTAndVADProcessor.run_stt()``, so the rest of the pipeline (VAD
alignment, segmentation, dataset generation) requires no modification.

Priority order (configurable):
    1. Manual / community subtitles  → ``subtitle.prefer_manual: true``
    2. Auto-generated captions       → ``subtitle.allow_auto_generated: true``
    3. Whisper STT fallback          → ``subtitle.fallback_to_stt: true``

Subtitle files are cached as JSON under ``subtitle.cache_dir`` so repeated
runs skip the network round-trip.

Typical usage::

    fetcher = YouTubeSubtitleFetcher(config=config, logger=log)
    stt_result = fetcher.fetch("y8KJ5ga23zw")   # video ID = stem of WAV
    if stt_result is None:
        stt_result = processor.run_stt(audio_path)   # fall back to Whisper
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _ts_to_ms(ts: str) -> float:
    """Convert ``HH:MM:SS,mmm`` or ``HH:MM:SS.mmm`` to milliseconds.

    Args:
        ts: Timestamp string in SRT or VTT format.

    Returns:
        Time in milliseconds as a float.
    """
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        elif len(parts) == 2:
            h, m, s = 0, int(parts[0]), float(parts[1])
        else:
            return 0.0
        return (h * 3600 + m * 60 + s) * 1_000
    except (ValueError, IndexError):
        return 0.0


# ---------------------------------------------------------------------------
# Subtitle parsers
# ---------------------------------------------------------------------------


def _strip_tags(text: str) -> str:
    """Remove VTT/HTML inline tags and entities from a text string.

    Handles YouTube-specific tags such as ``<00:00:01.280>``, ``<c>``,
    ``</c>``, as well as ordinary HTML entities (``&nbsp;`` etc.).

    Args:
        text: Raw subtitle text possibly containing inline markup.

    Returns:
        Cleaned plain-text string with normalised whitespace.
    """
    # Remove VTT timestamp tags  e.g. <00:00:01.280>
    text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", text)
    # Remove remaining angle-bracket tags  e.g. <c>, </c>, <i>, <b>
    text = re.sub(r"<[^>]+>", "", text)
    # Remove HTML entities  e.g. &nbsp; &lrm;
    text = re.sub(r"&[a-zA-Z#\d]+;", " ", text)
    # Collapse whitespace
    return " ".join(text.split())


def _parse_srt(content: str) -> list[dict[str, Any]]:
    """Parse an SRT subtitle file into a list of segment dicts.

    Args:
        content: Full text of the ``.srt`` file.

    Returns:
        List of ``{"start_ms", "end_ms", "text"}`` dicts, ordered by time.
    """
    segments: list[dict[str, Any]] = []
    # SRT blocks are separated by one or more blank lines
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        # Find the timestamp line (contains -->)
        ts_idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ts_idx is None:
            continue
        m = re.match(
            r"([\d:,.]+)\s*-->\s*([\d:,.]+)",
            lines[ts_idx],
        )
        if not m:
            continue
        start_ms = _ts_to_ms(m.group(1))
        end_ms = _ts_to_ms(m.group(2))
        text = _strip_tags(" ".join(lines[ts_idx + 1 :]))
        if text:
            segments.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
    return segments


def _parse_vtt(content: str) -> list[dict[str, Any]]:
    """Parse a WebVTT subtitle file, including YouTube's rolling-caption format.

    YouTube auto-generated VTT files use a "rolling" structure where each
    cue incrementally reveals more words for the same time window.  The
    deduplication pass retains only the *final* (most complete) version of
    each group by skipping any cue whose text is a strict prefix of the
    immediately following cue.

    Args:
        content: Full text of the ``.vtt`` file.

    Returns:
        List of ``{"start_ms", "end_ms", "text"}`` dicts, ordered by time.
    """
    raw: list[dict[str, Any]] = []

    # A VTT cue block: optional cue-id line, then timestamp line, then text
    cue_re = re.compile(
        r"([\d]{2}:[\d]{2}:[\d]{2}[.,][\d]{3})"
        r"\s*-->\s*"
        r"([\d]{2}:[\d]{2}:[\d]{2}[.,][\d]{3})"
        r"[^\n]*\n"           # optional positioning tags on the same line
        r"((?:.+\n?)+?)"      # one or more text lines (non-greedy)
        r"(?=\n\s*\n|\Z)",    # terminated by blank line or end-of-string
        re.MULTILINE,
    )
    for m in cue_re.finditer(content):
        start_ms = _ts_to_ms(m.group(1))
        end_ms = _ts_to_ms(m.group(2))
        text = _strip_tags(m.group(3))
        if text:
            raw.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})

    if not raw:
        return []

    # Deduplicate rolling captions: if text[i] is a prefix of text[i+1],
    # drop cue[i] (it is an intermediate state of the same sentence).
    segments: list[dict[str, Any]] = []
    for i, cue in enumerate(raw):
        if i + 1 < len(raw):
            nxt = raw[i + 1]["text"]
            if nxt.startswith(cue["text"]) or cue["text"] == nxt:
                continue  # discard intermediate rolling state
        segments.append(cue)

    return segments


def _parse_subtitle_file(path: Path) -> list[dict[str, Any]] | None:
    """Dispatch to the correct parser based on file extension.

    Args:
        path: Path to the downloaded subtitle file (``.srt`` or ``.vtt``).

    Returns:
        Parsed segment list, or ``None`` if the file cannot be read / parsed.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None

    suffix = path.suffix.lower()
    if suffix == ".srt":
        return _parse_srt(content) or None
    if suffix == ".vtt":
        return _parse_vtt(content) or None
    return None


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


class YouTubeSubtitleFetcher:
    """Download YouTube subtitles and convert them to an STT-compatible result.

    The output dictionary is structurally identical to what
    ``STTAndVADProcessor.run_stt()`` returns, so it can be passed directly to
    ``STTAndVADProcessor.generate_metadata()`` and all downstream stages work
    without modification.

    Configuration is read from ``config["subtitle"]``.

    Args:
        config: Pipeline configuration dictionary (from ``load_config``).
        logger: Logger instance for diagnostic messages.
    """

    def __init__(self, config: dict[str, Any], logger: Any) -> None:
        self._logger = logger
        sub_cfg: dict[str, Any] = config.get("subtitle", {})

        self._language: str = sub_cfg.get("language", "ko")
        self._prefer_manual: bool = sub_cfg.get("prefer_manual", True)
        self._allow_auto: bool = sub_cfg.get("allow_auto_generated", True)
        self._cache_dir: Path = Path(
            sub_cfg.get(
                "cache_dir",
                "./backend/data_generation/subtitles",
            )
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, video_id: str) -> dict[str, Any] | None:
        """Return an STT-compatible result for *video_id*, or ``None``.

        Lookup order:

        1. Local JSON cache (avoids repeat downloads).
        2. Manual/community subtitles (when ``prefer_manual`` is ``true``).
        3. Auto-generated captions (when ``allow_auto_generated`` is ``true``).

        Args:
            video_id: YouTube video identifier (the stem of the WAV filename,
                e.g. ``"y8KJ5ga23zw"``).

        Returns:
            ``run_stt()``-compatible dictionary on success, ``None`` otherwise.
        """
        # 1. Cache hit
        cached = self._load_cache(video_id)
        if cached is not None:
            self._logger.info(f"Subtitle cache hit for {video_id}")
            return cached

        # 2. Attempt downloads in priority order
        attempts: list[tuple[bool, str]] = []  # (is_auto, label)
        if self._prefer_manual:
            attempts.append((False, "manual"))
            if self._allow_auto:
                attempts.append((True, "auto-generated"))
        else:
            if self._allow_auto:
                attempts.append((True, "auto-generated"))
            attempts.append((False, "manual"))

        for is_auto, label in attempts:
            sub_path = self._download(video_id, is_auto=is_auto)
            if sub_path is None:
                continue

            segments = _parse_subtitle_file(sub_path)
            if not segments:
                self._logger.warning(
                    f"Subtitle file for {video_id} parsed empty ({label})"
                )
                continue

            result = self._build_result(segments, is_auto=is_auto)
            self._save_cache(video_id, result)
            self._logger.info(
                f"Subtitles loaded for {video_id} "
                f"({label}, {len(segments)} cues)"
            )
            return result

        self._logger.info(f"No subtitles found for {video_id}")
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download(self, video_id: str, *, is_auto: bool) -> Path | None:
        """Run yt-dlp to download a subtitle file for *video_id*.

        Downloads are saved under ``cache_dir/{video_id}/``.

        Args:
            video_id: YouTube video identifier.
            is_auto: ``True`` to request auto-generated captions; ``False``
                for manual / community subtitles.

        Returns:
            Path to the downloaded subtitle file, or ``None`` on failure.
        """
        out_dir = self._cache_dir / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        url = f"https://www.youtube.com/watch?v={video_id}"
        sub_flag = "--write-auto-subs" if is_auto else "--write-subs"

        cmd = [
            "yt-dlp",
            sub_flag,
            "--skip-download",
            "--sub-langs", self._language,
            # Download in VTT (YouTube native) and also as SRT fallback
            "--sub-format", "vtt/srt/best",
            "--convert-subs", "srt",       # convert to SRT for clean parsing
            "-o", str(out_dir / "%(id)s.%(ext)s"),
            "--no-playlist",
            "--quiet",
            url,
        ]

        t0 = time.perf_counter()
        self._logger.info(
            f"Downloading {'auto' if is_auto else 'manual'} subtitles "
            f"for {video_id} …"
        )
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            self._logger.error(
                "yt-dlp not found. Install it with: pip install yt-dlp"
            )
            return None
        except subprocess.TimeoutExpired:
            self._logger.warning(f"yt-dlp timed out for {video_id}")
            return None

        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            self._logger.debug(
                f"yt-dlp exit {result.returncode} for {video_id}: "
                f"{result.stderr.strip()[:200]}"
            )
            return None

        # yt-dlp may produce  video_id.ko.srt  or  video_id.ko-KR.srt  etc.
        # Glob for any matching subtitle file in the output directory.
        candidates = sorted(out_dir.glob(f"{video_id}.*.srt"))
        if not candidates:
            candidates = sorted(out_dir.glob(f"{video_id}.*.vtt"))

        if not candidates:
            self._logger.debug(f"No subtitle file produced for {video_id}")
            return None

        path = candidates[0]
        self._logger.debug(
            f"Subtitle downloaded: {path.name} in {elapsed:.1f}s"
        )
        return path

    def _build_result(
        self,
        segments: list[dict[str, Any]],
        *,
        is_auto: bool,
    ) -> dict[str, Any]:
        """Convert parsed subtitle segments into a ``run_stt()``-compatible dict.

        Args:
            segments: List of ``{"start_ms", "end_ms", "text"}`` dicts from
                the subtitle parser.
            is_auto: Whether the source is auto-generated captions.

        Returns:
            Dictionary with the same keys as ``STTAndVADProcessor.run_stt()``.
        """
        full_transcript = " ".join(s["text"] for s in segments).strip()

        return {
            "transcript": full_transcript,
            # Subtitles are treated as high-confidence by default.
            # Auto-generated caps are assigned 0.9 to reflect occasional errors.
            "confidence": 0.9 if is_auto else 1.0,
            "language": self._language,
            "model_used": "youtube-subtitle",
            "model_version": "auto-generated" if is_auto else "manual",
            "processing_time_ms": 0,
            # Field name matches what generate_metadata() reads
            "whisper_segments": segments,
        }

    def _cache_path(self, video_id: str) -> Path:
        return self._cache_dir / video_id / f"{video_id}_subtitle.json"

    def _load_cache(self, video_id: str) -> dict[str, Any] | None:
        path = self._cache_path(video_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, video_id: str, result: dict[str, Any]) -> None:
        path = self._cache_path(video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._logger.warning(f"Failed to cache subtitles for {video_id}: {exc}")
