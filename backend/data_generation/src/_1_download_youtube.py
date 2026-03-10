"""YouTube audio downloader for the STT data generation pipeline.

Downloads audio from YouTube playlists (or single videos) using yt-dlp,
converts to WAV, and saves per-video metadata alongside each file.

Typical usage::

    python src/1_download_youtube.py --config config/generation.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve project root so the module can be run directly from any CWD.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_yt_dlp() -> Any:
    """Import yt-dlp lazily so the module is importable without the package.

    Returns:
        The ``yt_dlp`` module.

    Raises:
        ImportError: If yt-dlp is not installed.
    """
    try:
        import yt_dlp  # noqa: PLC0415

        return yt_dlp
    except ImportError as exc:
        raise ImportError(
            "yt-dlp is required. Install it with: pip install yt-dlp"
        ) from exc


def _is_valid_wav(path: Path) -> bool:
    """Return True if *path* is a readable, non-empty WAV file.

    Uses soundfile for fast header-only validation; falls back to librosa for
    edge cases (e.g. files without a proper WAV header after conversion).

    Args:
        path: Path to the WAV file.

    Returns:
        True if the file is valid audio, False otherwise.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        info = sf.info(str(path))
        return info.frames > 0
    except Exception:
        try:
            audio, _ = librosa.load(str(path), sr=None, duration=0.1)
            return len(audio) > 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class YouTubeDownloader:
    """Download audio from YouTube playlists or individual videos.

    Downloaded files are stored as 16 kHz mono WAV files. A JSON sidecar
    (``<video_id>_metadata.json``) is written next to each WAV file.

    Args:
        output_dir: Directory where downloaded files will be saved.
        config: Pipeline configuration dictionary (from ``load_config``).
        logger: Logger instance for this downloader.
    """

    _WAV_SAMPLE_RATE = 16_000

    def __init__(
        self,
        output_dir: str,
        config: dict[str, Any],
        logger: Any,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._logger = logger
        self._yt_dlp = _import_yt_dlp()
        self._downloaded: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_playlist(
        self,
        playlist_url: str,
        audio_type: str = "asmr",
    ) -> list[str]:
        """Download all videos in a YouTube playlist.

        Already-downloaded videos (WAV file present and valid) are skipped.

        Args:
            playlist_url: Full YouTube playlist URL or playlist ID.
            audio_type: Descriptive label stored in each video's metadata
                (e.g. ``"asmr"``).

        Returns:
            List of absolute paths to successfully downloaded WAV files.
        """
        if not playlist_url.startswith("http"):
            playlist_url = f"https://www.youtube.com/playlist?list={playlist_url}"

        self._logger.info(f"Fetching playlist info: {playlist_url}")
        video_urls = self._extract_playlist_urls(playlist_url)

        max_videos: int = (
            self._config.get("youtube", {}).get("max_videos_per_playlist", 10)
        )
        video_urls = video_urls[:max_videos]
        self._logger.info(
            f"Found {len(video_urls)} video(s) to process (limit={max_videos})"
        )

        paths: list[str] = []
        for url in tqdm(video_urls, desc="Downloading playlist", unit="video"):
            try:
                wav_path = self.download_single(url, audio_type=audio_type)
                paths.append(wav_path)
            except Exception as exc:
                self._logger.warning(f"Skipping {url}: {exc}")

        self._logger.info(
            f"Playlist done — {len(paths)}/{len(video_urls)} succeeded"
        )
        return paths

    def download_single(
        self,
        video_url: str,
        audio_type: str = "asmr",
    ) -> str:
        """Download a single YouTube video as a WAV file.

        If a valid WAV file for this video already exists, the download is
        skipped and the existing path is returned.

        Args:
            video_url: YouTube video URL.
            audio_type: Descriptive label stored in the metadata.

        Returns:
            Absolute path to the downloaded WAV file.

        Raises:
            RuntimeError: If the download or conversion fails after retries.
        """
        info = self._get_video_info(video_url)
        video_id: str = info["id"]
        wav_path = self._output_dir / f"{video_id}.wav"

        if wav_path.exists() and _is_valid_wav(wav_path):
            self._logger.info(f"Already downloaded, skipping: {video_id}")
            if str(wav_path) not in self._downloaded:
                self._downloaded.append(str(wav_path))
            return str(wav_path)

        self._logger.info(f"Downloading: {video_id} — {info.get('title', 'unknown')}")
        self._download_with_retry(video_url, video_id)

        if not _is_valid_wav(wav_path):
            raise RuntimeError(
                f"Downloaded file is missing or corrupted: {wav_path}"
            )

        metadata = {
            "video_id": video_id,
            "title": info.get("title", ""),
            "url": video_url,
            "audio_type": audio_type,
            "duration_s": info.get("duration"),
            "uploader": info.get("uploader", ""),
            "downloaded_at": datetime.now(tz=timezone.utc).isoformat(),
            "wav_path": str(wav_path),
            "sample_rate": self._WAV_SAMPLE_RATE,
        }
        self._save_metadata(video_id, metadata)

        self._downloaded.append(str(wav_path))
        self._logger.info(f"Saved: {wav_path}")
        return str(wav_path)

    def get_downloaded_files(self) -> list[str]:
        """Return paths of all files downloaded in this session.

        Returns:
            List of absolute WAV file paths downloaded so far.
        """
        return list(self._downloaded)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_video_info(self, video_url: str) -> dict[str, Any]:
        """Fetch video metadata without downloading the media.

        Args:
            video_url: YouTube video URL.

        Returns:
            yt-dlp info dictionary (id, title, duration, uploader, …).

        Raises:
            RuntimeError: If yt-dlp cannot retrieve the info.
        """
        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        try:
            with self._yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(video_url, download=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to retrieve video info for '{video_url}': {exc}"
            ) from exc

    def _extract_playlist_urls(self, playlist_url: str) -> list[str]:
        """Return individual video URLs from a playlist.

        Args:
            playlist_url: Full YouTube playlist URL.

        Returns:
            Ordered list of video URLs.

        Raises:
            RuntimeError: If the playlist cannot be fetched.
        """
        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,  # metadata only, no download
            "skip_download": True,
        }
        try:
            with self._yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch playlist '{playlist_url}': {exc}"
            ) from exc

        entries = info.get("entries") or []
        urls = []
        for entry in entries:
            vid_id = entry.get("id") or entry.get("url")
            if vid_id:
                urls.append(f"https://www.youtube.com/watch?v={vid_id}")
        return urls

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _download_with_retry(self, video_url: str, video_id: str) -> None:
        """Download audio and convert to 16 kHz mono WAV (3 attempts).

        Uses yt-dlp's post-processor chain:
        1. Download best audio stream.
        2. Convert to WAV via ffmpeg.
        3. Resample to 16 kHz mono via ffmpeg.

        Args:
            video_url: YouTube video URL.
            video_id: Video ID used to name the output file.

        Raises:
            Exception: Re-raised after all retry attempts are exhausted.
        """
        out_template = str(self._output_dir / f"{video_id}.%(ext)s")

        postprocessors = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            },
            {
                # Resample to 16 kHz mono after extraction
                "key": "FFmpegPostProcessor",  # handled via postprocessor_args
            },
        ]

        opts: dict[str, Any] = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "0",
                }
            ],
            # Resample + mono via ffmpeg args applied after extraction
            "postprocessor_args": {
                "FFmpegExtractAudio": [
                    "-ar", str(self._WAV_SAMPLE_RATE),
                    "-ac", "1",
                ]
            },
            "quiet": True,
            "no_warnings": True,
            "keepvideo": False,
        }

        try:
            with self._yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([video_url])
        except Exception as exc:
            self._logger.warning(
                f"Download attempt failed for {video_id}: {exc}. Retrying…"
            )
            raise

    def _save_metadata(self, video_id: str, metadata: dict[str, Any]) -> None:
        """Persist per-video metadata as a JSON sidecar file.

        Args:
            video_id: YouTube video ID (used for the filename).
            metadata: Dictionary to serialise.
        """
        meta_path = self._output_dir / f"{video_id}_metadata.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            self._logger.debug(f"Metadata saved: {meta_path}")
        except OSError as exc:
            self._logger.error(f"Failed to save metadata for {video_id}: {exc}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YouTube ASMR audio for STT data generation.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to generation.yaml config file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--playlist-id",
        default=None,
        help="Single playlist ID to download (overrides config).",
    )
    parser.add_argument(
        "--video-url",
        default=None,
        help="Single video URL to download.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (optional).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the YouTube download stage.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    args = _parse_args(argv)

    config = load_config(args.config)

    log_file = args.log_file or os.path.join(
        config.get("output_dirs", {}).get("logs", "./logs"),
        "1_download_youtube.log",
    )
    log = setup_logger("download_youtube", log_file=log_file)

    output_dir: str = args.output_dir or config["output_dirs"]["raw_downloads"]
    downloader = YouTubeDownloader(output_dir=output_dir, config=config, logger=log)

    if args.video_url:
        path = downloader.download_single(args.video_url)
        log.info(f"Downloaded: {path}")
        return

    playlist_ids: list[str]
    if args.playlist_id:
        playlist_ids = [args.playlist_id]
    else:
        playlist_ids = config.get("youtube", {}).get("playlist_ids", [])

    if not playlist_ids:
        log.error("No playlist IDs found in config or --playlist-id argument.")
        sys.exit(1)

    all_paths: list[str] = []
    for pid in playlist_ids:
        paths = downloader.download_playlist(pid)
        all_paths.extend(paths)

    log.info(f"Total downloaded: {len(all_paths)} file(s)")
    log.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
