"""Unit tests for the YouTube audio downloader (Stage 1)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> dict:
    return {
        "youtube": {"playlist_ids": ["PLtest"], "max_videos_per_playlist": 5},
        "output_dirs": {"raw_downloads": "./raw_downloads", "logs": "./logs"},
    }


@pytest.fixture()
def mock_logger() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def downloader(tmp_path, config, mock_logger):
    """YouTubeDownloader with yt-dlp stubbed out."""
    with patch("src.1_download_youtube._import_yt_dlp") as mock_import:
        mock_yt = MagicMock()
        mock_import.return_value = mock_yt

        # Delayed import so the module-level yt-dlp import is mocked
        from src._1_download_youtube import YouTubeDownloader  # noqa: PLC0415

        dl = YouTubeDownloader(
            output_dir=str(tmp_path),
            config=config,
            logger=mock_logger,
        )
        dl._yt_dlp = mock_yt
        return dl, mock_yt, tmp_path


# ---------------------------------------------------------------------------
# _is_valid_wav
# ---------------------------------------------------------------------------

class TestIsValidWav:
    def test_valid_wav_returns_true(self, tmp_path):
        from src._1_download_youtube import _is_valid_wav  # noqa: PLC0415

        wav = tmp_path / "good.wav"
        audio = np.zeros(16_000, dtype=np.float32)
        sf.write(str(wav), audio, 16_000)

        assert _is_valid_wav(wav) is True

    def test_missing_file_returns_false(self, tmp_path):
        from src._1_download_youtube import _is_valid_wav  # noqa: PLC0415

        assert _is_valid_wav(tmp_path / "ghost.wav") is False

    def test_empty_file_returns_false(self, tmp_path):
        from src._1_download_youtube import _is_valid_wav  # noqa: PLC0415

        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        assert _is_valid_wav(empty) is False

    def test_corrupt_file_returns_false(self, tmp_path):
        from src._1_download_youtube import _is_valid_wav  # noqa: PLC0415

        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not audio data at all")
        assert _is_valid_wav(bad) is False


# ---------------------------------------------------------------------------
# _save_metadata
# ---------------------------------------------------------------------------

class TestSaveMetadata:
    def test_writes_json_file(self, downloader):
        dl, _, tmp_path = downloader
        meta = {
            "video_id": "abc123",
            "title": "Test Video",
            "url": "https://youtube.com/watch?v=abc123",
            "audio_type": "asmr",
            "downloaded_at": "2026-01-01T00:00:00Z",
            "wav_path": str(tmp_path / "abc123.wav"),
            "sample_rate": 16_000,
        }
        dl._save_metadata("abc123", meta)

        json_path = tmp_path / "abc123_metadata.json"
        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["video_id"] == "abc123"
        assert loaded["sample_rate"] == 16_000

    def test_metadata_has_expected_keys(self, downloader):
        dl, _, tmp_path = downloader
        required_keys = {"video_id", "title", "url", "audio_type", "downloaded_at"}
        meta = {k: "x" for k in required_keys}
        dl._save_metadata("vid001", meta)

        with open(tmp_path / "vid001_metadata.json") as f:
            loaded = json.load(f)
        for key in required_keys:
            assert key in loaded


# ---------------------------------------------------------------------------
# get_downloaded_files
# ---------------------------------------------------------------------------

class TestGetDownloadedFiles:
    def test_empty_initially(self, downloader):
        dl, _, _ = downloader
        assert dl.get_downloaded_files() == []

    def test_returns_copy(self, downloader):
        dl, _, _ = downloader
        files = dl.get_downloaded_files()
        files.append("injected")
        assert "injected" not in dl.get_downloaded_files()


# ---------------------------------------------------------------------------
# download_single — skip logic
# ---------------------------------------------------------------------------

class TestDownloadSingleSkip:
    def test_skips_existing_valid_file(self, downloader, tmp_path):
        dl, mock_yt, _ = downloader

        # Pre-create a valid WAV
        wav_path = tmp_path / "vid42.wav"
        audio = np.zeros(16_000, dtype=np.float32)
        sf.write(str(wav_path), audio, 16_000)

        # Stub get_video_info
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.extract_info.return_value = {
            "id": "vid42",
            "title": "Skippable",
            "duration": 10,
            "uploader": "tester",
        }
        mock_yt.YoutubeDL.return_value = mock_ctx

        result = dl.download_single("https://youtube.com/watch?v=vid42")

        assert result == str(wav_path)
        # YoutubeDL.download should NOT have been called (file was skipped)
        mock_ctx.download.assert_not_called()


# ---------------------------------------------------------------------------
# _extract_playlist_urls
# ---------------------------------------------------------------------------

class TestExtractPlaylistUrls:
    def test_returns_video_urls(self, downloader):
        dl, mock_yt, _ = downloader

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.extract_info.return_value = {
            "entries": [{"id": "v1"}, {"id": "v2"}, {"id": "v3"}]
        }
        mock_yt.YoutubeDL.return_value = mock_ctx

        urls = dl._extract_playlist_urls("https://youtube.com/playlist?list=PLtest")
        assert len(urls) == 3
        assert all("youtube.com/watch?v=" in u for u in urls)

    def test_empty_playlist_returns_empty_list(self, downloader):
        dl, mock_yt, _ = downloader

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.extract_info.return_value = {"entries": []}
        mock_yt.YoutubeDL.return_value = mock_ctx

        urls = dl._extract_playlist_urls("https://youtube.com/playlist?list=PLempty")
        assert urls == []
