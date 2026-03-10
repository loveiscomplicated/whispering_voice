# Troubleshooting Guide

## Quick Diagnostics

Before diving into specific errors, run this checklist:

```bash
# 1. ffmpeg available?
ffmpeg -version

# 2. Python packages installed?
python -c "import whisper, pyannote.audio, yt_dlp, librosa, soundfile; print('OK')"

# 3. HF token set?
python -c "import os; print('HF_TOKEN:', 'SET' if os.getenv('HF_TOKEN') else 'MISSING')"

# 4. CUDA available (optional)?
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 5. Config file valid?
python -c "from backend.data_generation.src.utils.config import load_config; print(load_config('./backend/data_generation/config/generation.yaml'))"
```

---

## YouTube Download Issues

### `ERROR: Sign in to confirm you're not a bot`

YouTube is blocking the request.

**Fix:**
```bash
# Update yt-dlp to the latest version
pip install -U yt-dlp

# Use cookies from your browser (requires browser extension or manual export)
yt-dlp --cookies-from-browser chrome <URL>

# Or pass a cookies file
python src/1_download_youtube.py \
    --config config/generation.yaml \
    --video-url <URL>
# Add to yt-dlp opts in code: {"cookiesfile": "cookies.txt"}
```

### `ERROR: Requested format is not available`

**Fix:**
```bash
# Check available formats for the video
yt-dlp -F <video_URL>

# The downloader uses "bestaudio/best" — this should always work
# If not, the video may be region-locked or age-restricted
```

### Download stops mid-way (network timeout)

The pipeline uses `tenacity` for automatic 3-retry with exponential backoff
(2 s → 4 s → 8 s). If all retries fail:

```bash
# Increase wait time — edit src/1_download_youtube.py
# wait=wait_exponential(multiplier=2, min=5, max=60)

# Or resume from Stage 1 after fixing connectivity
python src/data_generation_pipeline.py \
    --config config/generation.yaml \
    --start-stage 1
```

### `FileNotFoundError: ffmpeg not found`

```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

# Verify yt-dlp can find it
yt-dlp --check-formats <URL>
```

---

## STT / Whisper Issues

### First run is slow or hangs

Whisper downloads the model on first use:

| Model | Size | Download time (50 Mbps) |
|-------|------|------------------------|
| tiny  | 39 MB  | ~8 s |
| base  | 74 MB  | ~15 s |
| small | 244 MB | ~40 s |

This is normal. Subsequent runs use the cached model at
`~/.cache/whisper/` (Linux/macOS) or `%USERPROFILE%\.cache\whisper\` (Windows).

### `RuntimeError: CUDA out of memory`

```bash
# Option 1: Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python src/3_run_stt_and_vad.py --input-dir raw_downloads --config config/generation.yaml

# Option 2: Use a smaller model
# In config/generation.yaml:
#   stt:
#     model: "whisper-tiny"
#     device: "cpu"
```

### Transcription quality is poor

1. **Check confidence scores** — the CSV report at
   `logs/quality_validation_strict_report.csv` shows per-file RMS energy. Very
   quiet files (< −40 dBFS) produce unreliable transcripts.

2. **Lower the confidence threshold cautiously:**
   ```yaml
   # config/generation.yaml
   stt:
     min_confidence: 0.75   # default 0.85 — lower = more data, lower quality
   ```

3. **Try a larger model:**
   ```yaml
   stt:
     model: "whisper-small"
   ```

4. **Verify the audio is 16 kHz mono** — Stage 4 (`quality_validation_strict`)
   enforces 16 kHz and will reject files with wrong sample rates. Stage 3
   (preprocessing) resamples all files to 16 kHz automatically.

---

## VAD / Pyannote Issues

### `RuntimeError: HF_TOKEN not set`

```bash
export HF_TOKEN="hf_your_token_here"
```

If the token is set but still failing:
```bash
# Re-authenticate
python -c "from huggingface_hub import login; login()"
```

### `OSError: pyannote/segmentation is not a local folder and is not a valid model identifier`

The model hasn't been downloaded yet, or the token is invalid.

```bash
# Force download manually
python -c "
from pyannote.audio import Pipeline
import os
p = Pipeline.from_pretrained('pyannote/segmentation',
                              use_auth_token=os.getenv('HF_TOKEN'))
print('Download OK')
"
```

### VAD finds no segments

Possible causes:

| Cause | Fix |
|-------|-----|
| Audio is truly silent | Check `rms_energy_db` in Stage 4 strict report |
| Threshold too high | Lower `vad.threshold` from 0.5 to 0.3 in config |
| `min_speech_duration_ms` too high | Lower from 300 to 100 |
| Wrong audio format | Ensure 16 kHz mono WAV going into Stage 5; Stage 3 preprocessing handles resampling automatically |

---

## Noise Synthesis Issues

### No noise files found

```
WARNING: No noise files found in: raw_downloads/noise
```

You need to provide noise files manually:

```
raw_downloads/noise/
├── ambient/  ← put .wav or .mp3 files here
├── traffic/
└── office/
```

### Synthesized audio is clipping

The synthesizer peak-normalises the output, so true hard clipping should not
occur. If the waveform *sounds* distorted:

1. Check the source ASMR is not already distorted (Stage 4 strict energy check).
2. Use a higher SNR level — at very low SNR (e.g. 0 dB) the noise dominates.
3. Verify noise files are not themselves clipping:
   ```bash
   python -c "
   from src.utils.audio_processor import load_audio, get_audio_info
   a, sr = load_audio('raw_downloads/noise/ambient/my_noise.wav')
   print(get_audio_info(a, sr))
   "
   ```

### `ValueError: Clean signal is silent`

The ASMR source file has near-zero RMS. This can happen if:

- Stage 3 preprocessing failed to normalise the file (check `logs/3_preprocessing.log`).
- The file passed Stage 4 strict validation despite low energy (check `min_energy_db`).

Tighten the strict energy threshold or the preprocessing target level:

```yaml
quality_validation_strict:
  min_energy_db: -35   # more aggressive silence rejection

preprocessing:
  target_rms_db: -20   # ensure files are normalised to a reasonable level
```

---

## Dataset Generation Issues

### Empty splits (train/val/test are empty)

Stage 7 requires audio files to be physically present in `stt_and_vad/`.
Stage 5 copies them there automatically from `final_files/`. Confirm:

```bash
# Stage 5 output: check WAV files exist in stt_and_vad/
ls backend/data_generation/stt_and_vad/*.wav

# Stage 6 output: check synthesized WAVs exist
find backend/data_generation/synthesized/ -name "*.wav" | head -5
```

If Stage 5 didn't copy WAVs, copy them manually:
```bash
cp backend/data_generation/final_files/*.wav backend/data_generation/stt_and_vad/
```

### `metadata.jsonl` is missing transcripts

Transcripts come from Stage 5 metadata JSONs in `stt_and_vad/metadata/`.
If the directory is empty, re-run from Stage 5:

```bash
python backend/data_generation/src/data_generation_pipeline.py \
    --config ./backend/data_generation/config/generation.yaml \
    --start-stage 5
```

---

## General Python / Import Issues

### `ModuleNotFoundError: No module named 'src'`

Run scripts from the project root (the directory that contains `backend/`), not from inside `src/`:

```bash
# Correct — run from repo root
python backend/data_generation/src/_1_download_youtube.py \
    --config ./backend/data_generation/config/generation.yaml

# Wrong
cd backend/data_generation/src
python _1_download_youtube.py ...
```

### `SyntaxError` when importing stage modules in tests

Stage modules are named with numeric prefixes (`1_download_youtube.py`) which
Python cannot import with a plain `import` statement. The `tests/conftest.py`
handles this automatically via `importlib`. Ensure you run tests with:

```bash
pytest tests/ -v
# NOT: python -m pytest tests/ (works too, but run from project root)
```

### `ImportError: No module named 'torch'`

`set_seed()` in `audio_processor.py` skips PyTorch seeding gracefully if
torch is not installed. If you need GPU support:

```bash
# Install PyTorch (CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch with CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Log Files

All stage logs are written to `logs/`:

| File | Stage |
|------|-------|
| `logs/1_download_youtube.log` | Stage 1 — Download |
| `logs/2_quality_validation_basic.log` | Stage 2 — Basic validation |
| `logs/3_preprocessing.log` | Stage 3 — Preprocessing |
| `logs/4_quality_validation_strict.log` | Stage 4 — Strict validation |
| `logs/3_run_stt_and_vad.log` | Stage 5 — STT + VAD |
| `logs/4_synthesize_noise.log` | Stage 6 — Noise synthesis |
| `logs/6_generate_finetuning_dataset.log` | Stage 7 — Dataset generation |
| `logs/pipeline.log` | Orchestrator |
| `logs/pipeline_checkpoint.json` | Resume state |

Key CSV + JSON reports:

| File | Contents |
|------|----------|
| `logs/quality_validation_basic_report.csv` | Per-file basic validation results |
| `logs/quality_validation_basic_report_passed.json` | Paths that passed Stage 2 |
| `logs/quality_validation_strict_report.csv` | Per-file strict validation results |
| `logs/quality_validation_strict_report_passed.json` | Paths that passed Stage 4 (fed into Stage 5) |

Increase log verbosity by changing the level in `setup_logger`:

```python
# In any stage script
log = setup_logger("my_stage", log_file="logs/debug.log", level=logging.DEBUG)
```
