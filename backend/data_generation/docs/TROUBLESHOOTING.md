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
   `logs/quality_validation_1_report.csv` shows per-file RMS energy. Very
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

4. **Verify the audio is 16 kHz mono** — Stage 2 (`quality_validation_1`)
   will flag files with wrong sample rates.

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
| Audio is truly silent | Check `rms_energy_db` in Stage 2 report |
| Threshold too high | Lower `vad.threshold` from 0.5 to 0.3 in config |
| `min_speech_duration_ms` too high | Lower from 300 to 100 |
| Wrong audio format | Ensure 16 kHz mono WAV going into Stage 3 |

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

1. Check the source ASMR is not already distorted (Stage 2 energy check).
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

The ASMR source file has near-zero RMS. This file was likely not caught by
Stage 2 (check `min_energy_db` in config — default −40 dBFS). Manually
remove the file or tighten the threshold:

```yaml
quality_validation:
  min_energy_db: -35   # more aggressive silence rejection
```

---

## Dataset Generation Issues

### Empty splits (train/val/test are empty)

Stage 6 requires audio files to be physically present. Confirm:

```bash
# Stage 3 output: check WAV files exist in stt_and_vad/
ls stt_and_vad/*.wav

# Stage 4 output: check synthesized WAVs exist
find synthesized/ -name "*.wav" | head -5
```

If Stage 3 didn't copy WAVs, copy them manually:
```bash
cp raw_downloads/*.wav stt_and_vad/
```

### `metadata.jsonl` is missing transcripts

Transcripts come from Stage 3 metadata JSONs in `stt_and_vad/metadata/`.
If the directory is empty, re-run Stage 3:

```bash
python src/data_generation_pipeline.py \
    --config config/generation.yaml \
    --start-stage 3
```

---

## General Python / Import Issues

### `ModuleNotFoundError: No module named 'src'`

Run scripts from the project root (`data_generation/`), not from inside `src/`:

```bash
# Correct
cd path/to/data_generation
python src/1_download_youtube.py --config config/generation.yaml

# Wrong
cd src
python 1_download_youtube.py ...
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
| `logs/1_download_youtube.log` | Stage 1 |
| `logs/2_quality_validation_1.log` | Stage 2 |
| `logs/3_run_stt_and_vad.log` | Stage 3 |
| `logs/4_synthesize_noise.log` | Stage 4 |
| `logs/5_quality_validation_2.log` | Stage 5 |
| `logs/6_generate_finetuning_dataset.log` | Stage 6 |
| `logs/pipeline.log` | Orchestrator |
| `logs/pipeline_checkpoint.json` | Resume state |

Increase log verbosity by changing the level in `setup_logger`:

```python
# In any stage script
log = setup_logger("my_stage", log_file="logs/debug.log", level=logging.DEBUG)
```
