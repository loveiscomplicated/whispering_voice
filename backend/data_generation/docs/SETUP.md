# Setup Guide — STT Data Generation Pipeline

## Requirements

| Item | Minimum |
|------|---------|
| Python | 3.10+ |
| RAM | 8 GB (16 GB recommended for GPU) |
| Disk | 20 GB free per 10 ASMR videos |
| GPU | Optional (CUDA 11.8+ for fast inference) |
| ffmpeg | System-level install required by yt-dlp |

---

## 1. Install System Dependencies

### macOS
```bash
brew install ffmpeg
```

### Ubuntu / Debian
```bash
sudo apt update && sudo apt install -y ffmpeg
```

### Windows
Download from https://ffmpeg.org/download.html and add to `PATH`.

Verify:
```bash
ffmpeg -version
```

---

## 2. Create Python Environment

```bash
# Create and activate virtualenv
python3.10 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Or with conda
conda create -n stt-datagen python=3.10 -y
conda activate stt-datagen
```

---

## 3. Install Python Dependencies

```bash
pip install --upgrade pip

# Core audio + ML
pip install openai-whisper          # Whisper STT
pip install pyannote.audio==2.1.1   # VAD
pip install yt-dlp                  # YouTube downloader

# Audio processing
pip install librosa soundfile scipy numpy

# Pipeline utilities
pip install tenacity tqdm pyyaml pandas

# Dev / test
pip install pytest pytest-cov
```
> These dependencies will be already installed via requirements.txt.

> **Note:** `openai-whisper` downloads the model (~140 MB for `base`) on
> first use. Ensure you have internet access for the first run.

---

## 4. Configure Hugging Face Token (required for Pyannote)

Pyannote models are gated on the Hub. Accept the license and create a token:

1. Go to https://huggingface.co/pyannote/segmentation and accept the terms.
2. Generate a token at https://huggingface.co/settings/tokens (read access is enough).
3. Export it:

```bash
# macOS / Linux
export HF_TOKEN="hf_your_token_here"

# Windows PowerShell
$env:HF_TOKEN = "hf_your_token_here"

# Permanent (add to ~/.zshrc or ~/.bashrc)
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.zshrc
```

---

## 5. Edit Configuration

```bash
cp config/generation.yaml.example config/generation.yaml   # if example exists
# or edit config/generation.yaml directly
```

Open `config/generation.yaml` and set your playlist IDs:

```yaml
youtube:
  playlist_ids:
    - "PLxxxxxxxxxxxxxxxx"   # replace with real playlist ID
  max_videos_per_playlist: 10
```

Find a playlist ID from any YouTube playlist URL:
```
https://www.youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx
                                       ^^^^^^^^^^^^^^^^^ this part
```

---

## 6. Prepare Noise Files

The synthesis stage (Stage 6) requires background noise samples organised by type:

```
backend/data_generation/raw_downloads/noise/
├── ambient/    ← café, nature, indoor ambience (WAV/MP3)
├── traffic/    ← road, street sounds
└── office/     ← keyboard, AC, ventilation
```

Free sources:
- https://freesound.org (CC-licensed noise samples)
- https://soundbible.com

---

## 7. First Run

### Full pipeline (all 7 stages)
```bash
python backend/data_generation/src/data_generation_pipeline.py --config ./backend/data_generation/config/generation.yaml
```

### Run individual stages
```bash
# Stage 1: Download audio
python backend/data_generation/src/_1_download_youtube.py \
    --config ./backend/data_generation/config/generation.yaml

# Stage 2: Basic quality validation (loose checks on raw downloads)
python backend/data_generation/src/_2_quality_validation_basic.py \
    --input-dir ./backend/data_generation/raw_downloads \
    --config ./backend/data_generation/config/generation.yaml \
    --copy-files

# Stage 3: Preprocessing (resample, normalise, length-adjust)
python backend/data_generation/src/_3_preprocessing.py \
    --input-dir ./backend/data_generation/passed_files \
    --output-dir ./backend/data_generation/preprocessed \
    --config ./backend/data_generation/config/generation.yaml

# Stage 4: Strict quality validation (16 kHz, 1–30 s, energy range)
python backend/data_generation/src/_4_quality_validation_strict.py \
    --input-dir ./backend/data_generation/preprocessed \
    --config ./backend/data_generation/config/generation.yaml \
    --copy-files

# Stage 5: STT + VAD
python backend/data_generation/src/_3_run_stt_and_vad.py \
    --input-dir ./backend/data_generation/final_files \
    --config ./backend/data_generation/config/generation.yaml \
    --passed-json ./backend/data_generation/logs/quality_validation_strict_report_passed.json

# Stage 6: Noise synthesis
python backend/data_generation/src/_4_synthesize_noise.py \
    --config ./backend/data_generation/config/generation.yaml

# Stage 7: Dataset generation
python backend/data_generation/src/_6_generate_finetuning_dataset.py \
    --config ./backend/data_generation/config/generation.yaml
```

### Resume after interruption
```bash
python backend/data_generation/src/data_generation_pipeline.py --config ./backend/data_generation/config/generation.yaml --resume
```

### Start from a specific stage
```bash
# Re-run from Stage 3 onwards (e.g. after tweaking preprocessing config)
python backend/data_generation/src/data_generation_pipeline.py \
    --config ./backend/data_generation/config/generation.yaml \
    --start-stage 3
```

---

## 8. Run Tests

```bash
# All tests
pytest backend/data_generation/tests/ -v

# Unit tests only (no ML models loaded)
pytest backend/data_generation/tests/unit/ -v

# With coverage report
pytest backend/data_generation/tests/ -v --cov=src --cov-report=term-missing
```

---

## 9. Expected Output Structure

After a successful full run:

```
raw_downloads/              # Stage 1: original audio + per-video metadata JSON
passed_files/               # Stage 2: files that passed basic validation
rejected_files/             # Stage 2: files rejected at basic validation
preprocessed/               # Stage 3: resampled, normalised, length-adjusted WAVs
final_files/                # Stage 4: files that passed strict validation
validation_failed/          # Stage 4: files rejected at strict validation
stt_and_vad/
  metadata/                 # Stage 5: STT + VAD metadata per file
  manifest.json             # index of all processed files
synthesized/
  snr_05/ambient/           # Stage 6: noise-mixed WAVs + JSON sidecars
  snr_10/ambient/
  …
dataset/
  train/audio/              # Stage 7: final fine-tuning data
  train/metadata.jsonl
  val/audio/
  val/metadata.jsonl
  test/audio/
  test/metadata.jsonl
  manifest.json
  statistics.json
logs/                       # per-stage log files + CSV reports + checkpoint
```
