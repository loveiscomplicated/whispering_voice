# Model Selection Rationale

## STT: OpenAI Whisper (base)

### Why Whisper?

| Criterion | Assessment |
|-----------|------------|
| Noise robustness | Trained on 680,000 hours of diverse audio — handles whisper/ASMR without special tuning |
| Korean support | Native multilingual model; Korean quality is production-grade |
| Cost | Free, runs locally, no API quota |
| Accuracy (base) | WER ~10–15% on clean Korean; sufficient for ground-truth bootstrapping |
| Inference speed | ~2× real-time on CPU; ~20× on GPU for `base` |

### Why `base` and not a larger variant?

The pipeline uses confidence filtering (`min_confidence ≥ 0.85`) to discard
low-quality transcripts before they reach the dataset. A `base` model with
strict filtering produces *fewer but higher-quality* samples than a `small`
or `medium` model with loose filtering — which matters more for fine-tuning
than raw throughput.

Upgrade path if accuracy is insufficient:

```yaml
# config/generation.yaml
stt:
  model: "whisper-small"     # ~460 MB, ~2× slower, ~15% relative WER improvement
  # model: "whisper-medium"  # ~1.5 GB, better for noisy recordings
```

### Alternatives Considered

| Model | Reason Not Chosen |
|-------|------------------|
| Wav2Vec 2.0 (Korean) | Requires Korean-specific checkpoint; less robust to background noise |
| Clova Speech (Naver) | API-only, cost, data privacy concerns |
| Google Speech-to-Text | API cost, data leaves local environment |
| ESPnet (Korean) | Complex setup, limited out-of-box noise robustness |

---

## VAD: Pyannote Audio (v2.1 segmentation)

### Why Pyannote?

| Criterion | Assessment |
|-----------|------------|
| Accuracy | State-of-the-art on VoxConverse, AMI benchmarks |
| Confidence scores | Per-segment probability output enables threshold-based filtering |
| Korean support | Language-agnostic acoustic model — works without Korean-specific training |
| Segment granularity | Returns precise start/end timestamps (ms precision) |
| Integration | Simple Python API; outputs compatible with standard audio pipelines |

### What the `segmentation` pipeline detects

- Voice vs. non-voice regions (used here)
- Speaker diarization (not used in this pipeline but available if needed)

### Threshold tuning

The default threshold `0.5` is conservative. Adjust based on your data:

```yaml
# config/generation.yaml
vad:
  threshold: 0.5    # lower → more segments included (more false positives)
                    # higher → fewer segments (more false negatives)
  min_speech_duration_ms: 300   # discard very short segments (breath sounds etc.)
```

A threshold of `0.6–0.7` is recommended for ASMR content where breathing and
soft sounds can trigger false positives near the 0.5 boundary.

### Alternatives Considered

| Model | Reason Not Chosen |
|-------|------------------|
| WebRTC VAD | Binary output only, no confidence scores; poor on whisper audio |
| Silero VAD | Lighter weight but lower accuracy on overlapping speech + noise |
| Kaldi VAD | Complex setup, less maintained Python bindings |
| SpeechBrain VAD | Good accuracy but larger dependency footprint |

---

## Audio Processing Libraries

### librosa

Used for: loading audio, computing RMS, spectral analysis, resampling.

Chosen over alternatives because:
- De-facto standard for research audio processing in Python
- Handles MP3/WAV/FLAC transparently via soundfile/audioread backends
- Resampling quality is high (scipy `resample_poly` under the hood)

### soundfile

Used for: writing WAV files, fast header-only validation.

Chosen over `scipy.io.wavfile` because:
- Supports float32 natively without integer scaling
- More robust header parsing (handles non-standard WAV variants)
- Reads audio info (duration, SR) without decoding the full file

### yt-dlp

Used for: YouTube audio download.

Chosen over `youtube-dl` because:
- Actively maintained fork with more frequent format/signature updates
- Better postprocessor API for direct WAV conversion via ffmpeg
- Built-in retry logic and format selection

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2024-03 | Whisper `base` selected | Balance of speed and quality for ground-truth generation |
| 2024-03 | Pyannote v2.1 selected | Best open-source VAD accuracy at time of evaluation |
| 2024-03 | SNR range 5–25 dB selected | Covers real-world conditions from very noisy (5 dB) to near-clean (25 dB) |
| 2024-03 | 8:1:1 train/val/test split | Standard for fine-tuning datasets; grouping by audio_id prevents leakage |
