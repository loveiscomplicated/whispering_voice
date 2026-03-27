"""
This script must be run by additional environment STT_env_2 !!!
See requirements_STT_env_2.txt in the root.

  - CLI 옵션 추가:
  # Groq (기본값)
  python track_b.py --backend groq

  # 로컬 Whisper large
  python track_b.py --backend local

  # 로컬 Whisper medium으로
  python track_b.py --backend local --whisper-model medium

  # denoiser 선택과 함께
  python track_b.py --backend local --denoisers noisereduce deepfilternet

"""

import os

# Must be set before torch/MPS initializes — Whisper large uses sparse buffers
# that SparseMPS doesn't support, so we need CPU fallback for those ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import whisper
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

load_dotenv()

from track_a import (
    save_result,
    cer,
    wer,
    normalize,
    get_data_dict,
    get_syn_path_list,
    parse_path_info,
    transcribe_with_groq,
)


# ---------------------------------------------------------------------------
# local Whisper transcription wrapper
# ---------------------------------------------------------------------------

_WHISPER_MODEL_CACHE: dict = {}


def transcribe_with_local_whisper(
    audio: np.ndarray,
    sr: int,
    model_name: str = "large",
    language: str = "ko",
) -> dict:
    """로컬 Whisper 모델로 transcribe한다.

    Returns the same dict shape as transcribe_with_groq:
    ``{"text": str, "segments": [...]}``
    """
    import torch

    if model_name not in _WHISPER_MODEL_CACHE:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading local Whisper model '%s' on %s …", model_name, device)
        model = whisper.load_model(model_name, device="cpu")
        if device == "mps":
            # SparseMPS does not support COO sparse tensor creation ops.
            # Convert sparse buffers to dense before moving to MPS.
            for name, buf in list(model.named_buffers()):
                if buf.is_sparse:
                    *path, attr = name.split(".")
                    parent = model
                    for p in path:
                        parent = getattr(parent, p)
                    setattr(parent, attr, buf.to_dense())
        _WHISPER_MODEL_CACHE[model_name] = model.to(device)

    model = _WHISPER_MODEL_CACHE[model_name]
    result = model.transcribe(audio, language=language, fp16=False)

    segments = [
        {
            "avg_logprob": float(s.get("avg_logprob", 0.0) or 0.0),
            "no_speech_prob": float(s.get("no_speech_prob", 0.0) or 0.0),
            "text": s.get("text", ""),
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
        }
        for s in result.get("segments") or []
    ]
    return {"text": result.get("text", ""), "segments": segments}


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

cur_dir = os.path.dirname(__file__)
data_generation_dir = os.path.abspath(os.path.join(cur_dir, "..", ".."))
log_path = os.path.join(cur_dir, "result", f"track_b_{timestamp}.log")
result_path = os.path.join(cur_dir, "result", f"track_b_{timestamp}.csv")

sys.path.insert(0, data_generation_dir)

from src.utils.logger import setup_logger

logger = setup_logger(name="Track B", log_file=log_path)


# ---------------------------------------------------------------------------
# denoiser wrappers
# Each wrapper takes (audio: np.ndarray, sr: int) -> np.ndarray
# ---------------------------------------------------------------------------


def denoise_noisereduce(audio: np.ndarray, sr: int) -> np.ndarray:
    import noisereduce as nr

    return nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)


_DF_MODEL_CACHE: dict = {}


def denoise_deepfilternet(audio: np.ndarray, sr: int) -> np.ndarray:
    """DeepFilterNet — 경량 실시간 노이즈 제거 모델."""
    import torch
    import torchaudio
    from df.enhance import enhance, init_df

    if "model" not in _DF_MODEL_CACHE:
        model, df_state, _ = init_df()
        _DF_MODEL_CACHE["model"] = model
        _DF_MODEL_CACHE["df_state"] = df_state
    model = _DF_MODEL_CACHE["model"]
    df_state = _DF_MODEL_CACHE["df_state"]
    target_sr = df_state.sr()

    # (1, T) tensor throughout — avoid numpy round-trips that break enhance()
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)
    if sr != target_sr:
        audio_t = torchaudio.functional.resample(audio_t, sr, target_sr)

    enhanced_t = enhance(model, df_state, audio_t)

    if target_sr != sr:
        enhanced_t = torchaudio.functional.resample(enhanced_t, target_sr, sr)

    return enhanced_t.squeeze(0).numpy()


def denoise_meta_denoiser(audio: np.ndarray, sr: int) -> np.ndarray:
    """Meta (Facebook) Denoiser — dns64 pretrained model."""
    import torch
    from denoiser import pretrained
    from denoiser.dsp import convert_audio

    model = pretrained.dns64().cpu()
    model.eval()

    # convert_audio expects (batch, channels, samples)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    audio_tensor = convert_audio(audio_tensor, sr, model.sample_rate, model.chin)

    with torch.no_grad():
        enhanced_tensor = model(audio_tensor)[0]

    enhanced = enhanced_tensor.squeeze().numpy()

    # resample back to 16kHz if model's sr differs
    if model.sample_rate != sr:
        from denoiser.dsp import convert_audio

        enhanced_tensor = torch.from_numpy(enhanced).unsqueeze(0).unsqueeze(0)
        enhanced = (
            convert_audio(enhanced_tensor, model.sample_rate, sr, 1).squeeze().numpy()
        )

    return enhanced


def denoise_resemble_enhance(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resemble Enhance — 음성 품질 향상 특화 모델."""
    import torch
    from resemble_enhance.enhancer.inference import enhance as _enhance

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    audio_tensor = torch.from_numpy(audio).float()
    enhanced_tensor, out_sr = _enhance(audio_tensor, sr, device=device)
    enhanced = enhanced_tensor.numpy()

    if out_sr != sr:
        import resampy

        enhanced = resampy.resample(enhanced, out_sr, sr)

    return enhanced


DENOISERS: dict[str, callable] = {
    "noisereduce": denoise_noisereduce,
    "deepfilternet": denoise_deepfilternet,
    "meta_denoiser": denoise_meta_denoiser,
    "resemble_enhance": denoise_resemble_enhance,
}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def get_result(
    data_dict: dict,
    snr: int,
    noise_type: str,
    transcribe_fn: callable,
    denoiser_name: str,
    denoiser_fn: callable,
) -> pd.DataFrame:
    """transcribe_fn signature: (audio: np.ndarray, sr: int) -> dict"""
    rows: list[dict] = []

    def transcribe(name: str) -> None:
        audio_path = data_dict[name]["path"]
        audio = whisper.load_audio(audio_path)  # float32 np.ndarray @ 16kHz

        try:
            audio_denoised = denoiser_fn(audio, sr=16000)
        except Exception as e:
            logger.warning(
                "[%s] denoiser '%s' failed: %s — using raw audio",
                name,
                denoiser_name,
                e,
            )
            audio_denoised = audio

        result = transcribe_fn(audio_denoised, 16000)

        pred = normalize(result["text"])
        ref = normalize(data_dict[name]["answer"])
        sample_cer = cer(ref, pred)
        sample_wer = wer(ref, pred)

        segments = result.get("segments", [])
        avg_no_speech_prob = (
            sum(s["no_speech_prob"] for s in segments) / len(segments)
            if segments
            else None
        )
        avg_logprob = (
            sum(s["avg_logprob"] for s in segments) / len(segments)
            if segments
            else None
        )

        rows.append(
            {
                "snr": snr,
                "noise_type": noise_type,
                "denoize_model_name": denoiser_name,
                "ref": ref,
                "pred": pred,
                "wer": sample_wer,
                "cer": sample_cer,
                "no_speech_prob": avg_no_speech_prob,
                "avg_logprob": avg_logprob,
                "is_hallucination": None,
            }
        )

        logger.info(
            "[%s | SNR=%d | %s | %s]  CER: %.4f  WER: %.4f  no_speech: %s  logprob: %s",
            name,
            snr,
            noise_type,
            denoiser_name,
            sample_cer,
            sample_wer,
            f"{avg_no_speech_prob:.4f}" if avg_no_speech_prob is not None else "N/A",
            f"{avg_logprob:.4f}" if avg_logprob is not None else "N/A",
        )

    with logging_redirect_tqdm(loggers=[logger]):
        for name in tqdm(data_dict.keys(), desc=f"{denoiser_name}"):
            transcribe(name)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_track_b(
    syn_path_list: list,
    denoisers: dict | None = None,
    backend: str = "groq",
    whisper_model: str = "large",
) -> None:
    """
    Parameters
    ----------
    backend : "groq" | "local"
        "groq"  — Groq cloud API (whisper-large-v3, requires GROQ_API_KEY)
        "local" — local Whisper model running on MPS/CPU
    whisper_model : str
        Whisper model name used when backend="local" (e.g. "large", "medium").
    """
    if denoisers is None:
        denoisers = DENOISERS

    logger.info("=== track_b experiment start (timestamp: %s) ===", timestamp)
    logger.info("result_path : %s", result_path)
    logger.info("log_path    : %s", log_path)
    logger.info("backend     : %s", backend)
    logger.info("denoisers   : %s", list(denoisers.keys()))

    # --- build transcribe_fn based on selected backend ---
    if backend == "groq":
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a key at https://console.groq.com and export it."
            )
        groq_client = Groq(api_key=api_key, max_retries=0)
        logger.info("Groq client ready (model: whisper-large-v3)")

        def transcribe_fn(audio: np.ndarray, sr: int) -> dict:
            return transcribe_with_groq(audio, sr, groq_client)

    elif backend == "local":
        logger.info("Local Whisper model: %s", whisper_model)

        def transcribe_fn(audio: np.ndarray, sr: int) -> dict:
            return transcribe_with_local_whisper(audio, sr, model_name=whisper_model)

    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'groq' or 'local'.")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    first_write = True
    total_rows = 0

    for synthesized_data_dir in syn_path_list:
        snr_value, noise_type = parse_path_info(synthesized_data_dir)
        data_dict = get_data_dict(synthesized_data_dir)

        for denoiser_name, denoiser_fn in denoisers.items():
            logger.info(
                "--- SNR=%d  noise=%s  denoiser=%s ---",
                snr_value,
                noise_type,
                denoiser_name,
            )
            result_df = get_result(
                data_dict,
                snr=snr_value,
                noise_type=noise_type,
                transcribe_fn=transcribe_fn,
                denoiser_name=denoiser_name,
                denoiser_fn=denoiser_fn,
            )

            # 배치가 끝날 때마다 CSV에 즉시 append 저장
            result_df.to_csv(
                result_path,
                mode="a",
                index=False,
                header=first_write,
                encoding="utf-8-sig",
            )
            first_write = False
            total_rows += len(result_df)
            logger.info(
                "Saved batch (%d rows, total %d) → %s",
                len(result_df),
                total_rows,
                result_path,
            )

    # 최종 요약: 이미 저장된 CSV를 다시 읽어 피벗 출력
    combined_df = pd.read_csv(result_path)
    logger.info("=" * 70)
    logger.info("  전체 샘플 수: %d", len(combined_df))

    pivot_cer = combined_df.pivot_table(
        values="cer", index="snr", columns="denoize_model_name", aggfunc="mean"
    )
    pivot_wer = combined_df.pivot_table(
        values="wer", index="snr", columns="denoize_model_name", aggfunc="mean"
    )
    logger.info("CER matrix (SNR x denoiser):\n%s", pivot_cer.to_string())
    logger.info("WER matrix (SNR x denoiser):\n%s", pivot_wer.to_string())
    logger.info("=" * 70)

    logger.info("Results saved to: %s", result_path)
    logger.info("=== track_b experiment done ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track B: denoiser comparison")
    parser.add_argument(
        "--denoisers",
        nargs="+",
        choices=list(DENOISERS.keys()),
        default=list(DENOISERS.keys()),
        help="Which denoisers to run (default: all)",
    )
    parser.add_argument(
        "--backend",
        choices=["groq", "local"],
        default="groq",
        help="STT backend: 'groq' (Groq cloud API) or 'local' (local Whisper, default: groq)",
    )
    parser.add_argument(
        "--whisper-model",
        default="large",
        help="Whisper model name for --backend=local (default: large)",
    )
    args = parser.parse_args()

    selected_denoisers = {k: DENOISERS[k] for k in args.denoisers}
    syn_path_list = get_syn_path_list(data_generation_dir)
    run_track_b(
        syn_path_list,
        denoisers=selected_denoisers,
        backend=args.backend,
        whisper_model=args.whisper_model,
    )
