import os
import sys
import datetime

import numpy as np
import pandas as pd
import whisper
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from track_a import (
    save_result,
    cer,
    wer,
    normalize,
    get_data_dict,
    get_syn_path_list,
    parse_path_info,
)


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


def denoise_deepfilternet(audio: np.ndarray, sr: int) -> np.ndarray:
    """DeepFilterNet — 경량 실시간 노이즈 제거 모델."""
    from df.enhance import enhance, init_df, load_audio, save_audio
    from df.io import resample

    model, df_state, _ = init_df()
    target_sr = df_state.sr()

    # resample to model's expected sr if needed
    if sr != target_sr:
        audio_resampled = resample(audio, sr, target_sr)
    else:
        audio_resampled = audio

    # DeepFilterNet expects (channels, samples); add channel dim
    if audio_resampled.ndim == 1:
        audio_resampled = audio_resampled[np.newaxis, :]

    import torch

    audio_tensor = torch.from_numpy(audio_resampled).float()
    enhanced_tensor = enhance(model, df_state, audio_tensor)
    enhanced = enhanced_tensor.squeeze(0).numpy()

    # resample back to whisper's 16kHz
    if target_sr != sr:
        enhanced = resample(enhanced, target_sr, sr)

    return enhanced


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
    model: "whisper.Whisper",
    denoiser_name: str,
    denoiser_fn: callable,
) -> pd.DataFrame:
    result_df = pd.DataFrame()

    def transcribe(name: str) -> None:
        nonlocal result_df

        audio_path = data_dict[name]["path"]
        audio = whisper.load_audio(audio_path)  # float32 np.ndarray @ 16kHz

        try:
            audio_denoised = denoiser_fn(audio, sr=16000)
        except Exception as e:
            logger.warning(
                "[%s] denoiser '%s' failed: %s — using raw audio", name, denoiser_name, e
            )
            audio_denoised = audio

        result = model.transcribe(
            audio_denoised,
            language="ko",
            condition_on_previous_text=False,
            no_speech_threshold=0.8,
            fp16=False,
        )

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

        result_df = save_result(
            result_df,
            snr=snr,
            noise_type=noise_type,
            denoize_model_name=denoiser_name,
            ref=ref,
            pred=pred,
            wer=sample_wer,
            cer=sample_cer,
            no_speech_prob=avg_no_speech_prob,
            avg_logprob=avg_logprob,
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

    return result_df


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_track_b(syn_path_list: list, denoisers: dict | None = None) -> None:
    if denoisers is None:
        denoisers = DENOISERS

    logger.info("=== track_b experiment start (timestamp: %s) ===", timestamp)
    logger.info("result_path : %s", result_path)
    logger.info("log_path    : %s", log_path)
    logger.info("denoisers   : %s", list(denoisers.keys()))

    logger.info("Loading Whisper large model on mps ...")
    model = whisper.load_model("large", device="mps").float()
    logger.info("Model loaded.")

    all_results = []

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
                model=model,
                denoiser_name=denoiser_name,
                denoiser_fn=denoiser_fn,
            )
            all_results.append(result_df)

    combined_df = pd.concat(all_results, ignore_index=True)

    # SNR × denoiser 매트릭스 요약 출력
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

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    combined_df.to_csv(result_path, index=False, encoding="utf-8-sig")
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
    args = parser.parse_args()

    selected_denoisers = {k: DENOISERS[k] for k in args.denoisers}
    syn_path_list = get_syn_path_list(data_generation_dir)
    run_track_b(syn_path_list, denoisers=selected_denoisers)