import io
import os
import sys
import json
import time
import datetime

import numpy as np
import pandas as pd
from typing import Literal
from dotenv import load_dotenv
import whisper
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import noisereduce as nr

load_dotenv()


def save_result(
    result_df: pd.DataFrame,
    snr: int,
    noise_type: str,
    denoize_model_name: str,
    ref: str,
    pred: str,
    wer: float,
    cer: float,
    no_speech_prob: float,
    avg_logprob: float,
    is_hallucination: (
        Literal["confirmed_hallucination", "suspected", "unchecked"] | None
    ) = None,
):
    """
    Save the result as pandas.DataFrame

    Args:
        snr (int): SNR. ex) -20 -10 0 5
        denoize_model_name (str): The name of denoizing(noize reduction) model.
        ref (str): Answer Transcript. It is the subtitle or the STT result of ASMR only (before synthesized).
        pred (str): The model's prediction.
        wer (float): Word Error Rate.
                     The metric used to measure the accuracy of Automatic Speech Recognition (ASR) systems,
                     representing the percentage of incorrect words in a transcript compared to the reference
        cer (float): Character Error Rate (CER)
                     A standard metric used to evaluate the performance of Automatic Speech Recognition (ASR) system.
                     It measures the percentage of characters that were incorrectly predicted compared to a reference text.

                     While Word Error Rate (WER) looks at entire words, CER zooms in on the individual character level,
                     making it particularly useful for languages(특히 한국어) with complex morphology or for tasks where every letter counts (like digit recognition).
        no_speech_prob (float): Metadata of Whisper model. 해당 세그먼트가 "음성이 아닐 확률".
        avg_logpob (float): Metadata of Whisper model. Whisper가 출력한 토큰들의 평균 로그 확률. 값이 0에 가까울수록 모델이 자신의 출력을 확신하고 있고, 음수로 클수록 불확실하다는 뜻.
        is_hallucination (str or None): Whether the model's prediction is hallucination or not?
                                         Options -> confirmed_hallucination / suspected / None / unchecked
    """
    row_dict = {
        "snr": snr,
        "noise_type": noise_type,
        "denoize_model_name": denoize_model_name,
        "ref": ref,
        "pred": pred,
        "wer": wer,
        "cer": cer,
        "no_speech_prob": no_speech_prob,
        "avg_logprob": avg_logprob,
        "is_hallucination": is_hallucination,
    }
    new_row = pd.DataFrame([row_dict])
    if result_df.empty:
        return new_row
    return pd.concat([result_df, new_row], ignore_index=True)


# ---------------------------------------------------------------------------
# Groq transcription helpers
# ---------------------------------------------------------------------------


def _numpy_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Convert a float32 numpy audio array to WAV bytes for API upload."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def transcribe_with_groq(
    audio: np.ndarray,
    sr: int,
    client,
    model: str = "whisper-large-v3",
    language: str = "ko",
    max_retries: int = 5,
) -> dict:
    """Send *audio* to Groq Whisper API and return a dict matching the local
    Whisper output format: ``{"text": str, "segments": [...]}``.

    SDK 내부 retry는 비활성화(Groq(max_retries=0))하고, Retry-After 헤더를
    직접 읽어 대기 시간을 로그에 기록한 뒤 재시도한다.
    """
    from groq import RateLimitError

    wav_bytes = _numpy_to_wav_bytes(audio, sr)

    resp = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            resp = client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model=model,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
            break
        except RateLimitError as e:
            elapsed = time.time() - t0
            # Groq가 Retry-After 헤더로 권장 대기 시간을 알려주면 그 값을 우선 사용한다.
            wait: float = 5 * (2**attempt)  # fallback: 5s → 10s → 20s → 40s
            try:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    wait = float(retry_after)
            except Exception:
                pass

            if attempt < max_retries - 1:
                logger.warning(
                    "Groq rate limit (elapsed=%.1fs) — waiting %.0fs before retry %d/%d",
                    elapsed,
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
            else:
                raise

    segments = []
    if resp is not None and hasattr(resp, "segments") and resp.segments:
        for seg in resp.segments:
            segments.append(
                {
                    "avg_logprob": float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                    "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                    "text": getattr(seg, "text", ""),
                    "start": float(getattr(seg, "start", 0.0)),
                    "end": float(getattr(seg, "end", 0.0)),
                }
            )

    return {
        "text": (resp.text if resp is not None and resp.text else ""),
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


def get_syn_path_list(data_generation_dir):
    syn_par_dir = os.path.join(data_generation_dir, "synthesized")
    snr_levels = os.listdir(syn_par_dir)
    syn_path_list = []
    for snr in snr_levels:
        snr_path = os.path.join(syn_par_dir, snr)
        situations = os.listdir(snr_path)
        for s in situations:
            path = os.path.join(snr_path, s)
            syn_path_list.append(path)
    return syn_path_list


now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

cur_dir = os.path.dirname(__file__)
data_generation_dir = os.path.abspath(os.path.join(cur_dir, "..", ".."))
log_path = os.path.join(cur_dir, "result", f"track_a_{timestamp}.log")
result_path = os.path.join(cur_dir, "result", f"track_a_{timestamp}.csv")

sys.path.insert(0, data_generation_dir)

from src.utils.logger import setup_logger

logger = setup_logger(name="Track A", log_file=log_path)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def get_answer(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data["transcript"]


def normalize(text: str) -> str:
    """공백 정규화 및 소문자 변환"""
    return " ".join(text.strip().lower().split())


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate (편집거리 기반)"""
    r, h = list(ref), list(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = (
                d[i - 1][j - 1]
                if r[i - 1] == h[j - 1]
                else min(
                    d[i - 1][j] + 1,  # 삭제
                    d[i][j - 1] + 1,  # 삽입
                    d[i - 1][j - 1] + 1,  # 대체
                )
            )
    return d[len(r)][len(h)] / max(len(r), 1)


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate (편집거리 기반)"""
    r, h = ref.split(), hyp.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = (
                d[i - 1][j - 1]
                if r[i - 1] == h[j - 1]
                else min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1,
                )
            )
    return d[len(r)][len(h)] / max(len(r), 1)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def get_data_dict(synthesized_data_dir: str) -> dict:
    logger.info("Loading synthesized data from: %s", synthesized_data_dir)

    data_dict: dict = {}

    for filename in os.listdir(synthesized_data_dir):
        name, extension = filename.rsplit(".", 1)
        data_dict.setdefault(name, {})
        filepath = os.path.join(synthesized_data_dir, filename)
        if extension == "wav":
            data_dict[name]["path"] = filepath
        elif extension == "json":
            data_dict[name]["answer"] = get_answer(filepath)

    logger.info("Loaded %d samples", len(data_dict))
    return data_dict


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def get_result(data_dict: dict, snr: int, noise_type: str) -> pd.DataFrame:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Get a key at https://console.groq.com and export it."
        )
    client = Groq(api_key=api_key, max_retries=0)
    logger.info("Groq client ready (model: whisper-large-v3)")

    rows: list[dict] = []

    def transcribe(name: str) -> None:
        audio_path = data_dict[name]["path"]
        audio = whisper.load_audio(audio_path)
        audio_denoised = nr.reduce_noise(
            y=audio,
            sr=16000,
            stationary=False,
            prop_decrease=0.8,
        )

        result = transcribe_with_groq(audio_denoised, 16000, client)

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

        rows.append({
            "snr": snr,
            "noise_type": noise_type,
            "denoize_model_name": "noisereduce",
            "ref": ref,
            "pred": pred,
            "wer": sample_wer,
            "cer": sample_cer,
            "no_speech_prob": avg_no_speech_prob,
            "avg_logprob": avg_logprob,
            "is_hallucination": None,
        })

        logger.info(
            "[%s]  CER: %.4f  WER: %.4f  no_speech: %s  logprob: %s",
            name,
            sample_cer,
            sample_wer,
            f"{avg_no_speech_prob:.4f}" if avg_no_speech_prob is not None else "N/A",
            f"{avg_logprob:.4f}" if avg_logprob is not None else "N/A",
        )

    with logging_redirect_tqdm(loggers=[logger]):
        for name in tqdm(data_dict.keys(), desc="Evaluating"):
            transcribe(name)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def parse_path_info(synthesized_data_dir: str) -> tuple[int, str]:
    """경로에서 SNR 값과 노이즈 타입을 파싱한다.

    예) .../synthesized/snr_-10/traffic -> (-10, "traffic")
    """
    parts = synthesized_data_dir.rstrip("/").split(os.sep)
    noise_type = parts[-1]
    snr_str = parts[-2]  # e.g. "snr_-10"
    snr_value = int(snr_str.replace("snr_", ""))
    return snr_value, noise_type


def run_track_a(syn_path_list: list) -> None:
    logger.info("=== track_a experiment start (timestamp: %s) ===", timestamp)
    logger.info("result_path : %s", result_path)
    logger.info("log_path    : %s", log_path)

    all_results = []
    for synthesized_data_dir in syn_path_list:
        snr_value, noise_type = parse_path_info(synthesized_data_dir)
        logger.info(
            "--- %s  (SNR=%d, noise=%s) ---",
            synthesized_data_dir,
            snr_value,
            noise_type,
        )
        data_dict = get_data_dict(synthesized_data_dir)
        result_df = get_result(data_dict, snr=snr_value, noise_type=noise_type)
        all_results.append(result_df)

    combined_df = pd.concat(all_results, ignore_index=True)

    avg_cer = combined_df["cer"].mean()
    avg_wer = combined_df["wer"].mean()
    logger.info("=" * 60)
    logger.info("  전체 샘플 수  : %d", len(combined_df))
    logger.info("  전체 평균 CER : %.4f  (%.2f%%)", avg_cer, avg_cer * 100)
    logger.info("  전체 평균 WER : %.4f  (%.2f%%)", avg_wer, avg_wer * 100)
    logger.info("=" * 60)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    combined_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    logger.info("Results saved to: %s", result_path)
    logger.info("=== track_a experiment done ===")


if __name__ == "__main__":
    syn_path_list = get_syn_path_list(data_generation_dir)
    run_track_a(syn_path_list)
