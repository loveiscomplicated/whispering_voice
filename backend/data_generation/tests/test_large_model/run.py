import os
import sys
import whisper

cur_dir = os.path.dirname(__file__)
"""import_path = os.path.join(cur_dir, "..", "..", "src")
import_path = os.path.abspath(import_path)

sys.path.insert(0, import_path)"""
data_generation_dir = os.path.join(cur_dir, "..", "..")
synthesized_data_dir = os.path.join(data_generation_dir, "synthesized/snr_-10/traffic")
clean_data_dir = os.path.join(data_generation_dir, "segments/audio")

import json
import noisereduce as nr
from tqdm import tqdm


def get_answer(path):
    if not os.path.exists(path):
        raise Exception("no dir")
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


listdir = os.listdir(synthesized_data_dir)
listdir_2 = os.listdir(clean_data_dir)

data_dict = {}

for i in listdir:
    name, extension = i.split(".")
    if name not in data_dict.keys():
        data_dict[name] = {}

    i_path = os.path.join(synthesized_data_dir, i)

    if extension == "wav":
        data_dict[name]["path"] = i_path
    if extension == "json":
        data_dict[name]["answer"] = get_answer(i_path)

for i in listdir_2:
    prefix, extension = i.split(".")
    name = prefix + "_snr-10_traffic"
    if name not in data_dict.keys():
        print("fuck")
        pass

    data_dict[name]["clean_path"] = os.path.join(clean_data_dir, i)


model = whisper.load_model("large", device="mps").float()

results = []
results_clean = []


def transcribe(data_dict, name, data_type):
    audio = whisper.load_audio(data_dict[name][data_type])
    audio_denoised = nr.reduce_noise(
        y=audio,
        sr=16000,
        stationary=False,
        prop_decrease=0.8,
    )

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

    if data_type == "path":
        results.append(
            {
                "name": name,
                "ref": ref,
                "pred": pred,
                "cer": sample_cer,
                "wer": sample_wer,
            }
        )
    else:  # data_type == "clean_path"
        results_clean.append(
            {
                "name": name,
                "ref": ref,
                "pred": pred,
                "cer": sample_cer,
                "wer": sample_wer,
            }
        )

    print(f"  {data_type} - PRED : {pred}")
    print(f"  {data_type} - CER  : {sample_cer:.4f}  WER : {sample_wer:.4f}")
    print()


for name in data_dict.keys():
    ref = normalize(data_dict[name]["answer"])

    print(f"[{name}]")
    print(f"  REF : {ref}")
    transcribe(data_dict, name, "path")
    transcribe(data_dict, name, "clean_path")
    print()

# 전체 평균 (synthesized)
avg_cer = sum(r["cer"] for r in results) / len(results)
avg_wer = sum(r["wer"] for r in results) / len(results)

print("=" * 60)
print("SYNTHESIZED")
print(f"샘플 수  : {len(results)}")
print(f"평균 CER : {avg_cer:.4f}  ({avg_cer * 100:.2f}%)")
print(f"평균 WER : {avg_wer:.4f}  ({avg_wer * 100:.2f}%)")
print("=" * 60)

# 전체 평균 (clean)
avg_cer_clean = sum(r["cer"] for r in results_clean) / len(results_clean)
avg_wer_clean = sum(r["wer"] for r in results_clean) / len(results_clean)

print("=" * 60)
print("CLEAN")
print(f"샘플 수  : {len(results_clean)}")
print(f"평균 CER : {avg_cer_clean:.4f}  ({avg_cer_clean * 100:.2f}%)")
print(f"평균 WER : {avg_wer_clean:.4f}  ({avg_wer_clean * 100:.2f}%)")
print("=" * 60)
