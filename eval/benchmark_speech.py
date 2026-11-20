"""Speech round trip: Piper speaks support sentences, whisper transcribes them, and the
Callie's normalizer turns spoken ids back into digits.

    python -m eval.benchmark_speech                       small.en and base.en
    python -m eval.benchmark_speech --models small.en --noise 15

Reports, per whisper model: how many ids and keywords came back, the word error rate
against what was said (after normalizing), and the time per clip. Synthetic speech is
clean, so treat these numbers as an upper bound; a real microphone is checked in V5.
"""

from __future__ import annotations

import argparse
import re
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from callie.speech.stt import WhisperTranscriber
from callie.speech.text import normalize_transcript
from callie.speech.tts import PiperSpeaker

SENTENCES = Path(__file__).resolve().parent / "speech_sentences.yaml"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = _words(reference), _words(hypothesis)
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(1, len(ref))


def add_noise(samples: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal = samples.astype(np.float32)
    noise = rng.normal(0, 1, len(signal)).astype(np.float32)
    noise *= np.sqrt(np.mean(signal**2) / (10 ** (snr_db / 10)) / np.mean(noise**2))
    return np.clip(signal + noise, -32768, 32767).astype(np.int16)


def check(item: dict[str, Any], heard: str) -> tuple[int, int]:
    text = normalize_transcript(heard)
    numbers = {int(n) for n in re.findall(r"\b\d+\b", text)}
    ids = sum(i in numbers for i in item["ids"])
    keywords = sum(k.lower() in text.lower().replace(" ", "") or k.lower() in _words(text) for k in item["keywords"])
    return ids, keywords


def expected_text(item: dict[str, Any]) -> str:
    return normalize_transcript(item["say"])


def run(models: list[str], noise: float | None) -> list[dict[str, Any]]:
    items = yaml.safe_load(SENTENCES.read_text(encoding="utf-8"))["sentences"]
    speaker = PiperSpeaker()
    clips = []
    tts_seconds = []
    for item in items:
        speech = speaker.speak(item["say"])
        assert speech is not None
        tts_seconds.append(speech.seconds / max(speech.audio_seconds, 1e-6))
        samples = speech.samples if noise is None else add_noise(speech.samples, noise, np.random.default_rng(0))
        clips.append((item, (speech.sample_rate, samples), speech.audio_seconds))
    print(f"Piper: {len(clips)} clips, {statistics.mean(c[2] for c in clips):.1f} s of audio each, "
          f"synthesized at {1 / statistics.median(tts_seconds):.0f}x real time")
    rows = []
    for model in models:
        stt = WhisperTranscriber(model)
        started = time.perf_counter()
        stt.load()
        load_s = time.perf_counter() - started
        stt.transcribe(clips[0][1])  # warm-up
        times, errors, ids, id_total, keys, key_total, misses = [], [], 0, 0, 0, 0, []
        for item, audio, _ in clips:
            heard = stt.transcribe(audio)
            times.append(heard.seconds)
            got_ids, got_keys = check(item, heard.text)
            ids, id_total = ids + got_ids, id_total + len(item["ids"])
            keys, key_total = keys + got_keys, key_total + len(item["keywords"])
            errors.append(wer(expected_text(item), normalize_transcript(heard.text)))
            if got_ids < len(item["ids"]) or got_keys < len(item["keywords"]):
                misses.append(f"{item['say']!r} -> {heard.text!r} -> {normalize_transcript(heard.text)!r}")
        rows.append({"model": model, "ids": f"{ids}/{id_total}", "keywords": f"{keys}/{key_total}",
                     "wer": statistics.mean(errors), "median_s": statistics.median(times), "max_s": max(times),
                     "load_s": load_s, "misses": misses})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="small.en,base.en")
    parser.add_argument("--noise", type=float, help="add white noise at this SNR in dB")
    args = parser.parse_args()
    rows = run(args.models.split(","), args.noise)
    label = f", +noise {args.noise:g} dB SNR" if args.noise is not None else ""
    print(f"\n| Whisper{label} | Ids recovered | Keywords | WER | Median / max per clip | Model load |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['ids']} | {r['keywords']} | {r['wer']:.3f} | {r['median_s']:.2f} / "
              f"{r['max_s']:.2f} s | {r['load_s']:.1f} s |")
    for r in rows:
        for miss in r["misses"]:
            print(f"  [{r['model']}] {miss}")


if __name__ == "__main__":
    main()
