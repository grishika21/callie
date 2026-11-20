"""Phone-line speech benchmark (plan P0): can whisper understand callers over the phone?

    python -m eval.benchmark_phone_speech                    small.en and base.en, clean and noisy
    python -m eval.benchmark_phone_speech --models small.en --show-misses

Piper speaks what callers say (last names of real customers, order numbers, the V4
support sentences, catalog questions); the audio goes through what the phone network
does to it (8 kHz, μ-law, optional line noise); whisper transcribes it; Callie's own
helpers extract the order number and the last name, and match the name like the
Support Desk will. Synthetic voices are clearer than real callers: an upper bound.
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
from typing import Any

import yaml

from callie.phone.audio import phone_line
from callie.phone.verify import extract_last_name, extract_order_number, last_name_of, names_match
from callie.speech.stt import WhisperTranscriber
from callie.speech.text import normalize_transcript
from callie.speech.tts import PiperSpeaker
from data.seed import generate as gen
from eval.benchmark_speech import SENTENCES, check

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
STOP = {"the", "of", "a", "an", "and", "in", "on", "to", "for", "with", "at", "by", "from"}


def spoken_digits(number: int) -> str:
    return " ".join(DIGIT_WORDS[int(d)] for d in str(number))


def build_items(n_names: int = 24) -> list[dict[str, Any]]:
    rng = random.Random(7)
    sales, catalog = gen.generate_sales(), gen.generate_catalog()
    items: list[dict[str, Any]] = []
    for customer in rng.sample(sales.customers, n_names):
        items.append({"kind": "name", "say": f"My last name is {customer.name.split(maxsplit=1)[1]}.",
                      "name": last_name_of(customer.name)})
    for order in rng.sample(sales.orders, 8):
        items.append({"kind": "order", "say": f"My order number is {spoken_digits(order.order_id)}.",
                      "order": order.order_id})
    for book in rng.sample(catalog.books, 6):
        title = book.title.split(" (")[0].split(":")[0]
        words = [w for w in re.findall(r"[A-Za-z']+", title) if w.lower() not in STOP and len(w) > 2]
        items.append({"kind": "catalog", "say": f"Do you have an audiobook of {title}?", "keywords": words})
    for sentence in yaml.safe_load(SENTENCES.read_text(encoding="utf-8"))["sentences"]:
        items.append({"kind": "support", **sentence})
    return items


def score(item: dict[str, Any], heard: str) -> tuple[bool, str]:
    if item["kind"] == "name":
        got = extract_last_name(heard)
        return names_match(got, item["name"]), f"name {got!r} vs {item['name']!r}"
    if item["kind"] == "order":
        got = extract_order_number(heard)
        return got == item["order"], f"order {got} vs {item['order']}"
    if item["kind"] == "catalog":
        text = normalize_transcript(heard).lower()
        hits = sum(w.lower() in text for w in item["keywords"])
        return hits == len(item["keywords"]), f"title words {hits}/{len(item['keywords'])}"
    ids, keywords = check(item, heard)
    return ids == len(item["ids"]) and keywords == len(item["keywords"]), f"ids {ids}/{len(item['ids'])}"


def run(models: list[str], conditions: list[float | None]) -> list[dict[str, Any]]:
    items = build_items()
    speaker = PiperSpeaker()
    voiced = [(item, speaker.speak(item["say"])) for item in items]
    rows = []
    for model in models:
        stt = WhisperTranscriber(model)
        stt.load()
        for snr in conditions:
            times, results, misses = [], {}, []
            for item, speech in voiced:
                audio = phone_line(speech.samples, speech.sample_rate, noise_snr_db=snr)
                heard = stt.transcribe(audio)
                times.append(heard.seconds)
                ok, detail = score(item, heard.text)
                results.setdefault(item["kind"], []).append(ok)
                if not ok:
                    misses.append(f"[{item['kind']}] {item['say']!r} -> {heard.text!r} ({detail})")
            rows.append({"model": model, "line": "phone" if snr is None else f"phone + noise {snr:g} dB",
                         **{k: f"{sum(v)}/{len(v)}" for k, v in results.items()},
                         "median_s": statistics.median(times), "misses": misses})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="small.en,base.en")
    parser.add_argument("--noise", type=float, default=20.0, help="SNR of the noisy run (dB)")
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()
    rows = run(args.models.split(","), [None, args.noise])
    print("\n| Whisper | Line | Last names matched | Order numbers | Catalog titles | Support sentences | "
          "Median per clip |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['line']} | {r['name']} | {r['order']} | {r['catalog']} | {r['support']} | "
              f"{r['median_s']:.2f} s |")
    if args.show_misses:
        for r in rows:
            for miss in r["misses"]:
                print(f"  [{r['model']}, {r['line']}] {miss}")


if __name__ == "__main__":
    main()
