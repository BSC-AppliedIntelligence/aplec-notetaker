#!/usr/bin/env python3
"""Nemotron 3 Diarization on one recording, as a subprocess of the worker.

It has its own environment (scripts/setup_nemotron.sh) because the model needs
transformers 5. The worker pipes the decoded recording in, 16 kHz mono float32,
and reads the turns back:

    diarize.py MODEL_DIR --threshold 0.35 < audio.f32le
    -> {"turns": [[start, end, "SPEAKER_00"], ...], "seconds": 7.2}
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

SAMPLE_RATE = 16000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="directory with the weights")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    audio = np.frombuffer(sys.stdin.buffer.read(), dtype=np.float32).copy()
    if audio.size == 0:
        raise SystemExit("no audio on stdin")

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        args.model, device_map="auto", dtype=torch.float32
    ).eval()
    inputs = processor(audio, sampling_rate=SAMPLE_RATE).to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        logits = model(**inputs).logits
    segments = processor.extract_speaker_dict(logits, inputs.attention_mask, threshold=args.threshold)[0]
    turns = [
        [float(s["Start"]), float(s["End"]), f"SPEAKER_{int(s['Speaker']):02d}"]
        for s in segments
    ]
    print(json.dumps({"turns": turns, "seconds": round(time.perf_counter() - started, 1)}))


if __name__ == "__main__":
    main()
