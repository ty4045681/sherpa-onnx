#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
"""Print the keyword context graph and live search state of one wav file.

Usage:

  python3 keyword-spotter-state.py \
    --tokens tokens.txt \
    --encoder encoder.onnx \
    --decoder decoder.onnx \
    --joiner joiner.onnx \
    --keywords-file keywords.txt \
    --wav test.wav
"""

import argparse
import json

import numpy as np
import sherpa_onnx
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--joiner", required=True)
    parser.add_argument("--keywords-file", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--tail-padding", type=float, default=0.66)
    return parser.parse_args()


def main():
    args = get_args()
    spotter = sherpa_onnx.KeywordSpotter(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        keywords_file=args.keywords_file,
        num_threads=args.num_threads,
    )
    samples, sample_rate = sf.read(args.wav, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples[:, 0]
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.concatenate(
        [samples, np.zeros(int(args.tail_padding * sample_rate), dtype=np.float32)]
    )

    stream = spotter.create_stream()
    stream.accept_waveform(sample_rate, samples)
    stream.input_finished()

    hits = []
    while spotter.is_ready(stream):
        spotter.decode_stream(stream)
        result = spotter.get_result(stream)
        if result:
            hits.append(result)
            spotter.reset_stream(stream)

    graph = spotter.get_context_graph(stream)
    state = spotter.get_search_state(stream)
    print("keyword hits:", hits)
    print("graph nodes:", len(graph["nodes"]), "root:", graph["root"])
    print(
        "frames:",
        len(state["frames"]),
        "offset:",
        state["frame_offset"],
        "best level:",
        state["best"]["level"],
    )
    print(
        json.dumps(
            {"graph": graph, "state": state},
            ensure_ascii=False,
        )[:2000]
    )


if __name__ == "__main__":
    main()
