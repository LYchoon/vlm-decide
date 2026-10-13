# vlm-decide

`vlm-decide` turns a compatible Hugging Face vision-language model into a lightweight, training-free image classifier. It keeps every model weight frozen, asks the VLM to choose among text-defined candidates, scores the complete bracketed candidate suffix, and returns a relative probability distribution over the supplied choices.

## Why candidate scoring

A normal VLM classifier often generates an answer such as JSON and then parses it. That spends decoding steps on syntax and can produce invalid output. `vlm-decide` instead:

1. maps the labels to numeric identifiers such as `[1]` and `[2]`;
2. computes the log-probability of each complete identifier suffix, including the closing bracket;
3. reuses one visual/text prefill and its KV cache across candidates; and
4. applies a softmax only across the supplied candidates.

This is zero-shot classification when classes are described with text. Supplying reference images enables reference-conditioned classification without updating the VLM. Returned probabilities are relative to the candidate set and are not calibrated confidence estimates.

The current implementation supports Qwen3-VL and Qwen2.5-VL architectures. The default model is `Qwen/Qwen3-VL-2B-Instruct`.

## Installation

Python 3.11 and a CUDA GPU are recommended.

```bash
pip install "git+https://github.com/LYchoon/vlm-decide.git"
```

## Python usage

```python
from vlm_decide import from_pretrained

classifier = from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    device="cuda:0",
    dtype="float16",
)

choices = {
    "present": "The described subject is visible in the image.",
    "absent": "The described subject is not visible in the image.",
}

result = classifier.classify(
    "frame.jpg",
    "Is the described subject visible?",
    choices,
    max_edge=224,
    method="cached",
)

print(result["label"])
print(result["probabilities"])
print(result["total_seconds"])
```

For reference-conditioned classification:

```python
result = classifier.classify(
    "frame.jpg",
    "Does the final image show the same subject as the references?",
    choices,
    reference_images=["reference-1.jpg", "reference-2.jpg"],
)
```

`classify_many` accepts an iterable of images and yields one result at a time. Calls on one classifier instance are sequential to keep GPU memory bounded.

## Command-line usage

```bash
vlm-decide frame.jpg \
  --question "Is the described subject visible?" \
  --choices '{"present":"The subject is visible.","absent":"The subject is absent."}' \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --device cuda:0 \
  --max-edge 224 \
  --method cached
```

Use `--reference reference.jpg` one or more times to add reference images. Use `--method full` as a correctness baseline that recomputes every candidate without the shared KV cache.

## Measured results

The following results use `Qwen/Qwen3-VL-2B-Instruct` revision `89644892e4d85e24eaac8bacfd4f463576704203`, FP16, batch size 1, a long-edge limit of 224 pixels, and one RTX 2080 Ti. The human-verified binary cohort contains 100 images: 40 development images and 60 held-out test images. Latency and throughput use 300 runs, corresponding to three repetitions of the same 100 images.

| Method | What it does | p50 latency | Images/s | Held-out accuracy | Invalid outputs |
|---|---|---:|---:|---:|---:|
| Full candidate scoring | Scores every complete candidate independently | 146.4 ms | 6.53 | 93.3% | 0 / 300 |
| Cached candidate scoring | Reuses one prefill and branches the KV cache | 127.0 ms | 7.50 | 93.3% | 0 / 300 |
| Short-answer generation | Generates only the bracketed choice | 122.7 ms | 7.82 | 93.3% | 0 / 300 |
| JSON generation | Generates and parses a JSON label | 352.2 ms | 2.70 | 86.7% | 15 / 300 |

Cached scoring reduced median latency by 13.2% and increased throughput by 14.9% relative to full candidate scoring while preserving every prediction. It was 2.77x faster than JSON generation and avoided invalid structured outputs. Short-answer generation was slightly faster in this two-class test, while cached scoring additionally returns deterministic scores and a probability distribution over every candidate. Across 900 cached-versus-full parity checks at 224, 448, and 672 pixels, the largest probability difference was 0.00397 and no predicted label changed.

These measurements describe this model, hardware, prompt, and image cohort. They are not a general accuracy claim, and the candidate probabilities are not calibrated.

## References

- Yinheng Li and Justin Wagle, [LLM-as-Jev: LLMs Are Already Jev-Style Decision Models - When and How to Fine-Tune Them](https://arxiv.org/abs/2610.02076).
- [Qwen-2.5-1B-RLCD model card](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD), an implementation reference for candidate-set decision models.
- [How to Make Your Own Jev Model From an Open LLM](https://towardsdatascience.com/how-to-make-your-own-jev-model-from-an-open-llm/), background discussion of the Jev-style interface.
- [Qwen3-VL-2B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), the default VLM used here.

## License

MIT
