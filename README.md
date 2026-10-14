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

### Imagenette validation sample (2026-10-14)

This comparison uses the frozen `Qwen/Qwen3-VL-2B-Instruct` model at revision
`89644892e4d85e24eaac8bacfd4f463576704203`, FP16, batch size 1, a long-edge
limit of 224 pixels, and one RTX 2080 Ti. We sample 20 clean validation images
from each of the ten [Imagenette v2 160 px](https://github.com/fastai/imagenette)
classes without replacement (200 images total, seed `20261013`). Every image
has the same ten candidate descriptions. No training or reference images are used.

Each method is measured once per image (800 runs total), after three warm-up
calls per method. Method order rotates by image. Latency includes preprocessing,
inference, decoding and answer parsing, and excludes loading, warm-up and result
logging. Images/s is the inverse of mean measured latency, not batched throughput.

| Method | Top-1 accuracy ↑ | p50 latency ↓ | p95 latency ↓ | Images/s ↑ | Invalid outputs ↓ |
|---|---:|---:|---:|---:|---:|
| **Cached candidate scoring (ours)** | **99.0% (198/200)** | 195.0 ms | 230.1 ms | 5.06 | **0/200** |
| Full candidate scoring | **99.0% (198/200)** | 790.1 ms | 853.3 ms | 1.25 | **0/200** |
| Short-answer generation | **99.0% (198/200)** | **138.9 ms** | **180.9 ms** | **7.04** | **0/200** |
| JSON generation, with fence removal | 92.0% (184/200) | 635.7 ms | 808.5 ms | 1.40 | 15/200 |

Full scoring takes 4.05x as long as cached scoring at the median; cached scoring
reduces median latency by 75.3% and achieves 4.03x the throughput. All 200
cached/full labels agree; the largest candidate
probability difference is `0.00006598`. Short-answer generation has the same
accuracy and 28.7% lower median latency than cached scoring. Cached/full return
every candidate's score and relative probability; short-answer generation returns
only a generated label. The JSON baseline with fence removal takes 3.26x as long
as cached scoring at the median.

JSON uses readable class labels such as `tench` and `French horn`, rather than
requiring WNID identifiers. The table accepts removal of one enclosing Markdown
JSON fence before parsing; this normalization is implemented in the experiment
runner, not in the library. Under the library's original strict `json.loads`
parser, all 200 JSON outputs are invalid and accuracy is 0.0%. Fence removal
recovers 185 valid labels, of which 184 are correct. Five outputs reach the
64-token generation limit. Invalid outputs count as incorrect. All four methods
complete without inference errors.

These are results on a balanced 200-image sample, not the full 3,925-image
Imagenette validation set or ImageNet-1K. This is a comparison of output strategies
on one pretrained model, not a from-scratch Imagenette leaderboard submission.
There is one measured repetition per image; other hardware, resolutions and
candidate counts require separate measurements. Other GPUs on the server have
unrelated jobs, so shared CPU/I/O interference remains possible. Candidate
probabilities are relative to these ten choices and are not calibrated.

[Metrics and per-class accuracy](results/imagenette-20261014/summary.json),
[all 800 raw predictions](results/imagenette-20261014/predictions.jsonl),
[protocol and software versions](results/imagenette-20261014/config.json),
[dataset provenance](results/imagenette-20261014/dataset.json), and
[sample manifest with image checksums](results/imagenette-20261014/manifest.json)
are recorded alongside the [reproducible runner](experiments/imagenette.py).
Infrastructure-specific host details and images are omitted from the published data.

To reproduce on Linux with Python 3.11 and CUDA:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
hf download Qwen/Qwen3-VL-2B-Instruct --revision 89644892e4d85e24eaac8bacfd4f463576704203
curl -fL -o imagenette2-160.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
python experiments/imagenette.py prepare --archive imagenette2-160.tgz --data data/imagenette200 --per-class 20 --seed 20261013
python experiments/imagenette.py self-check
python experiments/imagenette.py run --data data/imagenette200 --output results/rerun --device cuda:0 --dtype float16
```

### Earlier human-verified binary reference

The following results use `Qwen/Qwen3-VL-2B-Instruct` revision `89644892e4d85e24eaac8bacfd4f463576704203`, FP16, batch size 1, a long-edge limit of 224 pixels, and one RTX 2080 Ti. The human-verified binary cohort contains 100 images: 40 development images and 60 held-out test images. Latency and throughput use 300 runs, corresponding to three repetitions of the same 100 images.

| Method | What it does | p50 latency ↓ | Images/s ↑ | Held-out accuracy ↑ | Invalid outputs ↓ |
|---|---|---:|---:|---:|---:|
| Full candidate scoring | Scores every complete candidate independently | 146.4 ms | 6.53 | **93.3%** | **0 / 300** |
| Short-answer generation | Generates only the bracketed choice | **122.7 ms** | **7.82** | **93.3%** | **0 / 300** |
| JSON generation | Generates and parses a JSON label | 352.2 ms | 2.70 | 86.7% | 15 / 300 |
| **Cached candidate scoring (ours)** | Reuses one prefill and branches the KV cache | 127.0 ms | 7.50 | **93.3%** | **0 / 300** |

Cached scoring reduced median latency by 13.2% and increased throughput by 14.9% relative to full candidate scoring while preserving every prediction. It was 2.77x faster than JSON generation and avoided invalid structured outputs. Short-answer generation was 3.4% faster in median latency and had 4.3% higher throughput in this two-class test, making it the fastest option when only a generated label is needed. Cached scoring is intended for workflows that also need every candidate score and the resulting relative probability distribution, such as threshold-based filtering, ranking, and routing uncertain cases for review. Across 900 cached-versus-full parity checks at 224, 448, and 672 pixels, the largest probability difference was 0.00397 and no predicted label changed.

These measurements describe this model, hardware, prompt, and image cohort. They are not a general accuracy claim, and the candidate probabilities are not calibrated.

## Roadmap

The next step is to add optional RLCD-style training for stronger confidence separation and better calibration on task-specific data while retaining the same candidate-scoring interface. Future evaluations will report calibration metrics alongside accuracy and throughput; until then, the current probabilities should be interpreted only relative to the supplied candidate set.

## References

- Yinheng Li and Justin Wagle, [LLM-as-Jev: LLMs Are Already Jev-Style Decision Models - When and How to Fine-Tune Them](https://arxiv.org/abs/2610.02076).
- [Qwen-2.5-1B-RLCD model card](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD), an implementation reference for candidate-set decision models.
- [How to Make Your Own Jev Model From an Open LLM](https://towardsdatascience.com/how-to-make-your-own-jev-model-from-an-open-llm/), background discussion of the Jev-style interface.
- [Qwen3-VL-2B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), the default VLM used here.

## License

MIT
