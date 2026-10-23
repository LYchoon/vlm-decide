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

### Multiple-choice VQA (2026-10-23)

We evaluate 100 [A-OKVQA](https://github.com/allenai/aokvqa) validation questions
with distinct COCO val2017 images, retaining the four original choices and their
order (seed `20261023`). The frozen model revision, FP16 precision and RTX 2080 Ti
are the same as above; the long-edge limit is 448 pixels. Each method has one
unmeasured warm-up and one measured run per question, with rotating method order.
The experimental implementation bounds cached candidate branches to eight.

| Method | Accuracy ↑ | p50 latency ↓ | Invalid outputs ↓ |
|---|---:|---:|---:|
| **Cached candidate scoring** | 76% (76/100) | 159.9 ms | 0/100 |
| Full candidate scoring | 76% (76/100) | 354.6 ms | 0/100 |
| Short-answer generation | 67% (67/100) | 156.0 ms | 12/100 |
| JSON generation, with fence removal | 80% (80/100) | 418.4 ms | 1/100 |

All 100 cached/full predictions agree; full scoring takes 2.22x as long at the
median. Cached/full provide every candidate's score and relative probability.
Invalid generated answers count as incorrect, including the 12 short-answer
format failures. The JSON table accepts removal of one enclosing Markdown fence;
strict JSON parsing gives 60% accuracy and 28/100 invalid outputs. Every method's
prompt explicitly pairs answer labels A/B/C/D with their descriptions. This
prototype's normalization is outside the library's strict parser. The four-point
accuracy difference between JSON and cached scoring on this small sample does
not establish a general ranking.

### Queried single-instance localization (2026-10-23)

The query names the target category, which has zero or one annotated instance.
We select 100 distinct [COCO val2017](https://cocodataset.org/#download) images:
ten categories, each with five positives and five negatives (seed `20261023`).
Positives have exactly one annotation of the queried class, excluding crowd
instances; negatives have none. Other categories may be visible. Annotation-based
absence can miss unlabeled objects. This protocol is **not standard COCO mAP**.

The coordinate-choice prototype first decides present/absent, then predicts
`x_min`, `y_min`, `x_max`, `y_max` sequentially on a 0–1000 grid with step 25.
Earlier predictions constrain later choices to valid boxes. A direct-generation
baseline instead emits the complete continuous box or null in one request, with
a 100-token budget. An additional measured pipeline uses cached presence scoring
as a gate, invoking that same direct box generator only when present is selected.
No ground-truth boxes, proposals, masks or answers enter inference prompts.

All runs use the same frozen model, FP16, 448-pixel limit, warm-up and single-run
timing protocol as VQA. There are 50 positive and 50 negative queries per method.

| Method | Positive localization at IoU ≥ 0.5 ↑ | Negative false positives ↓ | F1 at IoU ≥ 0.5 ↑ | Positive p50 ↓ | Negative p50 ↓ |
|---|---:|---:|---:|---:|---:|
| Cached coordinate scoring | 0/50 (0%) | 0/50 | 0.000 | 1.527 s | 0.137 s |
| Full coordinate scoring | 0/50 (0%) | 0/50 | 0.000 | 20.074 s | 0.160 s |
| Short-answer coordinates | 0/50 (0%) | 0/50 | 0.000 | 1.106 s | 0.133 s |
| JSON coordinates, with fence removal | 5/50 (10%) | 0/50 | 0.106 | 3.155 s | 0.376 s |
| Direct box generation | 42/50 (84%) | 50/50 | 0.560 | 1.374 s | 1.403 s |
| **Cached presence gate → direct box (exploratory)** | **39/50 (78%)** | **0/50** | **0.830** | 1.456 s | 0.134 s |

Missed positives count as localization failures. A misplaced positive box counts
as both a false positive and a false negative. Latency includes all decisions for
one query; positive/negative times are separated because absence exits early.
JSON labels and boxes accept one outer Markdown fence and must otherwise parse
and validate exactly. Strict-format failure counts are retained in the metrics.

Cached/full coordinate predictions agree on all 100 queries. Their presence
decisions are 94% accurate, but the coordinate-choice extension fails to localize
in this setup. Rounding the ground-truth boxes to the grid retains IoU ≥ 0.5 for
96% of positives, showing the grid can represent most boxes; this diagnostic is
not a formal upper bound on model performance.

Direct generation achieves mean positive IoU 0.718 but produces a box on every
negative query. The measured gated pipeline achieves mean positive IoU 0.671 and
F1 0.830, trading some positive recall for fewer false alarms. It was selected
after inspecting pilot behavior and tested on the same fixed sample, so it is
**exploratory, not an independent held-out confirmation**. The results support
using candidate scoring for the presence decision in this prototype; they do not
establish candidate-coordinate scoring as a general box detector.

Peak PyTorch allocated memory is 4.16 GiB for cached VQA, 5.49 GiB for cached
coordinate scoring and 4.09 GiB for the gated pipeline. These exclude CUDA
context, driver and desktop memory. There is one measured repetition per case;
shared server CPU/I/O activity can affect latency. Candidate probabilities remain
uncalibrated. VQA contributes 400 measured runs, coordinate/direct localization
500, and the gated pipeline 100: 1,000 runs total, with no inference exceptions in
the completed measurements.

[CV metrics, including strict JSON accuracy and validity](results/cv-20261023/metrics.json)
and the [sampling, decoding and evaluation protocol](results/cv-20261023/protocol.json)
record these results. The localization and normalization pipelines are experimental
benchmark prototypes; the installed library's public API remains the classifier
shown above.

## Roadmap

The next step is to add optional RLCD-style training for stronger confidence separation and better calibration on task-specific data while retaining the same candidate-scoring interface. Future evaluations will report calibration metrics alongside accuracy and throughput; until then, the current probabilities should be interpreted only relative to the supplied candidate set.

## References

- Yinheng Li and Justin Wagle, [LLM-as-Jev: LLMs Are Already Jev-Style Decision Models - When and How to Fine-Tune Them](https://arxiv.org/abs/2610.02076).
- [Qwen-2.5-1B-RLCD model card](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD), an implementation reference for candidate-set decision models.
- [How to Make Your Own Jev Model From an Open LLM](https://towardsdatascience.com/how-to-make-your-own-jev-model-from-an-open-llm/), background discussion of the Jev-style interface.
- [Qwen3-VL-2B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), the default VLM used here.

## License

MIT
