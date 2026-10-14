"""Four-method Imagenette evaluation using public vlm-decide with semantic JSON labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import tarfile
import time

CLASSES = {
    "n01440764": "tench, a freshwater fish",
    "n02102040": "English springer spaniel, a dog breed",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church building",
    "n03394916": "French horn, a brass musical instrument",
    "n03417042": "garbage truck",
    "n03425413": "gas pump, a fuel dispenser",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
QUESTION = "What is the main object in the image? Choose the best matching category."
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


LABEL_NAMES = dict(zip(CLASSES, ["tench", "English springer spaniel", "cassette player", "chain saw",
                               "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]))


def parse_json_answer(text, choices):
    # Accept a single Markdown JSON fence, retaining strict-format validity separately.
    fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, re.DOTALL)
    try:
        parsed = json.loads(fenced.group(1) if fenced else text)
    except (ValueError, TypeError):
        return None
    return parsed.get("label") if isinstance(parsed, dict) and parsed.get("label") in choices else None


def self_check():
    choices = {"tench": "fish", "church": "building"}
    assert parse_json_answer('{"label":"tench"}', choices) == "tench"
    assert parse_json_answer('```json\n{"label":"church"}\n```', choices) == "church"
    for text in ['{"label":"unknown"}', '[]', '{', 'text {"label":"tench"}', '```json\n{}\n```']:
        assert parse_json_answer(text, choices) is None
    print("JSON parser self-check passed")


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(archive, destination, per_class, seed):
    from PIL import Image
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    with tarfile.open(archive) as source:
        by_class = defaultdict(list)
        for member in source.getmembers():
            parts = Path(member.name).parts
            if (member.isfile() and len(parts) == 4 and parts[:2] == ("imagenette2-160", "val")
                    and parts[2] in CLASSES and parts[3].lower().endswith((".jpeg", ".jpg"))):
                by_class[parts[2]].append(member)
        if set(by_class) != set(CLASSES):
            raise ValueError("Archive does not contain the ten expected clean validation classes")
        for label in CLASSES:
            available = sorted(by_class[label], key=lambda member: member.name)
            if len(available) < per_class:
                raise ValueError(f"Not enough validation images for {label}")
            for member in rng.sample(available, per_class):
                filename = f"{label}/{Path(member.name).name}"
                target = destination / "images" / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.extractfile(member).read())
                with Image.open(target) as image:
                    image.verify()
                rows.append({"image": f"images/{filename}", "label": label,
                             "archive_member": member.name, "sha256": sha256(target)})
    rng.shuffle(rows)
    dump(destination / "manifest.json", rows)
    dump(destination / "choices.json", CLASSES)
    dump(destination / "dataset.json", {
        "dataset": "Imagenette v2 160 px", "source_url": URL,
        "source_sha256": sha256(archive), "split": "val", "clean_labels": True,
        "validation_counts": {label: len(by_class[label]) for label in CLASSES},
        "sample_counts": dict(Counter(row["label"] for row in rows)),
        "seed": seed, "samples": len(rows), "candidate_classes": len(CLASSES),
        "sampling": "Uniform without replacement within each class; fixed WNID order; shuffled rows",
        "note": "Zero-shot frozen pretrained VLM; not comparable to ImageNet-1k or from-scratch leaderboards",
    })
    print(f"Prepared {len(rows)} images from ten validation classes", flush=True)


def summarize(rows, config):
    methods = {}
    for method in config["methods"]:
        selected = [row for row in rows if row["method"] == method]
        successful = [row for row in selected if "error" not in row]
        latency = sorted(row["total_seconds"] for row in successful)
        classes = {}
        for label in config["choice_order"]:
            group = [row for row in selected if row["truth"] == label]
            classes[label] = {"n": len(group), "correct": sum(row.get("label") == label for row in group)}
        methods[method] = {
            "n": len(selected), "errors": len(selected) - len(successful),
            "top1_accuracy": sum(row.get("label") == row["truth"] for row in selected) / len(selected) if selected else None,
            "invalid_outputs": sum(not row.get("valid_output", False) for row in selected),
            "strict_invalid_outputs": sum(not row.get("strict_valid_output", row.get("valid_output", False)) for row in selected),
            "strict_top1_accuracy": sum(row.get("strict_label", row.get("label")) == row["truth"] for row in selected) / len(selected) if selected else None,
            "p50_seconds": statistics.median(latency) if latency else None,
            "p95_seconds": latency[max(0, int(0.95 * len(latency) + 0.999999) - 1)] if latency else None,
            "images_per_second": len(latency) / sum(latency) if latency else None,
            "cuda_peak_allocated_gib": max((row.get("cuda_peak_allocated_bytes", 0) / 2**30 for row in successful), default=0) or None,
            "cuda_peak_reserved_gib": max((row.get("cuda_peak_reserved_bytes", 0) / 2**30 for row in successful), default=0) or None,
            "per_class": classes,
        }
    pairs = defaultdict(dict)
    for row in rows:
        if row["method"] in {"cached", "full"} and "probabilities" in row:
            pairs[(row["image"], row["repetition"])][row["method"]] = row
    paired = [pair for pair in pairs.values() if set(pair) == {"cached", "full"}]
    parity = {
        "paired_runs": len(paired),
        "label_disagreements": sum(pair["cached"]["label"] != pair["full"]["label"] for pair in paired),
        "max_probability_difference": max((abs(pair["cached"]["probabilities"][label] - pair["full"]["probabilities"][label])
                                           for pair in paired for label in config["choice_order"]), default=None),
    }
    planned = config["images"] * config["repetitions"] * len(config["methods"])
    return {"complete": len(rows) == planned, "completed_runs": len(rows), "planned_runs": planned,
            "methods": methods, "cached_full_parity": parity}


def run(data, output, args):
    import torch
    import transformers
    from vlm_decide import from_pretrained
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    data = data.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if args.limit:
        manifest = manifest[:args.limit]
    original_choices = json.loads((data / "choices.json").read_text(encoding="utf-8"))
    choices = {LABEL_NAMES[label]: description for label, description in original_choices.items()}
    for row in manifest:
        row["wnid"] = row["label"]
        row["label"] = LABEL_NAMES[row["label"]]
    for row in manifest:
        if row["label"] not in choices or sha256(data / row["image"]) != row["sha256"]:
            raise ValueError("Manifest label or image checksum mismatch")
    config = {"model": args.model, "revision": REVISION, "device": args.device, "dtype": args.dtype,
              "threads": args.threads, "methods": args.methods, "images": len(manifest),
              "repetitions": args.repetitions, "max_edge": args.max_edge,
              "candidate_batch_size": len(choices), "implementation": "public repository unchanged",
              "protocol_version": "semantic-labels-fenced-json-v1", "warmup_per_method": 3,
              "source_commit": subprocess.check_output(["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"], text=True).strip(),
              "manifest_sha256": sha256(data / "manifest.json"), "question": QUESTION,
              "choice_order": list(choices), "class_descriptions": choices,
              "classifier_sha256": sha256(Path(__file__).parents[1] / "src/vlm_decide/classifier.py"),
              "runner_sha256": sha256(Path(__file__)), "torch": torch.__version__,
              "transformers": transformers.__version__, "hostname": platform.node()}
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing output has a different protocol; choose a new output directory")
    dump(config_path, config)
    dump(output / "dataset.json", json.loads((data / "dataset.json").read_text()))
    hardware = {"platform": platform.platform(), "python": platform.python_version(),
                "cpu": subprocess.check_output(["lscpu"], text=True), "cuda": torch.version.cuda,
                "device": args.device, "torch_threads": torch.get_num_threads()}
    if args.device.startswith("cuda"):
        hardware["gpu"] = torch.cuda.get_device_name(args.device)
        hardware["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True)
    dump(output / "hardware.json", hardware)
    raw = output / "predictions.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    done = {(row["image"], row["repetition"], row["method"]) for row in rows}
    if len(done) != len(rows):
        raise ValueError("Duplicate recorded inference runs")
    print(f"Loading {args.model} on {args.device}; {len(rows)} completed runs", flush=True)
    classifier = from_pretrained(args.model, device=args.device, dtype=args.dtype,
                                 revision=REVISION, local_files_only=True)

    def infer(image, method):
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(classifier.device)
        started = time.perf_counter()
        if method in {"cached", "full"}:
            result = classifier.classify(image, QUESTION, choices, max_edge=args.max_edge,
                                         method=method)
            result["valid_output"] = result["label"] in choices
        else:
            result = classifier.generate_baseline(image, QUESTION, choices, max_edge=args.max_edge, output=method)
        if method == "json":
            result["strict_label"] = result["label"]
            result["strict_valid_output"] = result["valid_output"]
            result["label"] = parse_json_answer(result["text"], choices)
            result["valid_output"] = result["label"] is not None
        result["library_total_seconds"] = result["total_seconds"]
        result["total_seconds"] = time.perf_counter() - started
        if args.device.startswith("cuda"):
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(classifier.device)
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(classifier.device)
        return result

    warmup = {}
    for method in args.methods:
        for _ in range(3):
            result = infer(data / manifest[0]["image"], method)
        warmup[method] = {"seconds": result["total_seconds"], "label": result["label"]}
    dump(output / "warmup.json", warmup)
    print("Warm-up excluded from measurements: " + json.dumps(warmup), flush=True)
    with raw.open("a", encoding="utf-8", buffering=1) as stream:
        for repetition in range(args.repetitions):
            for index, sample in enumerate(manifest):
                offset = (index + repetition) % len(args.methods)
                order = args.methods[offset:] + args.methods[:offset]
                for method in order:
                    key = (sample["image"], repetition, method)
                    if key in done:
                        continue
                    started = time.perf_counter()
                    try:
                        result = infer(data / sample["image"], method)
                    except Exception as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}", "label": None,
                                  "valid_output": False, "failed_seconds": time.perf_counter() - started}
                        if args.device.startswith("cuda"):
                            torch.cuda.empty_cache()
                    row = {"image": sample["image"], "truth": sample["label"],
                           "wnid": sample["wnid"], "repetition": repetition, "method": method, **result}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    done.add(key)
                    dump(output / "summary.json", summarize(rows, config))
                    print(f"{len(rows)}/{config['images'] * args.repetitions * len(args.methods)} "
                          f"{method}: {result.get('label')} truth={sample['label']} "
                          f"seconds={result.get('total_seconds', result.get('failed_seconds')):.3f}", flush=True)
    print(json.dumps(summarize(rows, config), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    prep = sub.add_parser("prepare")
    prep.add_argument("--archive", type=Path, required=True)
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--per-class", type=int, default=20)
    prep.add_argument("--seed", type=int, default=20261013)
    evaluation = sub.add_parser("run")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    evaluation.add_argument("--threads", type=int, default=4)
    evaluation.add_argument("--max-edge", type=int, default=224)
    evaluation.add_argument("--candidate-batch-size", type=int, default=10)
    evaluation.add_argument("--methods", choices=["cached", "full", "choice", "json"],
                            nargs="+", default=["cached", "full", "choice", "json"])
    evaluation.add_argument("--repetitions", type=int, default=1)
    evaluation.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.per_class < 1:
            parser.error("per-class must be positive")
        prepare(args.archive, args.data, args.per_class, args.seed)
    elif args.command == "self-check":
        self_check()
    else:
        if (args.repetitions < 1 or args.threads < 1 or args.candidate_batch_size < 1
                or (args.limit is not None and args.limit < 1) or len(set(args.methods)) != len(args.methods)):
            parser.error("Counts must be positive and methods must be unique")
        run(args.data, args.output, args)


if __name__ == "__main__":
    main()
