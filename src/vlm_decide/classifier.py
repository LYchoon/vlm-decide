"""Frozen multimodal candidate likelihoods; calls on an instance are sequential."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import json
import re
import time

import torch
from PIL import Image, ImageOps

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
PREFIX = "Best answer: ["


def validate_choices(choices):
    if not isinstance(choices, dict) or len(choices) < 2:
        raise ValueError("choices must map at least two distinct labels to descriptions")
    if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
           for k, v in choices.items()):
        raise ValueError("labels and descriptions must be nonempty strings")


def resize_image(image, max_edge):
    if max_edge < 32:
        raise ValueError("max_edge must be at least 32")
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            result = ImageOps.exif_transpose(source).convert("RGB")
    else:
        result = ImageOps.exif_transpose(image).convert("RGB")
    result.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return result


class Classifier:
    def __init__(self, model, processor):
        self.model = model.eval().requires_grad_(False)
        self.processor = processor
        self.device = next(model.parameters()).device
        if model.config.model_type not in {"qwen3_vl", "qwen2_5_vl"}:
            raise ValueError("Supported architectures: qwen3_vl, qwen2_5_vl")

    @classmethod
    def from_pretrained(cls, model_id=DEFAULT_MODEL, device="cuda:0", dtype="float16", **kwargs):
        from transformers import AutoConfig, AutoProcessor
        from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False, **kwargs)
        classes = {"qwen3_vl": Qwen3VLForConditionalGeneration,
                   "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration}
        if config.model_type not in classes:
            raise ValueError(f"Unsupported architecture: {config.model_type}")
        model = classes[config.model_type].from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa",
            device_map=device, trust_remote_code=False, **kwargs)
        return cls(model, AutoProcessor.from_pretrained(model_id, trust_remote_code=False, **kwargs))

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _prepare(self, image, question, choices=None, reference_images=None,
                 max_edge=448, output="choice"):
        started = time.perf_counter()
        images = [resize_image(x, max_edge) for x in (reference_images or [])]
        images.append(resize_image(image, max_edge))
        content = []
        if len(images) > 1:
            content.append({"type": "text", "text": "Reference images precede the final image. Judge only the final image."})
        content.extend({"type": "image"} for _ in images)
        text = question
        if choices:
            validate_choices(choices)
            options = "\n".join(f"[{i}] {v}" for i, v in enumerate(choices.values(), 1))
            text += "\nOptions:\n" + options
            text += ("\nReturn only a JSON object with a label from: " + json.dumps(list(choices))
                     if output == "json" else "\nSelect one option. Answer only with its bracketed numeric identifier.")
        content.append({"type": "text", "text": text})
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        if output == "choice":
            rendered += PREFIX
        factor = self.processor.image_processor.patch_size * self.processor.image_processor.merge_size
        inputs = self.processor(text=[rendered], images=images, return_tensors="pt",
                                min_pixels=factor * factor, max_pixels=max_edge * max_edge,
                                do_resize=True)
        inputs = inputs.to(self.device)
        self._sync()
        grid = inputs.get("image_grid_thw")
        grids = grid.tolist() if grid is not None else []
        meta = {"requested_max_edge": max_edge, "input_sizes": [list(x.size) for x in images],
                "processed_sizes": [[g[2] * self.processor.image_processor.patch_size,
                                     g[1] * self.processor.image_processor.patch_size] for g in grids],
                "visual_tokens": sum(g[0] * g[1] * g[2] // self.processor.image_processor.merge_size ** 2 for g in grids),
                "preprocess_seconds": time.perf_counter() - started}
        return inputs, meta

    def _suffixes(self, choices):
        return [self.processor.tokenizer.encode(f"{i}]", add_special_tokens=False)
                for i in range(1, len(choices) + 1)]

    def _full_scores(self, inputs, suffixes):
        scores = []
        for suffix in suffixes:
            ids = torch.tensor([suffix], device=self.device)
            extra = ids[:, :-1]
            extended = dict(inputs)
            extended["input_ids"] = torch.cat([inputs["input_ids"], extra], dim=1)
            extended["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(extra)], dim=1)
            output = self.model(**extended, use_cache=False, logits_to_keep=len(suffix))
            logp = output.logits.float().log_softmax(-1)
            scores.append(logp.gather(-1, ids.unsqueeze(-1)).sum())
        return torch.stack(scores)

    def _cached_scores(self, inputs, suffixes):
        output = self.model(**inputs, use_cache=True, logits_to_keep=1)
        first = output.logits[:, -1].float().log_softmax(-1)[0]
        scores = torch.stack([first[s[0]] for s in suffixes])
        groups = defaultdict(list)
        for i, suffix in enumerate(suffixes):
            if len(suffix) > 1:
                groups[len(suffix)].append(i)
        n = inputs["input_ids"].shape[1]
        for length, indices in groups.items():
            cache = deepcopy(output.past_key_values)
            cache.batch_repeat_interleave(len(indices))
            ids = torch.tensor([suffixes[i] for i in indices], device=self.device)
            branch = self.model(input_ids=ids[:, :-1], past_key_values=cache,
                                attention_mask=torch.ones((len(indices), n + length - 1),
                                                          device=self.device, dtype=inputs["attention_mask"].dtype),
                                cache_position=torch.arange(n, n + length - 1, device=self.device),
                                use_cache=True, logits_to_keep=length - 1)
            logp = branch.logits.float().log_softmax(-1)
            scores[indices] += logp.gather(-1, ids[:, 1:].unsqueeze(-1)).sum((1, 2))
        return scores

    @torch.inference_mode()
    def classify(self, image, question, choices, *, reference_images=None, max_edge=448, method="cached"):
        validate_choices(choices)
        if method not in {"cached", "full"}:
            raise ValueError("method must be cached or full")
        started = time.perf_counter()
        inputs, meta = self._prepare(image, question, choices, reference_images, max_edge)
        suffixes = self._suffixes(choices)
        self._sync()
        model_started = time.perf_counter()
        scores = (self._cached_scores if method == "cached" else self._full_scores)(inputs, suffixes)
        if not torch.isfinite(scores).all():
            raise RuntimeError("Non-finite candidate scores")
        probabilities = scores.softmax(0)
        self._sync()
        elapsed = time.perf_counter() - model_started
        labels = list(choices)
        values = probabilities.tolist()
        ordered = sorted(values, reverse=True)
        return {"label": labels[probabilities.argmax().item()],
                "probabilities": dict(zip(labels, values)), "log_scores": dict(zip(labels, scores.tolist())),
                "margin": ordered[0] - ordered[1], "calibrated": False,
                "mode": "reference_conditioned" if reference_images else "zero_shot",
                "model_seconds": elapsed, "total_seconds": time.perf_counter() - started, **meta}

    def classify_many(self, images, question, choices, **kwargs):
        # ponytail: sequential images bound memory; batch when measured throughput requires it.
        for image in images:
            yield self.classify(image, question, choices, **kwargs)

    @torch.inference_mode()
    def generate_baseline(self, image, question, choices, *, max_edge=448, output="choice"):
        if output not in {"choice", "json"}:
            raise ValueError("output must be choice or json")
        started = time.perf_counter()
        inputs, meta = self._prepare(image, question, choices, max_edge=max_edge, output=output)
        self._sync()
        model_started = time.perf_counter()
        sequence = self.model.generate(**inputs, do_sample=False, max_new_tokens=64,
                                       use_cache=True, pad_token_id=self.processor.tokenizer.eos_token_id)
        tokens = sequence[:, inputs["input_ids"].shape[1]:]
        self._sync()
        elapsed = time.perf_counter() - model_started
        text = self.processor.tokenizer.decode(tokens[0], skip_special_tokens=True)
        label = None
        if output == "choice":
            match = re.match(r"\s*(\d+)\]", text)
            if match and 1 <= int(match[1]) <= len(choices):
                label = list(choices)[int(match[1]) - 1]
        else:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and parsed.get("label") in choices:
                    label = parsed["label"]
            except (ValueError, TypeError):
                pass
        return {"label": label, "text": text, "valid_output": label is not None,
                "generated_tokens": tokens.shape[1], "model_seconds": elapsed,
                "total_seconds": time.perf_counter() - started, **meta}

    @torch.inference_mode()
    def caption(self, image, *, max_edge=672, prompt="Describe only visible contents in one concise English sentence."):
        inputs, _ = self._prepare(image, prompt, max_edge=max_edge, output="text")
        sequence = self.model.generate(**inputs, do_sample=False, max_new_tokens=100,
                                       pad_token_id=self.processor.tokenizer.eos_token_id)
        return self.processor.tokenizer.decode(sequence[0, inputs["input_ids"].shape[1]:],
                                               skip_special_tokens=True).strip()


from_pretrained = Classifier.from_pretrained
