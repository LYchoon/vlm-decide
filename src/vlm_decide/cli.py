import argparse
import json
from .classifier import from_pretrained, DEFAULT_MODEL


def main():
    p = argparse.ArgumentParser(description="Frozen VLM candidate classification")
    p.add_argument("image")
    p.add_argument("--question", required=True)
    p.add_argument("--choices", required=True, help="JSON mapping of labels to descriptions")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-edge", type=int, default=448)
    p.add_argument("--method", choices=["cached", "full"], default="cached")
    p.add_argument("--reference", action="append", default=[])
    a = p.parse_args()
    c = from_pretrained(a.model, device=a.device)
    print(json.dumps(c.classify(a.image, a.question, json.loads(a.choices), max_edge=a.max_edge,
                               method=a.method, reference_images=a.reference)))
