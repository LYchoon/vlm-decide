import unittest
from types import SimpleNamespace
import torch
from torch import nn
from transformers import DynamicCache, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from vlm_decide.classifier import Classifier, validate_choices, resize_image
from PIL import Image


class ScoringTests(unittest.TestCase):
    def test_invalid_choices_and_resizing(self):
        for choices in ({"a": "one"}, {"a": "", "b": "two"}):
            with self.assertRaises(ValueError):
                validate_choices(choices)
        self.assertEqual(resize_image(Image.new("RGB", (800,400)), 224).size, (224,112))

    def test_real_tiny_decoder_cache_and_shared_prefixes(self):
        torch.manual_seed(3)
        config = Qwen3VLConfig(
            text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 2, "num_attention_heads": 4,
                         "num_key_value_heads": 2, "head_dim": 8,
                         "rope_scaling": {"rope_type": "default", "mrope_section": [1,1,2]}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 4, "out_hidden_size": 32, "deepstack_visual_indexes": [0]},
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        model = Qwen3VLForConditionalGeneration(config).eval()
        classifier = Classifier(model, SimpleNamespace())
        inputs = {"input_ids": torch.tensor([[4,5,6]]), "attention_mask": torch.ones(1,3,dtype=torch.long)}
        suffixes = [[1,9], [1,2,9], [2,9], [3]]
        with torch.inference_mode():
            full = classifier._full_scores(inputs, suffixes)
            cached = classifier._cached_scores(inputs, suffixes)
            self.assertTrue(torch.allclose(full, cached, atol=1e-5), (full, cached))
            reordered = classifier._cached_scores(inputs, list(reversed(suffixes)))
            self.assertTrue(torch.allclose(cached, reordered.flip(0), atol=1e-5))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
