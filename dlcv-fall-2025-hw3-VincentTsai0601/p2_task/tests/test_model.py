import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from decoder import Config, Decoder

try:
    from caption_model import (CaptionConfig, CaptionModel, caption_loss,
                               create_vision, load_adapter, read_checkpoint, save_adapter)
except ModuleNotFoundError as error:
    if error.name != 'caption_model':
        raise
    CaptionModel = None


def small_decoder():
    cfg = Config()
    cfg.hidden_size = 8
    cfg.intermediate_size = 16
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 1
    cfg.head_dim = 4
    cfg.max_position_embeddings = 64
    # The supplied Decoder has a fixed padding_idx of 151643.
    cfg.vocab_size = 151646
    return Decoder(cfg)


class TinyVision(nn.Module):
    num_features = 4
    num_prefix_tokens = 1

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 4, kernel_size=2, stride=2)

    def forward_features(self, images):
        patches = self.patch(images).flatten(2).transpose(1, 2)
        return torch.cat([patches.mean(1, keepdim=True), patches], dim=1)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(CaptionModel, 'Caption model has not been implemented')
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.decoder = small_decoder()
        self.original = copy.deepcopy(self.decoder).eval()
        self.vision = TinyVision()
        self.config = CaptionConfig(projector_hidden=8, lora_rank=2, lora_alpha=4)
        self.model = CaptionModel(self.decoder, self.vision, self.config)

    def test_lora_and_sdpa_preserve_pretrained_logits(self):
        self.model.eval()
        ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            expected = self.original(input_ids=ids)
            actual = self.model.decoder(input_ids=ids)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_caption_loss_aligns_first_word_eos_and_ignores_visual_padding(self):
        # visual tokens=2; text=BOS,word,EOS,PAD; supervised targets are word/EOS.
        logits = torch.zeros(1, 6, 4)
        logits[0, 2, 1] = 2.0
        logits[0, 3, 2] = 3.0
        ids = torch.tensor([[0, 1, 2, 3]])
        total, count = caption_loss(logits, ids, visual_tokens=2, pad_id=3)
        expected = torch.nn.functional.cross_entropy(
            torch.tensor([[0., 2., 0., 0.], [0., 0., 3., 0.]]),
            torch.tensor([1, 2]), reduction='sum')
        self.assertEqual(count, 2)
        torch.testing.assert_close(total, expected)
        logits[:, :2, :] = 1000 * torch.randn(1, 2, 4)
        logits[:, 4:, :] = 1000 * torch.randn(1, 2, 4)
        changed, _ = caption_loss(logits, ids, 2, 3)
        torch.testing.assert_close(changed, total)

    def test_only_projector_and_lora_receive_gradients(self):
        self.model.train()
        self.assertFalse(self.vision.training)
        ids = torch.tensor([[151644, 1, 2, 151645], [151644, 2, 151645, 151643]])
        logits, visual_tokens = self.model(torch.randn(2, 3, 4, 4), ids)
        total, count = caption_loss(logits, ids, visual_tokens, 151643)
        (total / count).backward()
        trainable = {name for name, p in self.model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith('projector.') or 'lora_' in name for name in trainable))
        self.assertTrue(all(p.grad is None for name, p in self.model.named_parameters()
                            if name not in trainable))
        self.assertGreater(self.model.projector[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for name, p in self.model.named_parameters() if 'lora_B' in name))

    def test_padding_does_not_change_valid_logits(self):
        self.model.eval()
        images = torch.randn(1, 3, 4, 4)
        with torch.no_grad():
            short, _ = self.model(images, torch.tensor([[151644, 1, 151645]]))
            padded, _ = self.model(images, torch.tensor([[151644, 1, 151645, 151643]]))
        torch.testing.assert_close(short, padded[:, :-1], atol=3e-6, rtol=2e-5)

    def test_cached_generation_matches_full_sequence_recomputation(self):
        self.model.eval()
        images = torch.randn(2, 3, 4, 4)
        with torch.no_grad():
            visual = self.model.encode_images(images)
            ids = torch.full((2, 1), 151644, dtype=torch.long)
            expected = []
            finished = torch.zeros(2, dtype=torch.bool)
            for _ in range(4):
                embeds = torch.cat([visual, self.decoder.embed_tokens(ids)], dim=1)
                logits = self.decoder(inputs_embeds=embeds)[:, -1, :]
                logits[:, [151643, 151644]] = -torch.inf
                token = logits.argmax(-1).masked_fill(finished, 151645)
                expected.append(token)
                ids = torch.cat([ids, token[:, None]], 1)
                finished |= token.eq(151645)
                if finished.all():
                    break
            actual = self.model.generate(images, 151644, 151645, max_new_tokens=4)
        self.assertTrue(torch.equal(actual, torch.stack(expected, dim=1)))
        # A second batch must not inherit the first batch's cache.
        again = self.model.generate(images, 151644, 151645, max_new_tokens=4)
        self.assertTrue(torch.equal(actual, again))

    def test_checkpoint_contains_only_trainable_weights_and_roundtrips(self):
        self.model.eval()
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if 'lora_B' in name:
                    p.normal_(std=0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.pt'
            save_adapter(self.model, path, epoch=1, val_loss=2.0)
            payload = read_checkpoint(path)
            self.assertEqual(set(payload['adapter']),
                             {name for name, p in self.model.named_parameters() if p.requires_grad})
            self.assertFalse(any(name.startswith('vision.') for name in payload['adapter']))
            clone = CaptionModel(copy.deepcopy(self.original), TinyVision(), self.config)
            load_adapter(clone, payload)
            for name, p in clone.named_parameters():
                if p.requires_grad:
                    torch.testing.assert_close(p, payload['adapter'][name])
            payload['adapter'].pop(next(iter(payload['adapter'])))
            with self.assertRaisesRegex(ValueError, 'keys'):
                load_adapter(clone, payload)

    def test_parameter_budget_is_enforced(self):
        with self.assertRaisesRegex(ValueError, '10,000,000'):
            CaptionModel(small_decoder(), TinyVision(),
                         CaptionConfig(projector_hidden=1_000_000, lora_rank=2))

    def test_autocast_handles_mixed_normalization_and_projection_dtypes(self):
        self.model.train()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits, count = self.model(torch.randn(1, 3, 4, 4),
                                       torch.tensor([[151644, 1, 151645]]))
            total, targets = caption_loss(logits, torch.tensor([[151644, 1, 151645]]),
                                          count, 151643)
        (total / targets).backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(self.model.projector[0].weight.grad).all())

    def test_batch_with_different_eos_times_keeps_finished_rows_finished(self):
        self.model.eval()
        images = torch.randn(2, 3, 4, 4)
        with torch.no_grad():
            visual = self.model.encode_images(images)
            bos = torch.full((2, 1), 151644, dtype=torch.long)
            embeddings = torch.cat([visual, self.decoder.embed_tokens(bos)], dim=1)
            hidden = []
            handle = self.decoder.norm.register_forward_hook(lambda module, inputs, output: hidden.append(output))
            self.decoder(inputs_embeds=embeddings)
            handle.remove()
            # Choose a real output projection that makes row 0 finish immediately
            # while row 1 emits a word. Attention/generation remain unmocked.
            rows = hidden[0][:, -1, :].double()
            vector = torch.linalg.lstsq(rows, torch.tensor([2., -2.], dtype=torch.float64)).solution.float()
            self.decoder.lm_head.weight.zero_()
            self.decoder.lm_head.weight[151645].copy_(vector)
            self.decoder.lm_head.weight[1].copy_(-vector)
        actual = self.model.generate(images, 151644, 151645, max_new_tokens=3)
        self.assertEqual(actual[0, 0].item(), 151645)
        self.assertEqual(actual[1, 0].item(), 1)
        self.assertGreater(actual.size(1), 1)
        self.assertTrue(actual[0].eq(151645).all())

    def test_local_pretrained_vision_checkpoint_may_include_classifier(self):
        import timm
        source = timm.create_model('vit_tiny_patch16_224', pretrained=False).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vision.pt'
            torch.save(source.state_dict(), path)
            loaded = create_vision(CaptionConfig(vision_model='vit_tiny_patch16_224'), path)
        torch.testing.assert_close(loaded.patch_embed.proj.weight, source.patch_embed.proj.weight)
        self.assertEqual(loaded.num_classes, 0)


if __name__ == '__main__':
    unittest.main()
