import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor

from test_data import write_tokenizer
from test_model import small_decoder, TinyVision
from caption_model import CaptionConfig, CaptionModel, load_base_decoder

try:
    import train
    import inference
except ModuleNotFoundError as error:
    if error.name not in {'train', 'inference'}:
        raise
    train = inference = None


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(train, 'Training and image inference CLIs are not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(9)
        torch.set_num_threads(1)

    def test_token_weighted_accumulation_matches_a_single_batch(self):
        config = CaptionConfig(projector_hidden=8, lora_rank=2, lora_alpha=4)
        first = CaptionModel(small_decoder(), TinyVision(), config)
        # Build a second model from state, avoiding copied bound-method closures.
        second = CaptionModel(small_decoder(), TinyVision(), config)
        second.load_state_dict(first.state_dict())
        images = torch.randn(2, 3, 4, 4)
        ids = torch.tensor([[151644, 1, 2, 151645], [151644, 2, 151645, 151643]])
        data = TensorDataset(images, ids)
        results = []
        for model, batch, accumulation in ((first, 2, 1), (second, 1, 3)):
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            loss, updates = train.run_epoch(
                model, DataLoader(data, batch_size=batch), torch.device('cpu'), None,
                optimizer=optimizer, scaler=scaler, accumulation_steps=accumulation,
                max_grad_norm=1e6, show_progress=False)
            results.append(loss)
            self.assertEqual(updates, 1)
        self.assertAlmostEqual(results[0], results[1], places=5)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_train_save_reload_and_emit_image_caption_json(self):
        _, vocab, merges = write_tokenizer(self.root)
        images = self.root / 'images'
        images.mkdir()
        Image.new('RGB', (4, 4), 'red').save(images / 'a.photo.jpg')
        Image.new('RGB', (4, 4), 'blue').save(images / 'b.png')
        annotation = self.root / 'train.json'
        annotation.write_text(json.dumps({
            'images': [{'id': 5, 'file_name': 'a.photo.jpg'}, {'id': 9, 'file_name': 'b.png'}],
            'annotations': [{'image_id': 5, 'caption': 'red'}, {'image_id': 9, 'caption': 'blue'}],
        }))
        decoder = small_decoder()
        base_path = self.root / 'decoder.bin'
        torch.save(decoder.state_dict(), base_path)
        cfg = decoder.config
        out = self.root / 'checkpoints'
        train_args = train.parse_args([
            '--decoder_path', str(base_path), '--vocab_path', str(vocab),
            '--merges_path', str(merges), '--train_json', str(annotation),
            '--train_images', str(images), '--val_json', str(annotation),
            '--val_images', str(images), '--output_dir', str(out),
            '--epochs', '1', '--batch_size', '2', '--workers', '0',
            '--projector_hidden', '8', '--lora_rank', '2', '--lora_alpha', '4',
            '--device', 'cpu', '--amp', 'none',
        ])
        # Only substitute pretrained asset construction; run the real decoder,
        # tokenizer, LoRA, loss, optimizer, checkpointing, and output pipeline.
        def base_loader(path):
            return load_base_decoder(path, cfg)
        with patch.object(train, 'load_base_decoder', side_effect=base_loader), \
                patch.object(train, 'create_vision', side_effect=lambda *a: TinyVision()), \
                patch.object(train, 'image_transform', return_value=ToTensor()):
            best = train.run_training(train_args)
        self.assertTrue(best.is_file())
        self.assertTrue((out / 'loss.csv').is_file())
        prediction_path = self.root / 'predictions' / 'val.json'
        infer_args = inference.parse_args([
            '--decoder_path', str(base_path), '--checkpoint', str(best),
            '--images_root', str(images), '--output', str(prediction_path),
            '--vocab_path', str(vocab), '--merges_path', str(merges),
            '--device', 'cpu', '--batch_size', '2', '--workers', '0', '--max_new_tokens', '3',
        ])
        with patch.object(inference, 'load_base_decoder', side_effect=base_loader), \
                patch.object(inference, 'create_vision', side_effect=lambda *a: TinyVision()), \
                patch.object(inference, 'image_transform', return_value=ToTensor()):
            predictions = inference.run_inference(infer_args)
        self.assertEqual(set(predictions), {'a.photo', 'b'})
        self.assertEqual(predictions, json.loads(prediction_path.read_text()))
        self.assertTrue(all(isinstance(text, str) for text in predictions.values()))


if __name__ == '__main__':
    unittest.main()
