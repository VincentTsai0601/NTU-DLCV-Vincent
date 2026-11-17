import json
from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from tokenization_qwen3 import Qwen3Tokenizer, bytes_to_unicode

try:
    from caption_data import CaptionDataset, caption_collate, collect_images, decode_caption
except ModuleNotFoundError as error:
    if error.name != 'caption_data':
        raise
    CaptionDataset = None


def write_tokenizer(directory):
    vocab = Path(directory) / 'vocab.json'
    merges = Path(directory) / 'merges.txt'
    vocab.write_text(json.dumps({char: i for i, char in enumerate(bytes_to_unicode().values())}))
    merges.write_text('#version: 0.2\n')
    return Qwen3Tokenizer(str(vocab), str(merges)), vocab, merges


class DataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(CaptionDataset, 'Caption data loader has not been implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer, _, _ = write_tokenizer(self.root)
        for name in ('photo.name.jpg', 'second.png'):
            Image.new('RGB', (4, 4), 'red').save(self.root / name)
        self.annotations = self.root / 'train.json'
        self.payload = {
            'images': [{'id': 17, 'file_name': 'photo.name.jpg'},
                       {'id': 90, 'file_name': 'second.png'}],
            'annotations': [{'image_id': 90, 'caption': 'dog'},
                            {'image_id': 17, 'caption': 'a cat'},
                            {'image_id': 17, 'caption': 'cat'}],
        }
        self.annotations.write_text(json.dumps(self.payload))

    def test_joins_image_ids_and_keeps_multiple_captions(self):
        data = CaptionDataset(self.annotations, self.root, self.tokenizer, ToTensor(), 8)
        self.assertEqual(len(data), 3)
        image, ids = data[1]
        self.assertEqual(tuple(image.shape), (3, 4, 4))
        self.assertEqual(ids[0].item(), 151644)
        self.assertEqual(ids[-1].item(), 151645)
        self.assertEqual(self.tokenizer.decode(ids[1:-1].tolist()), 'a cat')

    def test_truncation_keeps_eos_and_collate_uses_supplied_pad_id(self):
        data = CaptionDataset(self.annotations, self.root, self.tokenizer, ToTensor(), 4)
        _, ids = data[1]
        self.assertEqual(ids.tolist()[-1], 151645)
        self.assertLessEqual(len(ids), 4)
        image = data[0][0]
        _, padded = caption_collate([(image, torch.tensor([151644, 1, 151645])),
                                     (image, torch.tensor([151644, 151645]))], 151643)
        self.assertEqual(padded.tolist(), [[151644, 1, 151645], [151644, 151645, 151643]])

    def test_image_limit_keeps_all_captions_of_selected_images(self):
        data = CaptionDataset(self.annotations, self.root, self.tokenizer,
                              ToTensor(), 8, limit_images=1)
        self.assertEqual(len(data), 2)
        self.assertEqual(self.tokenizer.decode(data[1][1][1:-1].tolist()), 'cat')

    def test_unknown_image_id_is_reported(self):
        self.payload['annotations'][0]['image_id'] = 999
        self.annotations.write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ValueError, '999'):
            CaptionDataset(self.annotations, self.root, self.tokenizer, ToTensor(), 8)

    def test_output_stems_and_duplicate_stems(self):
        paths = collect_images(self.root)
        self.assertEqual([path.stem for path in paths], ['photo.name', 'second'])
        Image.new('RGB', (4, 4)).save(self.root / 'second.jpg')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            collect_images(self.root)

    def test_empty_image_folder_is_reported(self):
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, 'No images'):
            collect_images(empty)

    def test_decode_stops_at_eos_and_removes_special_tokens(self):
        ids = [151644] + self.tokenizer.encode('cat') + [151645] + self.tokenizer.encode('dog')
        self.assertEqual(decode_caption(self.tokenizer, ids), 'cat')


if __name__ == '__main__':
    unittest.main()
