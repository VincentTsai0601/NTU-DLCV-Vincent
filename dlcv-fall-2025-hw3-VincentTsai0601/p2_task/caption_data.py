"""COCO caption pairs, image-only inference data, and output decoding."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    def __init__(self, annotation_file, images_root, tokenizer, transform,
                 max_length=48, limit_images=None):
        if max_length < 2:
            raise ValueError("max_length must leave room for BOS and EOS.")
        if limit_images is not None and limit_images < 1:
            raise ValueError("limit_images must be positive.")
        payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not {'images', 'annotations'} <= payload.keys():
            raise ValueError("Expected COCO JSON with images and annotations arrays.")
        files = {}
        for image in payload['images']:
            if image['id'] in files:
                raise ValueError(f"Duplicate image id: {image['id']}")
            files[image['id']] = image['file_name']
        selected = set(list(files)[:limit_images]) if limit_images else set(files)
        self.samples = []
        for annotation in payload['annotations']:
            image_id = annotation['image_id']
            if image_id not in files:
                raise ValueError(f"Caption references unknown image id {image_id}.")
            if image_id not in selected:
                continue
            caption = annotation['caption']
            if not isinstance(caption, str):
                raise ValueError(f"Caption for image {image_id} must be a string.")
            ids = [tokenizer.encoder['<|im_start|>']]
            ids += tokenizer.encode(caption)[:max_length - 2]
            ids += [tokenizer.encoder['<|im_end|>']]
            self.samples.append((Path(images_root) / files[image_id],
                                 torch.tensor(ids, dtype=torch.long)))
        if not self.samples:
            raise ValueError("No image-caption pairs were selected.")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, ids = self.samples[index]
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, ids


def caption_collate(batch, pad_id):
    images, ids = zip(*batch)
    return torch.stack(images), pad_sequence(ids, batch_first=True, padding_value=pad_id)


def collect_images(images_root):
    root = Path(images_root)
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    paths = sorted(path for path in root.iterdir()
                   if path.is_file() and path.suffix.lower() in extensions)
    if not paths:
        raise ValueError(f"No images found in {root}.")
    seen = set()
    for path in paths:
        if path.stem in seen:
            raise ValueError(f"Duplicate output filename stem: {path.stem}")
        seen.add(path.stem)
    return paths


class ImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, path.stem


def decode_caption(tokenizer, ids):
    eos = tokenizer.encoder['<|im_end|>']
    skip = {tokenizer.encoder['<|im_start|>'], tokenizer.encoder['<|endoftext|>']}
    clean = []
    for token in ids:
        token = int(token)
        if token == eos:
            break
        if token not in skip:
            clean.append(token)
    return tokenizer.decode(clean).strip()
