#!/usr/bin/env python3
"""Generate Problem 2 image captions from a trained projector/LoRA checkpoint."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from caption_data import ImageDataset, collect_images, decode_caption
from caption_model import (CaptionConfig, CaptionModel, create_vision, image_transform,
                           load_adapter, load_base_decoder, read_checkpoint)
from tokenization_qwen3 import Qwen3Tokenizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--images_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decoder_path', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=here / 'checkpoints' / 'best.pt')
    parser.add_argument('--vocab_path', type=Path, default=here / 'vocab.json')
    parser.add_argument('--merges_path', type=Path, default=here / 'merges.txt')
    parser.add_argument('--vision_path', type=Path, default=None,
                        help='Optional local pretrained timm encoder checkpoint')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto')
    args = parser.parse_args(argv)

    if args.batch_size < 1 or args.max_new_tokens < 1 or args.workers < 0:
        parser.error('batch_size/max_new_tokens must be positive and workers nonnegative')

    for field in ('decoder_path', 'checkpoint', 'vocab_path', 'merges_path', 'vision_path'):
        path = getattr(args, field)
        if path is not None and not path.is_file():
            parser.error(f'--{field}: file not found: {path}')

    if not args.images_root.is_dir():
        parser.error(f'Image directory does not exist: {args.images_root}')

    return args


def run_inference(args):
    start = time.monotonic()
    paths = collect_images(args.images_root)

    if any(args.output.resolve() == path.resolve() for path in paths):
        raise ValueError('The output path must not overwrite an input image.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)

    if device.type not in {'cpu', 'cuda'} or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError(f'Unsupported or unavailable device: {device}')
    dtype = (torch.float16 if device.type == 'cuda' else torch.float32) if args.dtype == 'auto' else getattr(torch, args.dtype)
    
    if device.type == 'cpu' and dtype != torch.float32:
        raise ValueError('Use --dtype float32 for CPU inference.')
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bfloat16.')
    
    payload = read_checkpoint(args.checkpoint)
    config_values = dict(payload['config'])
    config_values['lora_targets'] = tuple(config_values['lora_targets'])
    config = CaptionConfig(**config_values)
    tokenizer = Qwen3Tokenizer(str(args.vocab_path), str(args.merges_path))
    decoder = load_base_decoder(args.decoder_path)
    vision = create_vision(config, args.vision_path)
    transform = image_transform(vision)
    model = CaptionModel(decoder, vision, config)
    load_adapter(model, payload)
    del payload
    model.to(device=device, dtype=dtype).eval()
    loader = DataLoader(ImageDataset(paths, transform), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    
    predictions = {}
    bos, eos = tokenizer.encoder['<|im_start|>'], tokenizer.encoder['<|im_end|>']
    print(f'Captioning {len(paths)} images on {device}; trained parameters: {model.trainable_count:,}')
    with torch.inference_mode():
        for images, stems in tqdm(loader, desc='Caption inference'):
            images = images.to(device=device, dtype=dtype, non_blocking=True)
            generated = model.generate(images, bos, eos, args.max_new_tokens).cpu().tolist()
            for stem, ids in zip(stems, generated):
                predictions[stem] = decode_caption(tokenizer, ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(args.output)
    print(f'Saved {len(predictions)} captions to {args.output}; elapsed {time.monotonic()-start:.1f}s')
    
    return predictions


if __name__ == '__main__':
    run_inference(parse_args())
