#!/usr/bin/env python3
"""Train the image projector and decoder LoRA on Problem 2 COCO captions."""

import argparse
from contextlib import nullcontext
import csv
from functools import partial
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from caption_data import CaptionDataset, caption_collate
from caption_model import (CaptionConfig, CaptionModel, caption_loss, create_vision,
                           image_transform, load_base_decoder, save_adapter)
from tokenization_qwen3 import Qwen3Tokenizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--decoder_path', type=Path, required=True)
    parser.add_argument('--vocab_path', type=Path, default=here / 'vocab.json')
    parser.add_argument('--merges_path', type=Path, default=here / 'merges.txt')
    parser.add_argument('--train_json', type=Path, required=True)
    parser.add_argument('--train_images', type=Path, required=True)
    parser.add_argument('--val_json', type=Path, required=True)
    parser.add_argument('--val_images', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=here / 'checkpoints')
    parser.add_argument('--vision_model', default='vit_base_patch16_clip_224.openai')
    parser.add_argument('--vision_path', type=Path, default=None,
                        help='Optional local pretrained timm encoder checkpoint')
    parser.add_argument('--projector_hidden', type=int, default=1024)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=float, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_targets', nargs='+', default=['q_proj', 'v_proj'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--projector_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--max_caption_tokens', type=int, default=48,
                        help='Total text length including BOS and EOS')
    parser.add_argument('--limit_train_images', type=int, default=None)
    parser.add_argument('--limit_val_images', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--amp', choices=['auto', 'none', 'bf16', 'fp16'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite_output_dir', action='store_true')
    args = parser.parse_args(argv)
    for field in ('epochs', 'batch_size', 'gradient_accumulation_steps'):
        if getattr(args, field) < 1:
            parser.error(f'--{field} must be positive')
    if args.workers < 0 or args.max_caption_tokens < 2:
        parser.error('workers must be nonnegative and max_caption_tokens must be at least 2')
    if not 0 <= args.warmup_ratio < 1:
        parser.error('--warmup_ratio must be in [0, 1)')
    if min(args.learning_rate, args.projector_lr, args.max_grad_norm) <= 0 or args.weight_decay < 0:
        parser.error('Learning rates/max_grad_norm must be positive and weight_decay nonnegative')
    for field in ('limit_train_images', 'limit_val_images'):
        if getattr(args, field) is not None and getattr(args, field) < 1:
            parser.error(f'--{field} must be positive')
    for field in ('decoder_path', 'vocab_path', 'merges_path', 'train_json', 'val_json', 'vision_path'):
        path = getattr(args, field)
        if path is not None and not path.is_file():
            parser.error(f'--{field}: file not found: {path}')
    for field in ('train_images', 'val_images'):
        if not getattr(args, field).is_dir():
            parser.error(f'--{field}: directory not found: {getattr(args, field)}')
    return args


def run_epoch(model, loader, device, amp_dtype, optimizer=None, scaler=None,
              accumulation_steps=1, max_grad_norm=1.0, scheduler=None, show_progress=True):
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_loss, total_tokens, group_tokens, updates = 0.0, 0, 0, 0
    progress = tqdm(loader, desc='Train' if training else 'Validation', disable=not show_progress)
    for index, (images, ids) in enumerate(progress):
        images, ids = images.to(device, non_blocking=True), ids.to(device, non_blocking=True)
        amp = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with torch.set_grad_enabled(training), amp:
            logits, visual_tokens = model(images, ids)
            summed_loss, token_count = caption_loss(logits, ids, visual_tokens, model.decoder.padding_idx)
        if not bool(torch.isfinite(summed_loss)):
            raise FloatingPointError('Non-finite caption loss; check data and AMP settings.')
        total_loss += float(summed_loss.detach())
        total_tokens += token_count
        if training:
            scaler.scale(summed_loss).backward()
            group_tokens += token_count
            if (index + 1) % accumulation_steps == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                # Normalize by actual caption tokens, including a partial final group.
                parameters = [p for p in model.parameters() if p.requires_grad]
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(group_tokens)
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= previous_scale
                if stepped:
                    updates += 1
                    if scheduler is not None:
                        scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group_tokens = 0
        if show_progress:
            progress.set_postfix(loss=f'{total_loss / total_tokens:.4f}')
    if not total_tokens:
        raise ValueError('Data loader produced no caption targets.')
    return total_loss / total_tokens, updates


def run_training(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_output_dir:
        raise ValueError('Output directory is not empty; choose another or use --overwrite_output_dir.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type not in {'cpu', 'cuda'} or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError(f'Unsupported or unavailable device: {device}')
    if device.type == 'cpu':
        if args.amp not in {'auto', 'none'}:
            raise ValueError('Use --amp none for CPU training.')
        amp_dtype = None
    elif args.amp == 'none':
        amp_dtype = None
    elif args.amp == 'bf16' or (args.amp == 'auto' and torch.cuda.is_bf16_supported()):
        if not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU does not support bf16; use --amp fp16.')
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    config = CaptionConfig(args.vision_model, args.projector_hidden, args.lora_rank,
                           args.lora_alpha, args.lora_dropout, tuple(args.lora_targets))
    tokenizer = Qwen3Tokenizer(str(args.vocab_path), str(args.merges_path))
    decoder = load_base_decoder(args.decoder_path)
    vision = create_vision(config, args.vision_path)
    transform = image_transform(vision)
    model = CaptionModel(decoder, vision, config).to(device)
    print(f'Device: {device}; AMP: {amp_dtype}; trained parameters: {model.trainable_count:,}')
    train_data = CaptionDataset(args.train_json, args.train_images, tokenizer, transform,
                                args.max_caption_tokens, args.limit_train_images)
    val_data = CaptionDataset(args.val_json, args.val_images, tokenizer, transform,
                              args.max_caption_tokens, args.limit_val_images)
    collate = partial(caption_collate, pad_id=tokenizer.encoder['<|endoftext|>'])
    options = dict(batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate,
                   pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val_data, shuffle=False, **options)
    groups = [
        {'params': list(model.projector.parameters()), 'lr': args.projector_lr},
        {'params': [p for name, p in model.decoder.named_parameters() if p.requires_grad],
         'lr': args.learning_rate},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)

    def lr_multiplier(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    with (args.output_dir / 'loss.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'val_loss', 'optimizer_updates'])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, updates = run_epoch(
                model, train_loader, device, amp_dtype, optimizer, scaler,
                args.gradient_accumulation_steps, args.max_grad_norm, scheduler)
            val_loss, _ = run_epoch(model, val_loader, device, amp_dtype)
            writer.writerow(dict(epoch=epoch, train_loss=train_loss,
                                 val_loss=val_loss, optimizer_updates=updates))
            stream.flush()
            metadata = dict(epoch=epoch, val_loss=val_loss, seed=args.seed,
                            max_caption_tokens=args.max_caption_tokens)
            save_adapter(model, args.output_dir / 'last.pt', **metadata)
            if val_loss < best:
                best = val_loss
                save_adapter(model, args.output_dir / 'best.pt', **metadata)
            print(f'Epoch {epoch}: train loss={train_loss:.4f}, validation loss={val_loss:.4f}')
    print(f'Best adapter by validation loss: {args.output_dir / "best.pt"}')
    return args.output_dir / 'best.pt'


if __name__ == '__main__':
    run_training(parse_args())
