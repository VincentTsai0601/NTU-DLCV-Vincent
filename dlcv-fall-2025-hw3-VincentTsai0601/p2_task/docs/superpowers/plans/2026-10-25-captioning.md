# Problem 2 Captioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Train and run image captioning with the supplied decoder/tokenizer and fewer than 10M trained parameters.

**Architecture:** Frozen CLIP ViT patch tokens feed a two-layer MLP and the provided decoder. Decoder q/v LoRA is trained with caption cross entropy. Runtime SDPA attention enables cached autoregressive inference without modifying the supplied source.

**Tech Stack:** Python 3.10, torch 2.5.1, torchvision 0.20.1, timm 1.0.9, loralib 0.1.2, Pillow, regex.

**Spec:** `docs/superpowers/specs/2026-10-25-captioning-design.md`

## Global Constraints

- Fewer than 10,000,000 trained parameters, including the projector.
- Supplied decoder, pretrained weights, and tokenizer; no transformers/PEFT imports.
- Save only trained parameters, not frozen vision/base-decoder weights.
- `hw3_2.sh $1 $2 $3`: image directory, output JSON, supplied decoder checkpoint.
- JSON keys are filename stems; inference has a 40-minute grading limit (must be measured on the target GPU).

## Review Focus

- Shifted targets must supervise the first caption token, EOS, and no visual/pad positions.
- LoRA insertion must preserve pretrained outputs at initialization.
- Cached decoding must match uncached decoding, including batch completion and RoPE positions.
- Checkpoints must contain exactly the expected trainable keys and no frozen weights.
- Filename collisions, empty images, and unavailable pretrained assets must fail clearly.

### Task 1: Model and numerical contracts

**Files:** `caption_model.py`, `tests/test_model.py`, unchanged supplied modules.
**Interfaces:** `CaptionConfig`, `CaptionModel(decoder, vision, config)`, `caption_loss`, `load_base_decoder`, `save_adapter`, `load_adapter`.
- [x] Write and run failing numerical tests for LoRA preservation, gradients/freezing, loss alignment, SDPA equivalence, cached decoding, and adapter roundtrip.
- [x] Implement the model and strict trained-checkpoint validation.
- [x] Run the numerical tests and record outcomes.

### Task 2: Data and runnable CLIs

**Files:** `caption_data.py`, `train.py`, `inference.py`, `hw3_2.sh`, `tests/test_data.py`.
**Interfaces:** COCO annotations become `(image_tensor, token_ids)` pairs; inference emits `{stem: caption}` using metadata from `save_adapter`.
- [x] Write and run failing dataset, padding, stem, CLI, and tiny training/inference tests.
- [x] Implement deterministic data loading, token-weighted accumulation, validation loss, trainable checkpoints, and cached batched inference.
- [x] Run all tests, syntax checks, and CLI checks.

### Task 3: Documentation and delivery

**Files:** `README.md`, `requirements.txt`, zip archive.
- [x] Document small-subset training, full training, LoRA comparison, evaluation, pretrained-asset caching, and grading interface.
- [x] Request a fresh review, resolve important findings, and rerun the suite.
- [x] Save the complete runnable project and updated inference script.
