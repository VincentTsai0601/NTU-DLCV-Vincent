# DLCV HW3 Problem 2: ViT + projector + decoder LoRA

This package extends the earlier text-only check into image-caption training
and inference. The provided `decoder.py` and `tokenization_qwen3.py` are included
unchanged. No Hugging Face Transformers/PEFT implementation is used.

## What is trained

| Component | Setting | Trained parameters |
|---|---|---:|
| Encoder | Frozen timm CLIP ViT-B/16, 224px, 196 patch tokens | 0 |
| Projector | Linear(768,1024), GELU, Linear(1024,1024) | 1,837,056 |
| Decoder | Supplied pretrained model; q/v LoRA, rank 8, alpha 16 | 1,146,880 |
| Total | Must remain strictly below 10M | 2,983,936 |

The supplied decoder already accepts `inputs_embeds`. We prefix projected image
patches to the text embeddings. Only caption/EOS next-token targets contribute
to training loss. Runtime SDPA attention preserves the original projections and
RoPE while enabling per-layer KV caches for batched greedy inference.

## Files you must supply

Extract this package into your homework repository root, beside `hw3_data/`.
Place the supplied `vocab.json` and `merges.txt` beside the Python files, or pass
their paths explicitly. The required pretrained text weights remain at a path
you supply, such as `hw3_data/p2_data/decoder_model.bin`.

The package contains source code and tests, **not a trained caption adapter**.
Train first; inference deliberately rejects missing/incomplete trained weights.
It also rejects an incompatible decoder or projector/LoRA configuration.

Activate your existing Problem 2 environment:

```bash
conda activate DLCV-HW3-P2
python -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build suitable for your driver. The grading
versions are Python 3.10, torch 2.5.1, torchvision 0.20.1, and timm 1.0.9.
CIDEr/CLIPScore evaluation additionally needs the assignment's evaluation
dependencies; they are not needed by this training/inference code.
`huggingface_hub==0.25.2` is pinned as a timm download dependency because newer
Hub releases removed an import required by timm 1.0.9. This is separate from
the prohibited direct use of Transformers/PEFT.

## First: run a small training check

```bash
python3 train.py \
  --decoder_path hw3_data/p2_data/decoder_model.bin \
  --train_json hw3_data/p2_data/train.json \
  --train_images hw3_data/p2_data/images/train \
  --val_json hw3_data/p2_data/val.json \
  --val_images hw3_data/p2_data/images/val \
  --output_dir runs/p2_smoke \
  --limit_train_images 32 \
  --limit_val_images 32 \
  --epochs 10 \
  --batch_size 2 \
  --gradient_accumulation_steps 2
```

The image limit selects unique images and keeps all their caption pairs. Check
that loss falls. To check memorization, run inference on those training images
and inspect the captions; a small validation subset has different images and
does not measure memorization. No metric threshold is guaranteed by this code.

## Then: train on the full dataset

```bash
python3 train.py \
  --decoder_path hw3_data/p2_data/decoder_model.bin \
  --train_json hw3_data/p2_data/train.json \
  --train_images hw3_data/p2_data/images/train \
  --val_json hw3_data/p2_data/val.json \
  --val_images hw3_data/p2_data/images/val \
  --output_dir checkpoints \
  --epochs 5 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lora_rank 8 \
  --lora_alpha 16
```

Defaults: AdamW, projector LR 1e-3, LoRA LR 1e-4, 5% warmup then cosine decay,
48 text tokens including BOS/EOS, and automatic bf16/fp16 CUDA autocast. Frozen
weights remain float32 during AMP training. Reduce batch size if GPU memory is
insufficient. `--workers 0` is useful when debugging data-loader errors.

Each epoch creates:

- `checkpoints/best.pt`: lowest validation cross entropy, not necessarily best CIDEr.
- `checkpoints/last.pt`: latest epoch.
- `checkpoints/loss.csv`: actual epoch train/validation losses and optimizer updates.

The checkpoints contain only projector/LoRA parameters and architecture metadata;
the encoder and original decoder weights are not copied into them. Existing
nonempty output directories require `--overwrite_output_dir`; this starts a new
run and does not resume the old optimizer state.

For the report's second LoRA experiment, repeat full training with a distinct
output directory and `--lora_rank 16 --lora_alpha 32` (4,130,816 trained parameters
for the default architecture). Compare CIDEr and CLIPScore for both experiments,
and use the better measured setting for submission.

## Caption inference

```bash
python3 inference.py \
  --images_root hw3_data/p2_data/images/val \
  --output outputs/p2_val.json \
  --decoder_path hw3_data/p2_data/decoder_model.bin \
  --checkpoint checkpoints/best.pt \
  --batch_size 4 \
  --max_new_tokens 32
```

The checkpoint determines encoder, projector, LoRA rank/alpha/targets; repeat
those training flags only when training, not during inference. Caption generation
uses the image once, then cached autoregressive greedy decoding. File keys are
stems (e.g. `photo.name.jpg` becomes `photo.name`). Different image files with the
same stem are rejected to prevent silently lost predictions. Supported inputs
are images directly inside the specified directory, not nested directories.

Example output:

```json
{
  "000000000000": "A person playing a guitar.",
  "000000000001": "Two dogs standing on grass."
}
```

Use the supplied `evaluate.py` to compute CIDEr and CLIPScore. Check its `--help`
for exact flags; the slides describe prediction, annotation, and image-root
arguments. Expected usage, if those flags match your copy:

```bash
python3 evaluate.py \
  --pred_file outputs/p2_val.json \
  --annotation_file hw3_data/p2_data/val.json \
  --images_root hw3_data/p2_data/images/val
```

## Pretrained vision assets and submission

By default timm loads `vit_base_patch16_clip_224.openai` pretrained weights from
its download cache, downloading them on the first run when absent. The frozen
encoder must be the same pretrained encoder used during training. For an offline
machine, `--vision_path PATH` can load an equivalent local timm-compatible
pretrained checkpoint; do not substitute a randomly initialized encoder.

Prepare the pretrained encoder cache before timed grading, using the assignment's
permitted checkpoint-download workflow where needed. Test in a clean environment;
your workstation's existing cache will not automatically exist on the TA machine.
No frozen weights are included in the adapter checkpoint or this package.

The TA entrypoint is:

```bash
bash hw3_2.sh ../hw3_data/p2_data/images/val ../hw3/output_p2/pred.json ../hw3_data/p2_data/decoder_model.bin
```

It uses `checkpoints/best.pt`, `vocab.json`, and `merges.txt` beside the Python
files. Include those assets and all imported Python files in your final repo.
If using externally hosted trained weights, supply the permitted download script.
Keep only your chosen best trained adapter for the submitted setting; the supplied
decoder weights are passed as the third argument and need not be uploaded.

Measure the entire inference command on the grading-like GPU to confirm the
40-minute limit. Cached decoding is implemented, but no full-size GPU throughput
or dataset caption scores have been measured here.

## Checks

```bash
python3 -m unittest discover -s tests -v
bash -n hw3_2.sh
```

Tests exercise the real supplied decoder with smaller dimensions and a small
vision fixture: initial LoRA output preservation, SDPA equivalence, gradients,
target alignment, padding, cache equivalence, adapter serialization, image-ID
joins, filename handling, token-weighted accumulation, and one complete tiny
training/reload/image-inference cycle. They do not establish full-model quality.

External references:
- https://github.com/microsoft/LoRA
- https://huggingface.co/timm/vit_base_patch16_clip_224.openai
- https://huggingface.co/docs/timm/feature_extraction
