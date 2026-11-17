# Problem 2 captioning design

Build on the supplied, unchanged `decoder.py` and `tokenization_qwen3.py`.
Use a frozen timm CLIP ViT-B/16 at 224px, its 196 patch tokens, a trainable
768→1024→1024 GELU MLP, and loralib adapters on decoder q_proj/v_proj.
The default rank is 8 and alpha is 16; the expected trained parameter count
with the supplied decoder is 2,983,936. Enforce strictly fewer than 10M.

Use all image-caption pairs for teacher-forcing training. Prefix visual
embeddings to BOS + caption + EOS, with next-token loss only on caption/EOS.
Exclude visual positions and padding from the loss. Train using AdamW, optional
CUDA AMP, and gradient accumulation. Save only trainable parameters plus
architecture metadata. The supplied decoder and pretrained vision weights
must be loaded independently at inference.

Replace eager attention on the model instance with equivalent PyTorch SDPA;
support separate per-layer KV caches for batched greedy caption generation.
The base decoder source and tokenizer source remain unchanged. JSON predictions
map filename stems to generated captions. Reject duplicate stems and missing,
incomplete, or incompatible trained checkpoints instead of using random weights.

Deliver training and inference CLIs, `hw3_2.sh` with image-dir/output/decoder
positional arguments, documentation, tests, and a zip containing the two supplied
modules. No trained caption weights or baseline scores can be claimed here.

Approaches: a text-only wrapper verifies imports but does not finish captioning;
editing the provided decoder in place complicates comparison to the original;
a separate model module with runtime attention adaptation supports the required
pipeline and preserves original modules. The latter is the implementation choice.
