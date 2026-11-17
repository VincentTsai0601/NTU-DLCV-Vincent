"""Vision-prefix captioning and decoder LoRA; no Transformers dependency."""

from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
import warnings

import loralib as lora
import torch
from torch import nn
from torch.nn import functional as F

from decoder import Config, Decoder, apply_rotary_pos_emb, repeat_kv


@dataclass(frozen=True)
class CaptionConfig:
    vision_model: str = "vit_base_patch16_clip_224.openai"
    projector_hidden: int = 1024
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "v_proj")

    def __post_init__(self):
        if self.projector_hidden < 1 or self.lora_rank < 1 or self.lora_alpha <= 0:
            raise ValueError("Projector width, LoRA rank, and alpha must be positive.")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1).")
        allowed = {"q_proj", "k_proj", "v_proj", "o_proj"}
        if (not self.lora_targets or len(set(self.lora_targets)) != len(self.lora_targets)
                or not set(self.lora_targets) <= allowed):
            raise ValueError("LoRA targets must be distinct attention projection names.")


def load_base_decoder(path, config=None):
    model = Decoder(config or Config())
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise ValueError("Incomplete base decoder weights; missing keys: "
                         + ", ".join(incompatible.missing_keys[:10]))
    if incompatible.unexpected_keys:
        warnings.warn("Unused base checkpoint keys: "
                      + ", ".join(incompatible.unexpected_keys[:10]))
    return model


def create_vision(config, vision_path=None):
    """Load pretrained timm ViT weights, optionally from a local timm checkpoint."""
    import timm
    options = {}
    if vision_path is not None:
        if not Path(vision_path).is_file():
            raise ValueError(f"Pretrained vision checkpoint does not exist: {vision_path}")
        # Use timm's pretrained loader/filter so ordinary checkpoints may retain
        # their classifier head while the captioner requests patch features only.
        options['pretrained_cfg_overlay'] = {
            'file': str(vision_path),
            # Some ViT default tags point to JAX .npz files; local torch state
            # dictionaries must use timm's ordinary loader instead.
            'custom_load': Path(vision_path).suffix.lower() == '.npz',
        }
    model = timm.create_model(config.vision_model, pretrained=True, num_classes=0, **options)
    if not hasattr(model, "patch_embed"):
        raise ValueError("The vision encoder must be a Vision Transformer.")
    return model


def image_transform(vision):
    from timm.data import create_transform, resolve_model_data_config
    # Use the exact preprocessing associated with the pretrained encoder.
    return create_transform(**resolve_model_data_config(vision), is_training=False)


def _attention_forward(self, hidden_states, position_embeddings, attention_mask):
    """Same Qwen3 projections/RoPE with SDPA and optional inference KV caching."""
    shape = (*hidden_states.shape[:-1], -1, self.head_dim)
    query = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    # RMSNorm/RoPE may promote Q/K while autocast leaves V in half precision.
    query, key = query.to(value.dtype), key.to(value.dtype)
    past = self._caption_cache if self._caption_use_cache else None
    if past is not None:
        key = torch.cat([past[0], key], dim=2)
        value = torch.cat([past[1], value], dim=2)
    if self._caption_use_cache:
        self._caption_cache = (key, value)
    key = repeat_kv(key, self.num_key_value_groups)
    value = repeat_kv(value, self.num_key_value_groups)
    # The supplied decoder uses True=blocked; SDPA uses True=allowed.
    allowed = None if attention_mask is None else ~attention_mask
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=allowed, dropout_p=0.0,
        is_causal=attention_mask is None, scale=self.scaling,
    )
    output = output.transpose(1, 2).contiguous().reshape(*hidden_states.shape[:-1], -1)
    return self.o_proj(output), None


def caption_loss(logits, input_ids, visual_tokens, pad_id):
    """Return summed next-token CE and number of caption/EOS targets."""
    if logits.shape[:2] != (input_ids.size(0), visual_tokens + input_ids.size(1)):
        raise ValueError("Logits and visual/text sequence lengths do not match.")
    labels = torch.full(logits.shape[:2], -100, device=input_ids.device, dtype=torch.long)
    labels[:, visual_tokens:] = input_ids
    labels[:, visual_tokens] = -100  # Do not train the last visual token to predict BOS.
    labels.masked_fill_(labels.eq(pad_id), -100)
    targets = labels[:, 1:]
    count = int(targets.ne(-100).sum().item())
    if count == 0:
        raise ValueError("Batch contains no caption/EOS targets.")
    total = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                           targets.reshape(-1), ignore_index=-100, reduction="sum")
    return total, count


class CaptionModel(nn.Module):

    def __init__(self, decoder, vision, config):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.vision = vision

        decoder.requires_grad_(False)
        vision.requires_grad_(False)
        targets = []

        for layer in decoder.layers:
            for name in config.lora_targets:
                linear = getattr(layer.self_attn, name)
                targets.append((layer.self_attn, name, linear))
        width, hidden = vision.num_features, decoder.config.hidden_size
        projected_count = (width + 1) * config.projector_hidden + (config.projector_hidden + 1) * hidden
        projected_count += sum(config.lora_rank * (linear.in_features + linear.out_features)
                               for _, _, linear in targets)
        if projected_count >= 10_000_000:
            raise ValueError(f"Trained parameter count {projected_count:,} must be below 10,000,000.")
        self.projector = nn.Sequential(nn.Linear(width, config.projector_hidden), nn.GELU(),
                                       nn.Linear(config.projector_hidden, hidden))
        for attention, name, linear in targets:
            adapted = lora.Linear(
                linear.in_features, linear.out_features, r=config.lora_rank,
                lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                bias=linear.bias is not None, merge_weights=False,
            ).to(device=linear.weight.device, dtype=linear.weight.dtype)
            adapted.weight = linear.weight
            if linear.bias is not None:
                adapted.bias = linear.bias
            setattr(attention, name, adapted)
        for layer in decoder.layers:
            attention = layer.self_attn
            attention._caption_use_cache = False
            attention._caption_cache = None
            attention.forward = MethodType(_attention_forward, attention)
        self.vision.eval()
        if self.trainable_count >= 10_000_000:
            raise ValueError("Trained parameter count must be below 10,000,000.")

    @property
    def trainable_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        return self

    def encode_images(self, images):
        with torch.no_grad():
            features = self.vision.forward_features(images)
            if features.ndim != 3:
                raise ValueError("ViT must return unpooled [batch, tokens, features].")
            features = features[:, getattr(self.vision, "num_prefix_tokens", 1):]
        if features.size(1) == 0:
            raise ValueError("Vision encoder returned no patch tokens.")
        return self.projector(features)

    def forward(self, images, input_ids):
        visual = self.encode_images(images)
        embeddings = torch.cat([visual, self.decoder.embed_tokens(input_ids)], dim=1)
        batch, length = embeddings.shape[:2]
        if length > self.decoder.config.max_position_embeddings:
            raise ValueError("Visual/text sequence exceeds the decoder context length.")
        blocked = torch.ones(length, length, device=embeddings.device,
                             dtype=torch.bool).triu(1)[None, None].expand(batch, 1, length, length)
        padding = torch.cat([torch.zeros(batch, visual.size(1), device=input_ids.device,
                                         dtype=torch.bool),
                             input_ids.eq(self.decoder.padding_idx)], dim=1)
        blocked = blocked | padding[:, None, None, :]
        return self.decoder(inputs_embeds=embeddings, attention_mask=blocked), visual.size(1)

    def _cache(self, enabled):
        for layer in self.decoder.layers:
            layer.self_attn._caption_use_cache = enabled
            layer.self_attn._caption_cache = None

    @torch.inference_mode()
    def generate(self, images, bos_id, eos_id, max_new_tokens=32):
        if self.training:
            raise ValueError("Call model.eval() before caption generation.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        visual = self.encode_images(images)
        batch = visual.size(0)
        bos = torch.full((batch, 1), bos_id, device=images.device, dtype=torch.long)
        embeddings = torch.cat([visual, self.decoder.embed_tokens(bos)], dim=1)
        prefix_length = embeddings.size(1)
        if prefix_length + max_new_tokens > self.decoder.config.max_position_embeddings:
            raise ValueError("Generation exceeds the decoder context length.")
        finished = torch.zeros(batch, device=images.device, dtype=torch.bool)
        generated = []
        self._cache(True)
        try:
            for step in range(max_new_tokens):
                if step == 0:
                    # Decoder builds the usual full-prefix causal mask and positions.
                    logits = self.decoder(inputs_embeds=embeddings)[:, -1, :].clone()
                else:
                    embeddings = self.decoder.embed_tokens(generated[-1][:, None])
                    position = prefix_length + step - 1
                    positions = torch.tensor([[position]], device=images.device)
                    # A single new token may attend to all cached tokens and itself.
                    blocked = torch.zeros(batch, 1, 1, position + 1,
                                          device=images.device, dtype=torch.bool)
                    logits = self.decoder(inputs_embeds=embeddings, position_ids=positions,
                                          attention_mask=blocked)[:, -1, :].clone()
                logits[:, [self.decoder.padding_idx, bos_id]] = -torch.inf
                token = logits.argmax(-1).masked_fill(finished, eos_id)
                generated.append(token)
                finished |= token.eq(eos_id)
                if bool(finished.all()):
                    break
        finally:
            self._cache(False)
        return torch.stack(generated, dim=1)


def save_adapter(model, path, **metadata):
    state = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    payload = {"format_version": 1, "config": asdict(model.config),
               "decoder_config": vars(model.decoder.config), "adapter": state,
               "trainable_count": model.trainable_count, **metadata}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Expected a trained caption adapter checkpoint (format_version=1).")
    for key in ("config", "adapter", "decoder_config", "trainable_count"):
        if key not in payload:
            raise ValueError(f"Caption checkpoint is missing {key}.")
    return payload


def load_adapter(model, payload):
    if payload["config"] != asdict(model.config):
        raise ValueError("Caption checkpoint architecture does not match this model.")
    if payload["decoder_config"] != vars(model.decoder.config):
        raise ValueError("Caption checkpoint was trained with a different decoder config.")
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    actual = payload["adapter"]
    if set(expected) != set(actual):
        raise ValueError(f"Adapter keys differ: missing={sorted(set(expected)-set(actual))}, "
                         f"unexpected={sorted(set(actual)-set(expected))}")
    for name, parameter in expected.items():
        if not isinstance(actual[name], torch.Tensor) or actual[name].shape != parameter.shape:
            raise ValueError(f"Adapter tensor has the wrong shape: {name}")
    if payload["trainable_count"] != model.trainable_count:
        raise ValueError("Checkpoint trained-parameter count does not match the model.")
    model.load_state_dict(actual, strict=False)
