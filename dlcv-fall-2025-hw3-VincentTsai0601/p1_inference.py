"""
Problem 1: POPE Hallucination Evaluation with Optional VCD

This script evaluates LLaVA's yes/no predictions on the POPE dataset.
It supports Visual Contrastive Decoding (VCD) to reduce object hallucinations.

VCD Formula:
    p_vcd(y | v, v', x) = softmax[(1+α) × logit(y|v,x) - α × logit(y|v',x)]
    where:
        - v: original image
        - v': contrastive (noisy) image
        - x: text input
        - α (cd_alpha): confidence decay parameter
        - β (cd_beta): contrastive weight parameter
"""

import argparse
import json
import os
import random
import zipfile
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers.generation.stopping_criteria import StoppingCriteriaList

# ============================================================================
# IMPORT LLaVA COMPONENTS
# ============================================================================
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# ============================================================================
# IMPORT VCD UTILITIES
# ============================================================================
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling


def enable_vcd_model_inputs(model) -> None:
    """Adapt this model instance to the VCD sampler without editing LLaVA.

    Explicit VCD parameters let Transformers validate generation kwargs.
    Both image branches delegate to LLaVA's original preparation method so
    token slicing, attention masks, and each branch's KV cache are retained.
    """
    original_prepare = model.prepare_inputs_for_generation

    def prepare_with_vcd(
        self, input_ids, images_cd=None, cd_alpha=None, cd_beta=None, **kwargs
    ):
        return original_prepare(input_ids, **kwargs)

    def prepare_contrastive(
        self, input_ids, images_cd=None, cd_alpha=None, cd_beta=None, **kwargs
    ):
        if images_cd is None:
            raise ValueError("VCD requires a contrastive image (images_cd).")
        kwargs["images"] = images_cd
        return original_prepare(input_ids, **kwargs)

    model.prepare_inputs_for_generation = MethodType(prepare_with_vcd, model)
    if not callable(getattr(model, "prepare_inputs_for_generation_cd", None)):
        model.prepare_inputs_for_generation_cd = MethodType(prepare_contrastive, model)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Problem 1 inference with optional VCD"
    )
    
    # ========== Required Arguments ==========
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        default="liuhaotian/llava-v1.5-7b",
        help="Path or repo id of pretrained LLaVA weights (e.g., liuhaotian/llava-v1.5-7b)",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        default="hw3_data/p1_data/val.json",
        help="Annotation json file (e.g., hw3_data/p1_data/val.json)",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        default="./hw3_data/p1_data/images/val",
        help="Directory containing validation images",
    )
    
    # ========== Model Configuration ==========
    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
        help="Base model path when loading delta weights",
    )
    parser.add_argument(
        "--conv_mode",
        type=str,
        default="llava_v1",
        help="Conversation template (e.g., llava_v1, llava_v0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (e.g., cuda:0, cuda:1, cpu)",
    )
    
    # ========== Generation Parameters ==========
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Maximum tokens to generate (for yes/no answers)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy, >0 = sample)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter",
    )
    
    # ========== VCD Parameters ==========
    parser.add_argument(
        "--use_cd",
        action="store_true",
        default=False,
        help="Enable Visual Contrastive Decoding (VCD)",
    )
    parser.add_argument(
        "--noise_step",
        type=int,
        default=500,
        help="DDPM noise step for VCD (0-999, higher = more noise)",
    )
    parser.add_argument(
        "--cd_alpha",
        type=float,
        default=1.0,
        help="Alpha hyperparameter for VCD (original image weight)",
    )
    parser.add_argument(
        "--cd_beta",
        type=float,
        default=0.1,
        help="Beta hyperparameter for VCD (contrastive image weight)",
    )
    
    # ========== Output and Misc ==========
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/p1_pred.json",
        help="Output path for predictions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_annotations(path: str) -> List[Dict[str, Any]]:
    """Load annotations from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question: str, conv_mode: str, uses_image_tokens: bool) -> str:
    """
    Build the prompt for the model.
    
    Args:
        question: The question to ask
        conv_mode: Conversation template (e.g., "llava_v1")
        uses_image_tokens: Whether to use <|im_start|> and <|im_end|> tokens
    
    Returns:
        Formatted prompt string
    """
    text = question.strip()
    
    # ========== Build image tokens ==========
    if uses_image_tokens:
        # Format: <|im_start|><image></image><|im_end|>
        prefix = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n"
    else:
        # Format: <image>
        prefix = f"{DEFAULT_IMAGE_TOKEN}\n"
    
    # ========== Add question and instruction ==========
    prompt_question = f"{prefix}{text}\nPlease answer this question with one word."
    
    # ========== Use conversation template ==========
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_question)  # User message
    conv.append_message(conv.roles[1], None)              # Assistant (empty)
    
    return conv.get_prompt()


def normalize_answer(text: str) -> str:
    """
    Extract yes/no answer from model output.
    
    Examples:
        "Yes, there is a cat" -> "yes"
        "No, I don't see a dog" -> "no"
        "yes" -> "yes"
    
    Args:
        text: Raw model output
    
    Returns:
        Normalized answer ("yes", "no", or original text)
    """
    cleaned = text.strip().lower()
    
    # Check if starts with yes/no
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"
    
    # Check if contains yes or no
    if "yes" in cleaned and "no" not in cleaned:
        return "yes"
    if "no" in cleaned:
        return "no"
    
    # Return first word if nothing matches
    return cleaned.split()[0] if cleaned else ""


def main() -> None:
    """Main inference loop."""
    # ========== Parse and setup ==========
    args = parse_args()
    set_seed(args.seed)
    
    # ========== Initialize VCD if enabled ==========
    if args.use_cd:
        print("[VCD ENABLED]")
        print(f"  noise_step : {args.noise_step}")
        print(f"  cd_alpha   : {args.cd_alpha}")
        print(f"  cd_beta    : {args.cd_beta}")
        print("  Patching transformers.generation with VCD sampling...")
        # This patches the sampling function in transformers to support VCD
        evolve_vcd_sampling()
    
    disable_torch_init()
    
    # ========== Select device ==========
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    print(f"Using device: {device_str}")
    
    # ========== Handle zipped models ==========
    model_path = os.path.expanduser(args.model_path)
    if os.path.isfile(model_path) and zipfile.is_zipfile(model_path):
        target_dir = Path(model_path).with_suffix("")
        if not target_dir.exists():
            print(f"Extracting model archive from {model_path} to {target_dir}...")
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(model_path) as zf:
                zf.extractall(target_dir)
        model_path = str(target_dir)
    
    # ========== Load model ==========
    print(f"Loading model from {model_path}...")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device_map=None,
        device=device_str,
    )
    model.eval()
    model.to(device=device)
    if args.use_cd:
        enable_vcd_model_inputs(model)
    print("Model loaded successfully!")
    
    # ========== Load annotations ==========
    annotations = load_annotations(args.annotation_file)
    print(f"Loaded {len(annotations)} annotations")
    
    # ========== Inference loop ==========
    predictions: List[Dict[str, Any]] = []
    num_correct = 0
    
    for ann in tqdm(annotations, desc="Running inference"):
        # ===== Extract data =====
        image_id = ann["image_source"]
        question = ann["question"]
        answer = ann.get("answer", "").strip().lower()
        
        # ===== Build prompt with proper image tokens =====
        uses_image_tokens = getattr(model.config, "mm_use_im_start_end", False)
        prompt = build_prompt(question, args.conv_mode, uses_image_tokens)
        
        # ===== Tokenize prompt with IMAGE_TOKEN_INDEX =====
        # This is crucial! tokenizer_image_token replaces <image> with IMAGE_TOKEN_INDEX
        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0)
        input_ids = input_ids.to(device=device)
        
        # ===== Load and process image =====
        image_path = Path(args.images_root) / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        
        # Extract pixel_values (shape: [1, 3, H, W]) and get first item
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.to(device=device, dtype=model.dtype)
        
        # ===== Create contrastive image if VCD enabled =====
        image_tensor_cd = None
        if args.use_cd:
            # Add diffusion noise to create contrastive image
            # This uses DDPM noise at a specific timestep
            image_tensor_cd = add_diffusion_noise(
                image_tensor.detach().cpu(),
                args.noise_step
            )
            image_tensor_cd = image_tensor_cd.to(device=device, dtype=model.dtype)
        
        # ===== Setup stopping criteria =====
        conv = conv_templates[args.conv_mode]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = (
            StoppingCriteriaList([KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)])
            if stop_str
            else None
        )
        
        # ===== Generate output =====
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor.unsqueeze(0),                          # Original image
                # images_cd=(
                #     image_tensor_cd.unsqueeze(0) if image_tensor_cd is not None else None
                # ),                                                          # Contrastive image (if VCD)
                # cd_alpha=args.cd_alpha,                                    # VCD: original image weight
                # cd_beta=args.cd_beta,                                      # VCD: contrastive image weight
                do_sample=args.use_cd,  # Enable sampling when using VCD
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                **({
                "images_cd": image_tensor_cd.unsqueeze(0),
                "cd_alpha": args.cd_alpha,
                "cd_beta": args.cd_beta,
                } if args.use_cd else {}),
            )
        
        
        # ===== Decode output =====
        # Extract only the generated tokens (skip input tokens)
        generated_tokens = output_ids[:, input_ids.shape[1]:]
        output_text = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0].strip()
        
        # Remove stopping string if present
        if stop_str and output_text.endswith(stop_str):
            output_text = output_text[: -len(stop_str)].strip()
        
        # ===== Normalize to yes/no =====
        normalized = normalize_answer(output_text)
        
        # Fallback logic if normalization fails
        if normalized not in {"yes", "no"}:
            lower = output_text.lower()
            if "yes" in lower and "no" not in lower:
                normalized = "yes"
            elif "no" in lower and "yes" not in lower:
                normalized = "no"
        
        # ===== Save prediction =====
        predictions.append(
            {
                "image_source": image_id,
                "question": question,
                "predict": normalized,
                # Uncomment to save raw output for debugging:
                # "raw_output": output_text,
            }
        )
        
        # ===== Track accuracy (if ground truth available) =====
        if normalized == answer:
            num_correct += 1
    
    # ========== Compute and print accuracy ==========
    accuracy = num_correct / len(annotations) if annotations else 0.0
    print(f"\n{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({num_correct}/{len(annotations)})")
    print(f"{'='*50}")
    
    # ========== Save predictions ==========
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
