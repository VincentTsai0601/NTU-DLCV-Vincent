#!/usr/bin/env python3
"""# Add src to path to import our modules
sys.path.append('src')
try:
    from p2_semantic_segmentation_train import UNet, mask_to_class, class_to_mask
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
    print("✅ Successfully imported from p2_semantic_segmentation_train")
except ImportError as e:
    print(f"Import error from p2_semantic_segmentation_train: {e}")
    # Fallback to semantic_segmentation
    try:
        from semantic_segmentation import UNet, mask_to_class, class_to_mask
        import torchvision.models as models
        TORCHVISION_AVAILABLE = True
        print("✅ Successfully imported from semantic_segmentation")
    except ImportError as e2:
        print(f"Import error from semantic_segmentation: {e2}")
        try:
            from semantic_segmentation import UNet
            print("✅ Successfully imported UNet only")
            UNET_AVAILABLE = True
        except ImportError:
            print("❌ Failed to import UNet")
            UNET_AVAILABLE = False
        TORCHVISION_AVAILABLE = Falseation Inference Script for HW1

This script performs semantic segmentation inference on satellite images using trained models.
Compatible with hw1_2.sh script for automated testing.

Usage: python semantic_inference.py input_dir output_dir
Where:
- input_dir: directory containing images named 'xxxx_sat.jpg'
- output_dir: directory to save segmentation masks as 'xxxx_mask.png'

Author: GitHub Copilot
Date: September 2025
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import argparse
from torchvision import transforms

# Add src to path to import our modules
sys.path.append('src')
try:
    from p2_semantic_segmentation_train import UNet, mask_to_class, class_to_mask
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    # Try to import UNet directly if other imports fail
    try:
        from semantic_segmentation import UNet
        print("✅ Successfully imported UNet")
        UNET_AVAILABLE = True
    except ImportError:
        print("❌ Failed to import UNet")
        UNET_AVAILABLE = False
    TORCHVISION_AVAILABLE = False

# Color mapping for class to RGB conversion
CLASS_TO_COLOR = {
    0: (0, 255, 255),    # Urban land - Cyan
    1: (255, 255, 0),    # Agriculture land - Yellow
    2: (255, 0, 255),    # Rangeland - Purple
    3: (0, 255, 0),      # Forest land - Green
    4: (0, 0, 255),      # Water - Blue
    5: (255, 255, 255),  # Barren land - White
    6: (0, 0, 0),        # Unknown - Black
}

def load_model(checkpoint_path, model_type='deeplabv3'):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        model_type (str): Type of model ('deeplabv3' or 'unet')
        
    Returns:
        tuple: (model, device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model based on type
    if model_type == 'deeplabv3' and TORCHVISION_AVAILABLE:
        print("Loading DeepLabV3 model...")
        # Load pre-trained DeepLabV3+ with ResNet-50 backbone (same as training)
        model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        
        # Modify classifier heads for 7 classes (same as training)
        model.classifier[4] = torch.nn.Conv2d(256, 7, kernel_size=1)
        model.aux_classifier[4] = torch.nn.Conv2d(256, 7, kernel_size=1)
    else:
        # Fallback to UNet
        if model_type == 'deeplabv3' and not TORCHVISION_AVAILABLE:
            print("⚠️ DeepLabV3 requested but torchvision not available, using UNet")
        print("Loading UNet model...")
        
        # Check if UNet is available
        if 'UNET_AVAILABLE' in globals() and not UNET_AVAILABLE:
            raise ImportError("UNet class not available. Please check your imports.")
        
        # Use correct parameters based on which file we imported from
        try:
            # Try p2_semantic_segmentation_train parameters first
            model = UNet(n_classes=7, base_channels=64, drop_skip_layer=-1)
            print("Using p2_semantic_segmentation_train UNet parameters")
        except TypeError:
            # Fallback to semantic_segmentation parameters
            model = UNet(n_classes=7, base_channels=64)
            print("Using semantic_segmentation UNet parameters")
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loading model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint
        print("Loading model weights")
    
    # Filter out unexpected keys from the state_dict
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                print(f"⚠️ Skipping {key}: shape mismatch ({model_state_dict[key].shape} vs {value.shape})")
        else:
            print(f"⚠️ Skipping unexpected key: {key}")
    
    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"✅ Successfully loaded {len(filtered_state_dict)} parameters")
    
    model.to(device)
    model.eval()
    
    return model, device

def preprocess_image(image_path):
    """
    Preprocess image for model inference.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load and convert image
    image = Image.open(image_path).convert('RGB')
    
    # Convert to tensor (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    
    return image_tensor

def class_mask_to_rgb(class_mask):
    """
    Convert class index mask to RGB mask.
    
    Args:
        class_mask (numpy.ndarray): Class index mask
        
    Returns:
        numpy.ndarray: RGB mask
    """
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in CLASS_TO_COLOR.items():
        rgb_mask[class_mask == class_idx] = color
    
    return rgb_mask

def run_inference(model, device, image_tensor):
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        device: Computing device
        image_tensor: Preprocessed image tensor
        
    Returns:
        numpy.ndarray: Predicted class mask
    """
    # Add batch dimension and move to device
    image_input = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_input)
        
        # Handle different output formats
        if isinstance(output, dict):
            logits = output['out']
        else:
            logits = output
        
        # Get prediction
        pred_mask = torch.argmax(logits, dim=1).cpu().numpy()[0]
    
    return pred_mask

def process_images(input_dir, output_dir, model, device):
    """
    Process all images in the input directory.
    
    Args:
        input_dir (str): Input directory containing images
        output_dir (str): Output directory for segmentation masks
        model: Trained model
        device: Computing device
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all satellite images
    image_pattern = os.path.join(input_dir, "*_sat.jpg")
    image_paths = glob.glob(image_pattern)
    image_paths.sort()
    
    if not image_paths:
        print(f"No images found in {input_dir} with pattern '*_sat.jpg'")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        # Extract image ID from filename (e.g., "0001_sat.jpg" -> "0001")
        image_filename = os.path.basename(image_path)
        image_id = image_filename.replace("_sat.jpg", "")
        
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_path)
            
            # Run inference
            pred_mask = run_inference(model, device, image_tensor)
            
            # Convert class mask to RGB
            rgb_mask = class_mask_to_rgb(pred_mask)
            
            # Save segmentation mask
            output_filename = f"{image_id}_mask.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save as PNG
            mask_image = Image.fromarray(rgb_mask)
            mask_image.save(output_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"Processing completed. Results saved to: {output_dir}")

def main():
    """Main function for semantic segmentation inference."""
    
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python semantic_inference.py <input_dir> <output_dir>")
        print("  input_dir: directory containing images named 'xxxx_sat.jpg'")
        print("  output_dir: directory to save segmentation masks as 'xxxx_mask.png'")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Validate inputs
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Model checkpoint path (relative to script location)
    checkpoint_path = "src/p2_outuput/deeplabv3_clean_best_balanced.pth"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    print("🚀 Semantic Segmentation Inference")
    print("=" * 40)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model checkpoint: {checkpoint_path}")
    
    # Load model
    model, device = load_model(checkpoint_path, model_type='deeplabv3')
    
    # Process images
    process_images(input_dir, output_dir, model, device)
    
    print("🎉 Inference completed successfully!")

if __name__ == "__main__":
    main()
