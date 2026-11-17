#!/usr/bin/env python3
"""
Office-Home Dataset Inference Script for HW1

This script performs inference on the Office-Home dataset using trained ResNet50 models.
Compatible with hw1_1.sh script for automated testing.

Usage: python inference.py csv_file image_dir output_csv
Where:
- csv_file: path to the images csv file (e.g., hw1_hiddendata/p1_data/office/test.csv)
- image_dir: path to the folder containing images (e.g., hw1_hiddendata/p1_data/office/test/)
- output_csv: path of output .csv file (predicted labels) (e.g., output_p1/test_pred.csv)

Author: GitHub Copilot
Date: September 2025
"""

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


class OfficeHomeTestDataset(Dataset):
    """
    Custom Dataset class for Office-Home test inference.
    Loads images from CSV file for testing.
    """
    
    def __init__(self, csv_path, img_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            csv_path (str): Path to the CSV file containing image filenames
            img_dir (str): Directory containing the images
            transform (callable, optional): Optional transform to be applied on images
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # Read the CSV file
        self.data = pd.read_csv(csv_path)
        
        print(f"Loaded {len(self.data)} images for inference from {csv_path}")
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get an image at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, filename, id) where image is the transformed PIL image,
                   filename is the image filename, and id is the sample id
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get data from CSV
        row = self.data.iloc[idx]
        
        # Extract filename and id
        if 'filename' in row:
            filename = row['filename']
        elif 'image' in row:
            filename = row['image']
        else:
            # Assume the first column contains filenames
            filename = row.iloc[0]
        
        # Extract id if available
        if 'id' in row:
            sample_id = row['id']
        else:
            sample_id = idx
        
        # Construct full image path
        img_path = os.path.join(self.img_dir, filename)
        
        # Open and convert image to RGB
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a dummy black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image, filename, sample_id


def get_inference_transform():
    """
    Define image transformations for inference.
    
    Returns:
        transforms.Compose: The transformation pipeline for inference
    """
    # Inference transformations (same as validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform


def create_model(num_classes=65):
    """
    Create a ResNet50 model with a custom classifier head.
    
    Args:
        num_classes (int): Number of output classes (default: 65 for Office-Home)
        
    Returns:
        torch.nn.Module: ResNet50 model with custom classifier
    """
    # Load ResNet50 architecture compatible with DINO weights
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    
    # Replace the final fully connected layer
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)
    
    return model


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """
    Load model weights from a checkpoint file.
    
    Args:
        model (torch.nn.Module): The model to load weights into
        checkpoint_path (str): Path to the checkpoint file
        device (str): Device to load the model on
        
    Returns:
        dict: Checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise
    
    # Extract the state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    
    print(f"Checkpoint loaded successfully!")
    
    return checkpoint


def run_inference(model, dataloader, device):
    """
    Run inference on the dataset and return predictions.
    
    Args:
        model (torch.nn.Module): The trained model
        dataloader (DataLoader): Data loader containing the samples
        device (torch.device): Device to run inference on
        
    Returns:
        tuple: (predictions, filenames, ids) where predictions is a list
               of predicted class indices, filenames is a list of filenames,
               and ids is a list of sample ids
    """
    model.eval()
    predictions = []
    filenames = []
    ids = []
    
    print("Running inference...")
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Inference")
        for batch in pbar:
            images, batch_filenames, batch_ids = batch
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            filenames.extend(batch_filenames)
            ids.extend(batch_ids)
            
            pbar.set_postfix({'Processed': f'{len(predictions)}'})
    
    print(f"Inference completed on {len(predictions)} samples")
    return predictions, filenames, ids


def save_predictions(predictions, filenames, ids, output_path):
    """
    Save predictions to a CSV file in the required format.
    Format: 'id,filename,label' (NO space after the comma)
    
    Args:
        predictions (list): List of predicted class indices
        filenames (list): List of corresponding filenames (basename only)
        ids (list): List of sample ids
        output_path (str): Path to save the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create DataFrame with required format
    # Convert tensor IDs to integers if necessary
    processed_ids = []
    for id_val in ids:
        if hasattr(id_val, 'item'):  # If it's a tensor
            processed_ids.append(id_val.item())
        elif isinstance(id_val, str) and id_val.startswith('tensor('):
            # Extract number from tensor string format
            import re
            match = re.search(r'tensor\((\d+)\)', id_val)
            if match:
                processed_ids.append(int(match.group(1)))
            else:
                processed_ids.append(id_val)
        else:
            processed_ids.append(id_val)
    
    results_df = pd.DataFrame({
        'id': processed_ids,
        'filename': [os.path.basename(f) for f in filenames],  # Only basename, not full path
        'label': predictions
    })
    
    # Save to CSV with exact header format required
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(f"Output format: id,filename,label")
    print(f"First few predictions:")
    print(results_df.head())


def main():
    """Main function for inference - uses sys.argv for command line arguments."""
    
    # Check command line arguments
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python inference.py <csv_file> <image_dir> <output_csv> [checkpoint_path]")
        print("  csv_file: path to the images csv file")
        print("  image_dir: path to the folder containing images") 
        print("  output_csv: path of output .csv file (predicted labels)")
        print("  checkpoint_path: (optional) path to the model checkpoint file")
        print("                   (default: checkpoint_setting_C_best.pth)")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    image_dir = sys.argv[2] 
    output_csv = sys.argv[3]
    checkpoint_path = sys.argv[4] if len(sys.argv) == 5 else None
    
    # Validate inputs
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Handle checkpoint path
    if checkpoint_path:
        # Use provided checkpoint path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        print(f"Using provided checkpoint: {checkpoint_path}")
    else:
        # Use default checkpoint path (relative to script location)
        checkpoint_path = 'checkpoint_setting_C_best.pth'
        
        if not os.path.exists(checkpoint_path):
            # Fallback: search for Setting C checkpoint in various locations
            print(f"Default checkpoint not found at {checkpoint_path}, searching...")
            
            # Look for checkpoint in checkpoints directory
            checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
            
            # Look for Setting C checkpoint first (best performing model)
            possible_paths = [
                os.path.join(checkpoint_dir, 'checkpoint_setting_C_best.pth'),
                os.path.join(checkpoint_dir, 'setting_c', 'best_model.pth'),
                os.path.join(checkpoint_dir, 'setting_c', 'resnet50_setting_c_best.pth'),
                os.path.join(checkpoint_dir, 'resnet50_setting_c_best.pth'),
                os.path.join(checkpoint_dir, 'best_model.pth'),
                os.path.join('checkpoints', 'checkpoint_setting_C_best.pth'),
                os.path.join('checkpoints', 'setting_c', 'best_model.pth'),
                os.path.join('checkpoints', 'resnet50_setting_c_best.pth'),
                'checkpoints/checkpoint_setting_C_best.pth',
                'checkpoints/setting_c/best_model.pth',
                'checkpoints/resnet50_setting_c_best.pth'
            ]
            
            checkpoint_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoint file found. Please ensure Setting C model is trained and saved.")
        
        print(f"Using checkpoint: {checkpoint_path}")
    
    # Create model and load checkpoint
    print("Creating model...")
    model = create_model(num_classes=65)
    load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    
    # Get transformations
    transform = get_inference_transform()
    
    # Create dataset and dataloader
    dataset = OfficeHomeTestDataset(csv_file, image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, 
                          num_workers=4, pin_memory=True if device.type == 'cuda' else False)
    
    # Run inference
    predictions, filenames, ids = run_inference(model, dataloader, device)
    
    # Save predictions in required format
    save_predictions(predictions, filenames, ids, output_csv)
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()