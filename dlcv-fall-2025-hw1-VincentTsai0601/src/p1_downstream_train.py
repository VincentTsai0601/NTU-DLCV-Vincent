#!/usr/bin/env python3
"""
Office-Home Dataset Classification Script

This script performs image classification on the Office-Home dataset using ResNet50
with different experimental settings including training from scratch, fine-tuning,
and linear probing with both official DINO and custom DINO weights.

Author: GitHub Copilot
Date: September 2025
"""



import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import datetime
import pickle
import sys


class OfficeHomeDataset(Dataset):
    """
    Custom Dataset class for Office-Home dataset.
    
    This dataset loads images and their corresponding labels from CSV files
    and applies the specified transformations.
    """
    
    def __init__(self, csv_path, img_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            csv_path (str): Path to the CSV file containing image filenames and labels
            img_dir (str): Directory containing the images
            transform (callable, optional): Optional transform to be applied on images
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # Read the CSV file containing image filenames and labels
        self.data = pd.read_csv(csv_path)
        
        print(f"Loaded {len(self.data)} samples from {csv_path}")
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get an image and its label at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, label) where image is the transformed PIL image
                   and label is the integer class label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get filename and label from the CSV data
        filename = self.data.iloc[idx]['filename']
        label = self.data.iloc[idx]['label']
        
        # Construct full image path
        img_path = os.path.join(self.img_dir, filename)
        
        # Open and convert image to RGB (handles grayscale images)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms():
    """
    Define image transformations for training and validation.
    
    Returns:
        tuple: (train_transform, val_transform) containing the transformation pipelines
    """
    # Training transformations with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transformations without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


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
    # ResNet50 has 2048 features before the final layer
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)
    
    return model


def load_dino_weights(model, weights_path, strict=False):
    """
    Load DINO weights into the model backbone.
    
    Args:
        model (torch.nn.Module): The model to load weights into
        weights_path (str): Path to the weights file
        strict (bool): Whether to strictly enforce that the keys match
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    print(f"Loading weights from {weights_path}")
    
    try:
        # Try loading with weights_only=False first
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Failed to load with weights_only=False: {e}")
        try:
            # Fallback: try with pickle module compatibility
            import pickle
            import sys
            
            # Add compatibility for numpy._core
            if 'numpy._core' not in sys.modules:
                try:
                    import numpy.core as numpy_core
                    sys.modules['numpy._core'] = numpy_core
                    sys.modules['numpy._core.multiarray'] = numpy_core.multiarray
                except ImportError:
                    pass
            
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        except Exception as e2:
            print(f"Failed with compatibility fix: {e2}")
            # Last resort: try loading with pickle protocol compatibility
            try:
                with open(weights_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location='cpu', weights_only=False, pickle_module=pickle)
            except Exception as e3:
                raise RuntimeError(f"Could not load checkpoint: {e3}")
    
    # Extract the state dict (handle different checkpoint formats)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    elif 'student' in checkpoint:
        state_dict = checkpoint['student']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel training)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        elif key.startswith('backbone.'):
            new_key = key[9:]  # Remove 'backbone.' prefix  
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Load weights into the model (excluding the final fc layer)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() 
                      if k in model_dict and not k.startswith('fc')}
    
    print(f"Loading {len(pretrained_dict)} layers from pretrained weights")
    print(f"Model has {len(model_dict)} total layers")
    
    if len(pretrained_dict) == 0:
        print("Warning: No matching layers found!")
        print("Available keys in checkpoint:")
        print(list(new_state_dict.keys())[:10])  # Show first 10 keys
        print("Model keys:")
        print(list(model_dict.keys())[:10])  # Show first 10 keys
    
    model.load_state_dict(pretrained_dict, strict=False)


def freeze_backbone(model):
    """
    Freeze all parameters in the backbone (everything except the final fc layer).
    
    Args:
        model (torch.nn.Module): The model whose backbone should be frozen
    """
    print("Freezing backbone parameters...")
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if not name.startswith('fc'):  # Freeze everything except the final fc layer
            param.requires_grad = False
            frozen_params += 1
    
    print(f"Frozen {frozen_params}/{total_params} parameters")


def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, 
                      epochs, device, setting, save_dir=None):
    """
    Train and evaluate the model.
    
    Args:
        model (torch.nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (torch.optim.Optimizer): Optimizer for training
        criterion (torch.nn.Module): Loss function
        epochs (int): Number of training epochs
        device (torch.device): Device to run training on
        setting (str): Experimental setting ('A', 'B', 'C', 'D', 'E')
        save_dir (str, optional): Directory to save model checkpoints
    
    Returns:
        dict: Dictionary containing training results and metrics
    """
    best_val_acc = 0.0
    model = model.to(device)
    
    # Initialize results tracking
    epoch_results = []
    best_model_path = None
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nStarting training for Setting {setting}...")
    print(f"Training on device: {device}")
    print(f"Training for {epochs} epochs")
    if save_dir:
        print(f"Checkpoints will be saved to: {save_dir}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_acc = 100.0 * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_acc:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_acc = 100.0 * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{val_acc:.2f}%'
                })
        
        # Calculate epoch metrics
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Store epoch results
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        }
        epoch_results.append(epoch_result)
        
        # Save checkpoint if save_dir is specified
        if save_dir:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'setting': setting
            }
            
            # Save latest checkpoint
            latest_path = os.path.join(save_dir, f'checkpoint_setting_{setting}_latest.pth')
            torch.save(checkpoint, latest_path)
            
            # Save first epoch checkpoint
            if epoch == 0:
                first_epoch_path = os.path.join(save_dir, f'checkpoint_setting_{setting}_first_epoch.pth')
                torch.save(checkpoint, first_epoch_path)
                print(f"First epoch checkpoint saved: {first_epoch_path}")
        
        # Update best validation accuracy and save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_dir:
                best_model_path = os.path.join(save_dir, f'checkpoint_setting_{setting}_best.pth')
                torch.save(checkpoint, best_model_path)
                print(f"New best model saved: {best_model_path}")
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{epochs}]:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Best Val Acc: {best_val_acc:.2f}%')
        print('-' * 60)
    
    # Save final checkpoint
    if save_dir:
        final_checkpoint = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'setting': setting,
            'best_val_acc': best_val_acc
        }
        final_path = os.path.join(save_dir, f'checkpoint_setting_{setting}_final.pth')
        torch.save(final_checkpoint, final_path)
        print(f"Final checkpoint saved: {final_path}")
    
    # Return comprehensive results
    results = {
        'setting': setting,
        'best_val_acc': best_val_acc,
        'final_train_acc': train_acc,
        'final_val_acc': val_acc,
        'epochs': epochs,
        'epoch_details': epoch_results,
        'best_model_path': best_model_path,
        'final_model_path': final_path if save_dir else None
    }
    
    return results


def save_results(results, output_file='office_home_results.json'):
    """
    Save experiment results to a JSON file.
    
    Args:
        results (dict): Results dictionary from training
        output_file (str): Path to output file
    """
    # Add timestamp and additional info
    results['timestamp'] = datetime.datetime.now().isoformat()
    results['device'] = str(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
    
    # Load existing results if file exists
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_results = []
    else:
        all_results = []
    
    # Add new results
    all_results.append(results)
    
    # Save updated results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def print_results_summary(output_file='office_home_results.json'):
    """
    Print a summary of all experiment results.
    
    Args:
        output_file (str): Path to results file
    """
    if not os.path.exists(output_file):
        print("No results file found.")
        return
    
    try:
        with open(output_file, 'r') as f:
            all_results = json.load(f)
        
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        # Create summary table
        print(f"{'Setting':<10} {'Best Val Acc':<15} {'Final Val Acc':<15} {'Epochs':<8} {'Timestamp':<20}")
        print("-" * 80)
        
        for result in all_results:
            setting = result.get('setting', 'N/A')
            best_acc = result.get('best_val_acc', 0)
            final_acc = result.get('final_val_acc', 0)
            epochs = result.get('epochs', 0)
            timestamp = result.get('timestamp', 'N/A')[:19]  # Remove milliseconds
            
            print(f"{setting:<10} {best_acc:<15.2f} {final_acc:<15.2f} {epochs:<8} {timestamp:<20}")
        
        print("-" * 80)
        print(f"Total experiments: {len(all_results)}")
        
        # Find best overall result
        if all_results:
            best_experiment = max(all_results, key=lambda x: x.get('best_val_acc', 0))
            print(f"Best overall result: Setting {best_experiment.get('setting')} with {best_experiment.get('best_val_acc', 0):.2f}% validation accuracy")
        
        print("="*80)
        
    except json.JSONDecodeError:
        print("Error reading results file.")


def main():
    """Main function to handle different experimental settings."""
    parser = argparse.ArgumentParser(description='Office-Home Classification')
    
    # Required arguments
    parser.add_argument('--setting', type=str, required=True, 
                       choices=['A', 'B', 'C', 'D', 'E'],
                       help='Experimental setting to run')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to the office dataset directory')
    
    # Optional arguments for DINO weights
    parser.add_argument('--official_dino_weights', type=str,
                       help='Path to official DINO model weights')
    parser.add_argument('--my_dino_weights', type=str,
                       help='Path to custom DINO model weights')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    # Results saving
    parser.add_argument('--output_file', type=str, default='office_home_results.json',
                       help='File to save experiment results')
    parser.add_argument('--show_summary', action='store_true',
                       help='Show summary of all previous results')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Show summary if requested
    if args.show_summary:
        print_results_summary(args.output_file)
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # Define paths
    train_csv = os.path.join(args.data_dir, 'train.csv')
    val_csv = os.path.join(args.data_dir, 'val.csv')
    train_img_dir = os.path.join(args.data_dir, 'train')
    val_img_dir = os.path.join(args.data_dir, 'val')
    
    # Validate required files
    for path in [train_csv, val_csv, train_img_dir, val_img_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")
    
    print(f"Running Setting {args.setting}")
    print(f"Data directory: {args.data_dir}")
    
    # Get transformations
    train_transform, val_transform = get_transforms()
    
    # Create datasets and data loaders
    train_dataset = OfficeHomeDataset(train_csv, train_img_dir, train_transform)
    val_dataset = OfficeHomeDataset(val_csv, val_img_dir, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers, 
                             pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=True if device.type == 'cuda' else False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = create_model(num_classes=65)
    print("Model created with ResNet50 backbone")
    
    # Configure model based on experimental setting
    if args.setting == 'A':
        # Setting A: Train from scratch
        print("Setting A: Training from scratch (random initialization)")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    elif args.setting == 'B':
        # Setting B: Fine-tune official DINO
        print("Setting B: Fine-tuning with official DINO weights")
        if not args.official_dino_weights:
            raise ValueError("--official_dino_weights is required for Setting B")
        load_dino_weights(model, args.official_dino_weights)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    elif args.setting == 'C':
        # Setting C: Fine-tune my DINO
        print("Setting C: Fine-tuning with custom DINO weights")
        if not args.my_dino_weights:
            raise ValueError("--my_dino_weights is required for Setting C")
        load_dino_weights(model, args.my_dino_weights)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    elif args.setting == 'D':
        # Setting D: Linear probe official DINO
        print("Setting D: Linear probing with official DINO weights")
        if not args.official_dino_weights:
            raise ValueError("--official_dino_weights is required for Setting D")
        load_dino_weights(model, args.official_dino_weights)
        freeze_backbone(model)
        # Only optimize the classifier head
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
        
    elif args.setting == 'E':
        # Setting E: Linear probe my DINO
        print("Setting E: Linear probing with custom DINO weights")
        if not args.my_dino_weights:
            raise ValueError("--my_dino_weights is required for Setting E")
        load_dino_weights(model, args.my_dino_weights)
        freeze_backbone(model)
        # Only optimize the classifier head
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Train and evaluate
    results = train_and_evaluate(
        model, train_loader, val_loader, optimizer, criterion, 
        args.epochs, device, args.setting, args.save_dir
    )
    
    # Add experiment configuration to results
    results.update({
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_workers': args.num_workers,
        'official_dino_weights': args.official_dino_weights,
        'my_dino_weights': args.my_dino_weights,
        'trainable_params': trainable_params,
        'total_params': total_params
    })
    
    # Save results to file
    save_results(results, args.output_file)
    
    # Print final results
    print("\n" + "="*60)
    print(f"FINAL RESULTS - Setting {args.setting}")
    print(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
    print(f"Final Validation Accuracy: {results['final_val_acc']:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("="*60)
    
    # Show current summary
    print_results_summary(args.output_file)


if __name__ == "__main__":
    main() 