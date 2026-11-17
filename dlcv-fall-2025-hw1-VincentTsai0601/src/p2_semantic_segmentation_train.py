#!/usr/bin/env python3
"""
Clean Semantic Segmentation Script for Satellite Images
Only using non-transformer architectures: U-Net and ResNet-UNet

This script implements improved U-Net models for semantic segmentation
without any transformer components - only CNN architectures.

Author: GitHub Copilot
Date: September 2025
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Color to class mapping for the 7 classes
COLOR_TO_CLASS = {
    (0, 255, 255): 0,    # Urban land
    (255, 255, 0): 1,    # Agriculture land
    (255, 0, 255): 2,    # Rangeland
    (0, 255, 0): 3,      # Forest land
    (0, 0, 255): 4,      # Water
    (255, 255, 255): 5,  # Barren land
    (0, 0, 0): 6,        # Unknown
}

# Class to color mapping (inverse of above)
CLASS_TO_COLOR = {v: k for k, v in COLOR_TO_CLASS.items()}

def mask_to_class(mask_rgb):
    """Convert RGB mask to class index tensor."""
    h, w = mask_rgb.shape[:2]
    mask_class = np.zeros((h, w), dtype=np.int64)
    
    for color, class_idx in COLOR_TO_CLASS.items():
        color_mask = np.all(mask_rgb == color, axis=2)
        mask_class[color_mask] = class_idx
    
    return torch.from_numpy(mask_class).long()

def class_to_mask(class_tensor):
    """Convert class index tensor to RGB mask."""
    if isinstance(class_tensor, torch.Tensor):
        class_tensor = class_tensor.cpu().numpy()
    
    h, w = class_tensor.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in CLASS_TO_COLOR.items():
        mask_rgb[class_tensor == class_idx] = color
    
    return mask_rgb

class SegmentationTransform:
    """Custom transform that applies the same random transformations to both image and mask."""
    
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, image, mask=None):
        image_np = np.array(image)
        
        if mask is not None:
            mask_np = np.array(mask)
            
            # Apply random horizontal flip if training
            if self.is_training and np.random.random() > 0.5:
                image_np = np.fliplr(image_np)
                mask_np = np.fliplr(mask_np)
            
            # Apply random vertical flip if training
            if self.is_training and np.random.random() > 0.5:
                image_np = np.flipud(image_np)
                mask_np = np.flipud(mask_np)
            
            image = Image.fromarray(image_np)
            mask = Image.fromarray(mask_np)
            
            image_tensor = self.to_tensor(image)
            mask_tensor = torch.from_numpy(np.array(mask))
            
            image_tensor = self.normalize(image_tensor)
            mask_class = mask_to_class(np.array(mask))
            
            return image_tensor, mask_class
        else:
            image = Image.fromarray(image_np)
            image_tensor = self.to_tensor(image)
            image_tensor = self.normalize(image_tensor)
            return image_tensor

class SatelliteDataset(Dataset):
    """Custom Dataset for satellite image segmentation."""
    
    def __init__(self, data_dir, transform=None, is_training=True):
        self.data_dir = data_dir
        self.transform = transform if transform else SegmentationTransform(is_training)
        self.is_training = is_training
        
        self.image_paths = []
        self.mask_paths = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('_sat.jpg'):
                image_path = os.path.join(data_dir, filename)
                mask_path = os.path.join(data_dir, filename.replace('_sat.jpg', '_mask.png'))
                
                if os.path.exists(mask_path):
                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)
        
        print(f"Found {len(self.image_paths)} image-mask pairs in {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.is_training or os.path.exists(self.mask_paths[idx]):
            mask = Image.open(self.mask_paths[idx]).convert('RGB')
            image_tensor, mask_tensor = self.transform(image, mask)
            return image_tensor, mask_tensor
        else:
            image_tensor = self.transform(image)
            return image_tensor

class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm and Dropout."""
    
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling - CNN-based multi-scale feature extraction."""
    
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # Different dilation rates for multi-scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        
        # Final conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        size = x.shape[-2:]
        
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=True)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.final_conv(x)
        
        return x

class UNet(nn.Module):
    """
    Improved U-Net for Skip Connection Ablation Study
    - Pure CNN architecture, no transformers
    - Supports selective skip connection dropping for ablation studies
    """
    
    def __init__(self, n_classes=7, base_channels=64, drop_skip_layer=-1):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.drop_skip_layer = drop_skip_layer  # For backward compatibility
        
        # Use deeper feature maps
        c1, c2, c3, c4, c5 = base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16
        
        # Encoder (down-sampling path)
        self.inc = DoubleConv(3, c1, dropout_p=0.0)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2, dropout_p=0.1))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3, dropout_p=0.1))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4, dropout_p=0.2))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c4, c5, dropout_p=0.2))
        
        # Decoder upsampling layers
        self.up1 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        
        # Decoder processing layers - WITH skip connections
        self.up_conv1_with_skip = DoubleConv(c5, c4, dropout_p=0.2)  # c4 + c4 = c5 input
        self.up_conv2_with_skip = DoubleConv(c4, c3, dropout_p=0.1)  # c3 + c3 = c4 input
        self.up_conv3_with_skip = DoubleConv(c3, c2, dropout_p=0.1)  # c2 + c2 = c3 input
        self.up_conv4_with_skip = DoubleConv(c2, c1, dropout_p=0.0)  # c1 + c1 = c2 input
        
        # Decoder processing layers - WITHOUT skip connections (NO-SKIP versions)
        self.up_conv1_no_skip = DoubleConv(c4, c4, dropout_p=0.2)    # only c4 input (upsampled)
        self.up_conv2_no_skip = DoubleConv(c3, c3, dropout_p=0.1)    # only c3 input (upsampled)
        self.up_conv3_no_skip = DoubleConv(c2, c2, dropout_p=0.1)    # only c2 input (upsampled)
        self.up_conv4_no_skip = DoubleConv(c1, c1, dropout_p=0.0)    # only c1 input (upsampled)
        
        # Final classifier
        self.outc = nn.Sequential(
            nn.Conv2d(c1, c1//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1//2, n_classes, kernel_size=1)
        )
        
        # Deep supervision outputs
        self.deep_supervision = True
        if self.deep_supervision:
            self.out1 = nn.Conv2d(c4, n_classes, kernel_size=1)
            self.out2 = nn.Conv2d(c3, n_classes, kernel_size=1)
            self.out3 = nn.Conv2d(c2, n_classes, kernel_size=1)
    
    def forward(self, x, drop_skip_index=None):
        """
        Forward pass with optional skip connection dropping
        
        Args:
            x: Input tensor
            drop_skip_index: Which skip connection to drop (0-3) or None for all active
                           - 3: drop deepest skip connection (x4)
                           - 2: drop deep skip connection (x3)  
                           - 1: drop shallow skip connection (x2)
                           - 0: drop shallowest skip connection (x1)
                           - None: use all skip connections
        """
        # Use drop_skip_layer if drop_skip_index is not provided (backward compatibility)
        if drop_skip_index is None and hasattr(self, 'drop_skip_layer') and self.drop_skip_layer >= 0:
            # Convert old indexing (0=deepest) to new indexing (3=deepest)
            drop_skip_index = 3 - self.drop_skip_layer
        
        # Encoder path
        x1 = self.inc(x)      # c1 channels (shallowest features)
        x2 = self.down1(x1)   # c2 channels
        x3 = self.down2(x2)   # c3 channels
        x4 = self.down3(x3)   # c4 channels (deepest features)
        x5 = self.down4(x4)   # c5 channels (bottleneck)
        
        # Decoder Stage 1: Process deepest skip connection (x4)
        d4 = self.up1(x5)  # Upsample from c5 to c4
        if drop_skip_index == 3:
            # Drop deepest skip connection - only use upsampled features
            d4 = self.up_conv1_no_skip(d4)
        else:
            # Use skip connection - concatenate x4 with upsampled features
            d4 = torch.cat([x4, d4], dim=1)  # c4 + c4 = c5
            d4 = self.up_conv1_with_skip(d4)
        
        # Decoder Stage 2: Process deep skip connection (x3)
        d3 = self.up2(d4)  # Upsample from c4 to c3
        if drop_skip_index == 2:
            # Drop deep skip connection - only use upsampled features
            d3 = self.up_conv2_no_skip(d3)
        else:
            # Use skip connection - concatenate x3 with upsampled features
            d3 = torch.cat([x3, d3], dim=1)  # c3 + c3 = c4
            d3 = self.up_conv2_with_skip(d3)
        
        # Decoder Stage 3: Process shallow skip connection (x2)
        d2 = self.up3(d3)  # Upsample from c3 to c2
        if drop_skip_index == 1:
            # Drop shallow skip connection - only use upsampled features
            d2 = self.up_conv3_no_skip(d2)
        else:
            # Use skip connection - concatenate x2 with upsampled features
            d2 = torch.cat([x2, d2], dim=1)  # c2 + c2 = c3
            d2 = self.up_conv3_with_skip(d2)
        
        # Decoder Stage 4: Process shallowest skip connection (x1)
        d1 = self.up4(d2)  # Upsample from c2 to c1
        if drop_skip_index == 0:
            # Drop shallowest skip connection - only use upsampled features
            d1 = self.up_conv4_no_skip(d1)
        else:
            # Use skip connection - concatenate x1 with upsampled features
            d1 = torch.cat([x1, d1], dim=1)  # c1 + c1 = c2
            d1 = self.up_conv4_with_skip(d1)
        
        # Main output
        logits = self.outc(d1)
        
        # Deep supervision outputs (only during training)
        if self.training and self.deep_supervision:
            out1 = F.interpolate(self.out1(d4), scale_factor=8, mode='bilinear', align_corners=True)
            out2 = F.interpolate(self.out2(d3), scale_factor=4, mode='bilinear', align_corners=True)
            out3 = F.interpolate(self.out3(d2), scale_factor=2, mode='bilinear', align_corners=True)
            
            return logits, [out1, out2, out3]
        
        return logits


class UNetWithResNetEncoder(nn.Module):
    """U-Net with pre-trained ResNet encoder + ASPP - Pure CNN, no transformers."""
    
    def __init__(self, n_classes=7, encoder_name='resnet50'):
        super(UNetWithResNetEncoder, self).__init__()
        self.n_classes = n_classes
        
        # Load pre-trained ResNet
        if encoder_name == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        
        self.encoder_channels = encoder_channels
        
        # Extract encoder layers
        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.encoder1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4
        
        # ASPP module for the bottleneck
        self.aspp = ASPP(encoder_channels[4], encoder_channels[4]//2)
        
        # Decoder blocks
        self.upconv4 = nn.ConvTranspose2d(encoder_channels[4]//2, encoder_channels[3], kernel_size=2, stride=2)
        self.decoder4 = self._make_decoder_block(encoder_channels[3] + encoder_channels[3], encoder_channels[3])
        
        self.upconv3 = nn.ConvTranspose2d(encoder_channels[3], encoder_channels[2], kernel_size=2, stride=2)
        self.decoder3 = self._make_decoder_block(encoder_channels[2] + encoder_channels[2], encoder_channels[2])
        
        self.upconv2 = nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], kernel_size=2, stride=2)
        self.decoder2 = self._make_decoder_block(encoder_channels[1] + encoder_channels[1], encoder_channels[1])
        
        self.upconv1 = nn.ConvTranspose2d(encoder_channels[1], encoder_channels[0], kernel_size=2, stride=2)
        self.decoder1 = self._make_decoder_block(encoder_channels[0] + encoder_channels[0], encoder_channels[0])
        
        # Final upsampling and classifier
        self.final_upconv = nn.ConvTranspose2d(encoder_channels[0], 64, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )
        
        # Deep supervision
        self.deep_supervision = True
        if self.deep_supervision:
            self.aux_classifier1 = nn.Conv2d(encoder_channels[3], n_classes, kernel_size=1)
            self.aux_classifier2 = nn.Conv2d(encoder_channels[2], n_classes, kernel_size=1)
            self.aux_classifier3 = nn.Conv2d(encoder_channels[1], n_classes, kernel_size=1)
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        e0 = self.encoder0(x)   # 64 channels, 1/2 resolution
        e1 = self.encoder1(e0)  # 256/64 channels, 1/4 resolution
        e2 = self.encoder2(e1)  # 512/128 channels, 1/8 resolution
        e3 = self.encoder3(e2)  # 1024/256 channels, 1/16 resolution
        e4 = self.encoder4(e3)  # 2048/512 channels, 1/32 resolution
        
        # Apply ASPP to bottleneck features
        e4_aspp = self.aspp(e4)
        
        # Decoder path with skip connections
        d4 = self.upconv4(e4_aspp)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.decoder1(d1)
        
        # Final upsampling to original resolution
        d0 = self.final_upconv(d1)
        logits = self.final_conv(d0)
        
        # Deep supervision outputs (only during training)
        if self.training and self.deep_supervision:
            aux1 = F.interpolate(self.aux_classifier1(d4), scale_factor=16, mode='bilinear', align_corners=True)
            aux2 = F.interpolate(self.aux_classifier2(d3), scale_factor=8, mode='bilinear', align_corners=True)
            aux3 = F.interpolate(self.aux_classifier3(d2), scale_factor=4, mode='bilinear', align_corners=True)
            
            return logits, [aux1, aux2, aux3]
        
        return logits

def calculate_class_weights(data_dir, num_classes=7):
    """
    Calculate class weights based on frequency in training data.
    
    Args:
        data_dir (str): Path to training data directory
        num_classes (int): Number of classes
        
    Returns:
        torch.Tensor: Class weights for loss function
    """
    class_counts = np.zeros(num_classes)
    total_pixels = 0
    
    print("Calculating class weights from training data...")
    
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith('_mask.png'):
            mask_path = os.path.join(data_dir, filename)
            mask = Image.open(mask_path).convert('RGB')
            mask_np = np.array(mask)
            
            # Convert to class indices
            mask_class = mask_to_class(mask_np)
            
            # Count pixels for each class
            for cls in range(num_classes):
                class_counts[cls] += (mask_class == cls).sum()
            
            total_pixels += mask_class.numel()
    
    # Calculate weights (inverse frequency)
    class_freq = class_counts / total_pixels
    class_weights = 1.0 / (class_freq + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"Class frequencies: {class_freq}")
    print(f"Class weights: {class_weights}")
    
    return torch.FloatTensor(class_weights)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha (float): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        weight (torch.Tensor): Class weights
        ignore_index (int): Index to ignore
    """
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, 
                                  weight=self.weight, 
                                  ignore_index=self.ignore_index, 
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """
    Combined loss function that switches between Focal Loss and Weighted CE during training.
    """
    def __init__(self, num_classes=7, ignore_index=6):
        super(CombinedLoss, self).__init__()
        
        # Calculate class weights based on inverse frequency (class 2 needs more weight)
        # These weights are tuned based on the dataset characteristics
        self.class_weights = torch.tensor([
            1.0,   # Urban land
            1.2,   # Agriculture land  
            5.0,   # Rangeland (most problematic class)
            1.5,   # Forest land
            2.0,   # Water
            1.8,   # Barren land
            1.0    # Unknown (ignored class, but weight needed for tensor size)
        ], dtype=torch.float32)
        
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0, ignore_index=ignore_index)
        self.weighted_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets, epoch=0, total_epochs=50):
        """
        Dynamic loss switching based on training progress.
        """
        # Move class weights to same device as inputs
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
            self.weighted_ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=self.ignore_index)
        
        # Dynamic loss strategy
        if epoch < total_epochs * 0.3:  # First 30% - Focal Loss for hard examples
            return self.focal_loss(inputs, targets)
        elif epoch < total_epochs * 0.7:  # Middle 40% - Weighted CE for balance
            return self.weighted_ce(inputs, targets)
        else:  # Final 30% - Continue weighted for stability
            return self.weighted_ce(inputs, targets)

def calculate_miou(pred, target, num_classes=6, ignore_index=None):
    """Calculate mean Intersection over Union (mIoU) using the same method as mean_iou_evaluate.py"""
    mean_iou = 0
    ious = []
    
    for i in range(num_classes):
        tp_fp = (pred == i).sum().float()  # Total predicted as class i
        tp_fn = (target == i).sum().float()  # Total ground truth as class i
        tp = ((pred == i) & (target == i)).sum().float()  # True positives
        
        if tp_fp + tp_fn - tp == 0:
            iou = 1.0 if tp_fp == 0 and tp_fn == 0 else 0.0
        else:
            iou = tp / (tp_fp + tp_fn - tp)
        
        ious.append(iou.item())
        mean_iou += iou.item() / num_classes
    
    return {
        'class_ious': ious,
        'miou': mean_iou
    }

def train_epoch(model, dataloader, criterion, optimizer, device, epoch=0, total_epochs=50):
    """Train for one epoch with deep supervision support and dynamic loss."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Handle different model output formats
        if isinstance(outputs, dict):  # DeepLabV3+ returns OrderedDict during training
            main_output = outputs['out']
            aux_output = outputs.get('aux', None)
            
            # Main loss with dynamic strategy
            if hasattr(criterion, 'forward') and 'epoch' in criterion.forward.__code__.co_varnames:
                loss = criterion(main_output, masks, epoch, total_epochs)
            else:
                loss = criterion(main_output, masks)
            
            # Auxiliary loss with reduced weight (if available)
            if aux_output is not None:
                aux_criterion = nn.CrossEntropyLoss(ignore_index=6)
                loss += 0.4 * aux_criterion(aux_output, masks)
                
        elif isinstance(outputs, tuple):  # U-Net models with deep supervision
            main_output, aux_outputs = outputs
            # Main loss with dynamic strategy
            if hasattr(criterion, 'forward') and 'epoch' in criterion.forward.__code__.co_varnames:
                loss = criterion(main_output, masks, epoch, total_epochs)
            else:
                loss = criterion(main_output, masks)
            
            # Auxiliary losses with reduced weight (use simpler loss for aux)
            aux_criterion = nn.CrossEntropyLoss(ignore_index=6)
            for aux_output in aux_outputs:
                loss += 0.4 * aux_criterion(aux_output, masks)
        else:  # Single output
            if hasattr(criterion, 'forward') and 'epoch' in criterion.forward.__code__.co_varnames:
                loss = criterion(outputs, masks, epoch, total_epochs)
            else:
                loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle different model output formats (only use main output for validation)
            if isinstance(outputs, dict):  # DeepLabV3+
                logits = outputs['out']
            elif isinstance(outputs, tuple):  # U-Net with deep supervision
                logits = outputs[0]
            else:  # Single output
                logits = outputs
                
            loss = criterion(logits, masks)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate mIoU
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    miou_results = calculate_miou(all_preds, all_targets)
    
    return total_loss / len(dataloader), miou_results

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, output_dir, model_name):
    """Train the model with enhanced class imbalance handling."""
    os.makedirs(output_dir, exist_ok=True)
    
    best_miou = 0
    train_losses = []
    val_losses = []
    val_mious = []
    learning_rates = []
    
    # Setup validation criterion based on training criterion type
    if isinstance(criterion, nn.CrossEntropyLoss):
        # For basic UNet: use same basic CrossEntropyLoss for validation
        val_criterion = nn.CrossEntropyLoss(ignore_index=6)
        print("🔍 Validation using basic CrossEntropyLoss (consistent with training)")
    else:
        # For enhanced models: use weighted CE for validation
        val_criterion = nn.CrossEntropyLoss(ignore_index=6)
        if hasattr(criterion, 'class_weights'):
            # Move class weights to device before creating validation criterion
            class_weights_device = criterion.class_weights.to(device)
            val_criterion = nn.CrossEntropyLoss(weight=class_weights_device, ignore_index=6)
        print("🔍 Validation using weighted CrossEntropyLoss")
    
    # Save initial checkpoint
    checkpoint_path = os.path.join(output_dir, f'{model_name}_epoch_0.pth')
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    print(f"Saved initial checkpoint: {checkpoint_path}")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Display current loss strategy
        if hasattr(criterion, 'forward') and 'epoch' in criterion.forward.__code__.co_varnames:
            if epoch < num_epochs * 0.3:
                loss_type = "🎯 Focal Loss (hard example mining)"
            elif epoch < num_epochs * 0.7:
                loss_type = "⚖️  Weighted CE (class balancing)" 
            else:
                loss_type = "🔄 Weighted CE (stable training)"
            print(f'Loss Strategy: {loss_type}')
        
        print('-' * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        train_losses.append(train_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Validate
        val_loss, miou_results = validate_epoch(model, val_loader, val_criterion, device)
        val_losses.append(val_loss)
        val_mious.append(miou_results['miou'])
        
        # Step scheduler
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val mIoU: {miou_results["miou"]:.4f}')
        print(f'Class IoUs: {[f"{iou:.3f}" for iou in miou_results["class_ious"]]}')
        
        # Special monitoring for class 2 (Rangeland)
        class_2_iou = miou_results["class_ious"][2]
        if class_2_iou > 0.4:
            print(f"🎉 Excellent! Class 2 (Rangeland) IoU: {class_2_iou:.3f}")
        elif class_2_iou > 0.25:
            print(f"✅ Good progress! Class 2 (Rangeland) IoU: {class_2_iou:.3f}")
        elif class_2_iou > 0.15:
            print(f"📈 Improving! Class 2 (Rangeland) IoU: {class_2_iou:.3f}")
        else:
            print(f"⚠️  Still low: Class 2 (Rangeland) IoU: {class_2_iou:.3f}")
        
        # Save checkpoint if best model
        if miou_results['miou'] > best_miou:
            best_miou = miou_results['miou']
            checkpoint_path = os.path.join(output_dir, f'{model_name}_best_balanced.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'miou': best_miou,
                'class_ious': miou_results["class_ious"],
                'loss_strategy': 'CombinedLoss' if hasattr(criterion, 'class_weights') else 'Standard'
            }, checkpoint_path)
            print(f"💾 NEW BEST MODEL! mIoU: {best_miou:.4f} - Saved: {checkpoint_path}")
        
        # Save middle checkpoint
        if epoch == num_epochs // 2:
            checkpoint_path = os.path.join(output_dir, f'{model_name}_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'class_ious': miou_results["class_ious"],
            }, checkpoint_path)
            print(f"Saved middle checkpoint: {checkpoint_path}")
    
    # Save final checkpoint
    checkpoint_path = os.path.join(output_dir, f'{model_name}_final.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_class_ious': miou_results["class_ious"],
    }, checkpoint_path)
    print(f"Saved final checkpoint: {checkpoint_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_mious': val_mious,
        'learning_rates': learning_rates
    }
    
    # Save detailed training history
    with open(os.path.join(output_dir, f'{model_name}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create and save loss curves visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(2, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Validation mIoU
    plt.subplot(2, 3, 2)
    plt.plot(range(1, len(val_mious) + 1), val_mious, 'g-', label='Validation mIoU', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Validation mIoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    plt.subplot(2, 3, 3)
    plt.plot(range(1, len(learning_rates) + 1), learning_rates, 'm-', label='Learning Rate', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 4: Loss Comparison (log scale)
    plt.subplot(2, 3, 4)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Curves (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 5: Training Progress Summary
    plt.subplot(2, 3, 5)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', alpha=0.7, label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', alpha=0.7, label='Val Loss')
    # Normalize mIoU to loss scale for visualization
    normalized_miou = [m * max(train_losses) for m in val_mious]
    plt.plot(epochs, normalized_miou, 'g-', alpha=0.7, label='Val mIoU (scaled)')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Progress Overview')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Skip Connection Info (if applicable)
    plt.subplot(2, 3, 6)
    if hasattr(model, 'drop_skip_layer') and model.drop_skip_layer >= 0:
        layer_names = ['Deepest\n(Bottleneck)', 'Deep', 'Shallow', 'Shallowest\n(Surface)']
        y_pos = [0, 1, 2, 3]
        colors = ['red' if i == model.drop_skip_layer else 'green' for i in range(4)]
        plt.barh(y_pos, [1, 1, 1, 1], color=colors, alpha=0.7)
        plt.yticks(y_pos, layer_names)
        plt.xlabel('Status')
        plt.title(f'Skip Connections\n(Layer {model.drop_skip_layer} DROPPED)')
        plt.xlim(0, 1.2)
        # Add text annotations
        for i, (layer, color) in enumerate(zip(layer_names, colors)):
            status = 'DROPPED' if color == 'red' else 'ACTIVE'
            plt.text(0.5, i, status, ha='center', va='center', fontweight='bold', 
                    color='white' if color == 'red' else 'black')
    else:
        plt.text(0.5, 0.5, 'All Skip\nConnections\nACTIVE', ha='center', va='center', 
                fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        plt.title('Skip Connections Status')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plots
    loss_curve_path = os.path.join(output_dir, f'{model_name}_loss_curves.png')
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training history saved: {os.path.join(output_dir, f'{model_name}_history.json')}")
    print(f"📈 Loss curves saved: {loss_curve_path}")
    
    # Print final summary
    print(f"\n🎯 Training Summary for {model_name}:")
    print(f"   Best Validation mIoU: {max(val_mious):.4f}")
    print(f"   Final Training Loss: {train_losses[-1]:.4f}")
    print(f"   Final Validation Loss: {val_losses[-1]:.4f}")
    if hasattr(model, 'drop_skip_layer') and model.drop_skip_layer >= 0:
        print(f"   Skip Connection Dropout: Layer {model.drop_skip_layer} dropped")
    else:
        print(f"   Skip Connection Dropout: None (all active)")
    
    return model, history

def predict(model, data_dir, output_dir, device, batch_size=4):
    """Generate predictions for test images."""
    os.makedirs(output_dir, exist_ok=True)
    
    transform = SegmentationTransform(is_training=False)
    
    test_images = []
    for filename in os.listdir(data_dir):
        if filename.endswith('_sat.jpg'):
            test_images.append(os.path.join(data_dir, filename))
    
    print(f"Found {len(test_images)} test images")
    
    model.eval()
    with torch.no_grad():
        for img_path in tqdm(test_images, desc='Generating predictions'):
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            outputs = model(image_tensor)
            
            # Handle different model output formats
            if isinstance(outputs, dict):  # DeepLabV3+
                logits = outputs['out']
            elif isinstance(outputs, tuple):  # U-Net with deep supervision
                logits = outputs[0]
            else:  # Single output
                logits = outputs
                
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu()
            pred_mask = class_to_mask(pred)
            
            filename = os.path.basename(img_path)
            output_filename = filename.replace('_sat.jpg', '_mask.png')
            output_path = os.path.join(output_dir, output_filename)
            
            Image.fromarray(pred_mask).save(output_path)
    
    print(f"Saved predictions to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Semantic Segmentation for Satellite Images - CNN Only')
    
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True,
                       help='Mode: train or predict')
    parser.add_argument('--model_type', type=str, choices=['unet', 'unet_resnet', 'deeplabv3'], required=True,
                       help='Model type: unet, unet_resnet, or deeplabv3 (all CNN-only)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to the p2_data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to save outputs')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                       help='Path to checkpoint to load')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--drop_skip_layer', type=int, default=-1,
                       help='Which skip connection layer to drop (-1: no drop, 0-3: drop specific layer)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model - all are CNN-only, no transformers
    if args.model_type == 'unet':
        model = UNet(n_classes=7, base_channels=64, drop_skip_layer=args.drop_skip_layer)
        if args.drop_skip_layer >= 0:
            model_name = f'unet_clean_dropskip{args.drop_skip_layer}'
        else:
            model_name = 'unet_clean'
    elif args.model_type == 'unet_resnet':
        model = UNetWithResNetEncoder(n_classes=7, encoder_name='resnet50')
        model_name = 'unet_resnet_clean'
    elif args.model_type == 'deeplabv3':
        # Load pre-trained DeepLabV3+ with ResNet-50 backbone
        model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        # Modify classifier heads for 7 classes (originally 21 for COCO)
        model.classifier[4] = torch.nn.Conv2d(256, 7, kernel_size=1)
        model.aux_classifier[4] = torch.nn.Conv2d(256, 7, kernel_size=1)
        model_name = 'deeplabv3_clean'
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Architecture: 100% CNN-based, no transformer components")
    
    # Print skip connection dropout info for U-Net
    if args.model_type == 'unet':
        if args.drop_skip_layer >= 0:
            layer_names = ['deepest (bottleneck)', 'deep', 'shallow', 'shallowest (surface)']
            print(f"🔥 Skip Connection Dropout: Layer {args.drop_skip_layer} ({layer_names[args.drop_skip_layer]}) DROPPED")
        else:
            print("🔗 Skip Connection Dropout: All skip connections ACTIVE")
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.load_checkpoint}")
    
    if args.mode == 'train':
        # Create datasets
        train_dir = os.path.join(args.data_dir, 'train')
        val_dir = os.path.join(args.data_dir, 'validation')
        
        train_dataset = SatelliteDataset(train_dir, is_training=True)
        val_dataset = SatelliteDataset(val_dir, is_training=False)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers,
                                pin_memory=True if device.type == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True if device.type == 'cuda' else False)
        
        # Set up loss function based on model type
        if args.model_type == 'unet':
            print("🔧 Setting up basic CrossEntropyLoss for UNet...")
            criterion = nn.CrossEntropyLoss(ignore_index=6)
            print("🎯 Strategy: Basic CrossEntropyLoss (no class weighting)")
        else:
            # Use enhanced loss strategy for other models (DeepLabV3, etc.)
            print("🔧 Setting up enhanced loss function for class imbalance...")
            criterion = CombinedLoss(num_classes=7, ignore_index=6)
            print(f"📊 Class weights: {criterion.class_weights.tolist()}")
            print("🎯 Strategy: Focal Loss → Weighted CE → Weighted CE (stable)")
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
        # Train model
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            args.epochs, device, args.output_dir, model_name
        )
        
        print("Training completed!")
        
    elif args.mode == 'predict':
        if not args.load_checkpoint:
            raise ValueError("Must specify --load_checkpoint for prediction mode")
        
        predict(model, args.data_dir, args.output_dir, device, args.batch_size)
        print("Prediction completed!")

if __name__ == '__main__':
    main()


# python3 p2_semantic_segmentation_train.py \
# --mode train \
# --model_tpye deeplabv3 \
# --data_dir ../data_2025/p2_data \ 
# --output_dir  ./p2_outuput \