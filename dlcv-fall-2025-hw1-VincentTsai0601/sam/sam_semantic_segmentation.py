#!/usr/bin/env python3
"""
Semantic Segmentation using Segment Anything Model (SAM)

This script implements a three-step pipeline for semantic segmentation on satellite images:
1. Automatic Mask Generation using SAM
2. Mask Classification by Color matching
3. Final Mask Merging using painter's algorithm

Author: GitHub Copilot
Date: September 19, 2025
"""

import os
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
import random

# 設置中文字體支援
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# SAM imports
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("Error: segment_anything library not found. Please install it using:")
    print("pip install git+https://github.com/facebookresearch/segment-anything.git")
    exit(1)


class SAMSemanticSegmentation:
    """
    A class for performing semantic segmentation using SAM with color-based classification.
    """
    
    def __init__(self, sam_checkpoint_path: str, model_type: str = "vit_h", device: str = "auto"):
        """
        Initialize the SAM model and mask generator.
        
        Args:
            sam_checkpoint_path: Path to the SAM model checkpoint
            model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b')
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.sam_checkpoint_path = sam_checkpoint_path
        self.model_type = model_type
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load SAM model
        self.sam_model = None
        self.mask_generator = None
        self._load_sam_model()
        
        # Define default class color mappings
        # These represent the 7 classes in the satellite dataset
        self.class_color_map = {
            0: (0, 255, 255),    # Urban land
            1: (255, 255, 0),    # Agriculture land
            2: (255, 0, 255),    # Rangeland
            3: (0, 255, 0),      # Forest land
            4: (0, 0, 255),      # Water
            5: (255, 255, 255),  # Barren land
            6: (0, 0, 0),        # Unknown
        }
        
        # Reverse mapping for color to class
        self.color_to_class = {v: k for k, v in self.class_color_map.items()}
    
    def _load_sam_model(self):
        """Load the SAM model and initialize the mask generator."""
        try:
            # Load SAM model
            self.sam_model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint_path)
            self.sam_model.to(device=self.device)
            
            # Initialize mask generator with optimized parameters
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Requires mask to have at least 100 pixels
            )
            
            print(f"Successfully loaded SAM model: {self.model_type}")
            
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            raise
    
    def generate_masks(self, image: np.ndarray) -> List[Dict]:
        """
        Step 1: Generate masks using SAM's automatic mask generation.
        
        Args:
            image: Input image as RGB numpy array (H, W, 3)
        
        Returns:
            List of mask dictionaries containing 'segmentation' and 'predicted_iou'
        """
        if self.mask_generator is None:
            raise RuntimeError("Mask generator not initialized")
        
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # SAM expects RGB format
                masks = self.mask_generator.generate(image)
            else:
                raise ValueError("Image must be RGB format (H, W, 3)")
            
            print(f"Generated {len(masks)} masks")
            return masks
            
        except Exception as e:
            print(f"Error generating masks: {e}")
            return []
    
    def create_color_prototypes(self, train_data_dir: str, 
                              max_images: int = 10) -> Dict[int, np.ndarray]:
        """
        Create class color prototypes from training data.
        適配你的資料格式：所有檔案都在同一個目錄，格式為 XXXX_sat.jpg 和 XXXX_mask.png
        
        Args:
            train_data_dir: Directory containing both training images and masks
            max_images: Maximum number of training images to process
        
        Returns:
            Dictionary mapping class index to average RGB prototype color
        """
        print("Creating color prototypes from training data...")
        
        train_dir = Path(train_data_dir)
        
        # Storage for color accumulation
        class_colors = {i: [] for i in range(7)}  # Assuming 7 classes
        
        # Get satellite image files (XXXX_sat.jpg)
        sat_files = list(train_dir.glob("*_sat.jpg"))
        sat_files = sorted(sat_files)[:max_images]
        
        print(f"Found {len(sat_files)} satellite images for prototype creation")
        
        for sat_file in sat_files:
            print(f"Processing {sat_file.name}...")
            
        # Build corresponding mask filename (XXXX_mask.png)
            base_name = sat_file.stem.replace("_sat", "")  # Remove "_sat"
            mask_file = train_dir / f"{base_name}_mask.png"
            
            if not mask_file.exists():
                print(f"Warning: No mask found for {sat_file.name}")
                continue
            
            # Load image and mask
            image = cv2.imread(str(sat_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(str(mask_file))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            
            # Extract colors for each class
            for class_idx, class_color in self.class_color_map.items():
                # Find pixels matching this class color
                color_mask = np.all(mask == class_color, axis=2)
                
                if np.any(color_mask):
                    # Extract corresponding pixels from original image
                    class_pixels = image[color_mask]
                    if len(class_pixels) > 0:
                        class_colors[class_idx].extend(class_pixels)
        
        # Calculate average colors for each class
        color_prototypes = {}
        for class_idx, pixels in class_colors.items():
            if len(pixels) > 0:
                pixels_array = np.array(pixels)
                prototype = np.mean(pixels_array, axis=0)
                color_prototypes[class_idx] = prototype
                print(f"Class {class_idx}: prototype color {prototype}")
            else:
                # Use default color if no pixels found
                color_prototypes[class_idx] = np.array(self.class_color_map[class_idx])
                print(f"Class {class_idx}: using default color (no training data)")
        
        return color_prototypes
    
    def classify_masks(self, image: np.ndarray, masks: List[Dict], 
                      color_prototypes: Dict[int, np.ndarray]) -> List[Dict]:
        """
        Step 2: Classify masks by comparing average colors to prototypes.
        
        Args:
            image: Original RGB image
            masks: List of mask dictionaries from SAM
            color_prototypes: Dictionary mapping class index to prototype color
        
        Returns:
            List of masks with added 'class_id' field
        """
        print(f"Classifying {len(masks)} masks...")
        
        classified_masks = []
        
        for i, mask in enumerate(masks):
            segmentation = mask['segmentation']
            
            # Extract pixels from original image using the mask
            masked_pixels = image[segmentation]
            
            if len(masked_pixels) == 0:
                # Empty mask, assign to unknown class
                mask['class_id'] = 6  # Unknown class
                classified_masks.append(mask)
                continue
            
            # Calculate average color of masked region
            avg_color = np.mean(masked_pixels, axis=0)
            
            # Find closest prototype color
            min_distance = float('inf')
            best_class = 6  # Default to unknown
            
            for class_idx, prototype in color_prototypes.items():
                # Calculate Euclidean distance in RGB space
                distance = np.linalg.norm(avg_color - prototype)
                
                if distance < min_distance:
                    min_distance = distance
                    best_class = class_idx
            
            # Add classification to mask
            mask['class_id'] = best_class
            classified_masks.append(mask)
        
        # Print classification summary
        class_counts = {}
        for mask in classified_masks:
            class_id = mask['class_id']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        print("Classification summary:")
        for class_id, count in sorted(class_counts.items()):
            print(f"  Class {class_id}: {count} masks")
        
        return classified_masks
    
    def merge_masks_to_semantic_map(self, image_shape: Tuple[int, int], 
                                  classified_masks: List[Dict]) -> np.ndarray:
        """
        Step 3: Merge classified masks into final semantic map using painter's algorithm.
        
        Args:
            image_shape: Shape of the output image (H, W)
            classified_masks: List of classified mask dictionaries
        
        Returns:
            Final semantic map as RGB numpy array (H, W, 3)
        """
        print(f"Merging {len(classified_masks)} masks into semantic map...")
        
        # Create empty canvas
        height, width = image_shape[:2]
        semantic_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sort masks by predicted IoU in ascending order (painter's algorithm)
        # Lower confidence masks painted first, higher confidence masks painted last
        sorted_masks = sorted(classified_masks, key=lambda x: x.get('predicted_iou', 0))
        
        # Paint masks onto canvas
        for mask in sorted_masks:
            segmentation = mask['segmentation']
            class_id = mask['class_id']
            
            # Get color for this class
            class_color = self.class_color_map.get(class_id, (0, 0, 0))
            
            # Paint this color onto the canvas
            semantic_map[segmentation] = class_color
        
        return semantic_map
    
    def visualize_masks(self, image: np.ndarray, masks: List[Dict], 
                       output_path: str, title: str = "Generated Masks"):
        """
        Visualize mask results
        
        Args:
            image: Original image
            masks: List of masks
            output_path: Output path
            title: Plot title
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
        # 為每個遮罩添加隨機顏色
        for i, mask in enumerate(masks):
            segmentation = mask['segmentation']
            
            # Create random color
            color = [random.random(), random.random(), random.random(), 0.5]
            
            # Create mask contours
            contours, _ = cv2.findContours(
                segmentation.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                contour = contour.reshape(-1, 2)
                polygon = patches.Polygon(contour, closed=True, 
                                        facecolor=color, edgecolor='white', linewidth=1)
                ax.add_patch(polygon)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Mask visualization saved to: {output_path}")
    
    def visualize_classified_masks(self, image: np.ndarray, classified_masks: List[Dict], 
                                 output_path: str, title: str = "Classified Masks"):
        """
        Visualize classified masks
        
        Args:
            image: Original image
            classified_masks: List of classified masks
            output_path: Output path
            title: Plot title
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
        class_names = ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"]
        
        # Use different colors for each class
        class_colors = {
            0: [1.0, 0.0, 1.0, 0.6],  # Urban - Magenta
            1: [1.0, 1.0, 0.0, 0.6],  # Agriculture - Yellow
            2: [1.0, 0.5, 0.0, 0.6],  # Rangeland - Orange
            3: [0.0, 1.0, 0.0, 0.6],  # Forest - Green
            4: [0.0, 0.0, 1.0, 0.6],  # Water - Blue
            5: [0.5, 0.5, 0.5, 0.6],  # Barren - Gray
            6: [0.0, 0.0, 0.0, 0.6],  # Unknown - Black
        }
        
        for mask in classified_masks:
            segmentation = mask['segmentation']
            class_id = mask['class_id']
            confidence = mask.get('predicted_iou', 0)
            
            color = class_colors.get(class_id, [0.5, 0.5, 0.5, 0.6])
            
            # Create mask contours
            contours, _ = cv2.findContours(
                segmentation.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                if len(contour) > 2:
                    contour = contour.reshape(-1, 2)
                    polygon = patches.Polygon(contour, closed=True, 
                                            facecolor=color, edgecolor='white', linewidth=1)
                    ax.add_patch(polygon)
                    
        # Add class label
                    if len(contour) > 0:
                        center_x, center_y = np.mean(contour, axis=0)
                        ax.text(center_x, center_y, f"{class_names[class_id]}\n({confidence:.2f})",
                               fontsize=8, ha='center', va='center', 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Classified masks visualization saved to: {output_path}")
    
    def visualize_step_by_step(self, image_path: str, output_dir: str, 
                             color_prototypes: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Step-by-step visualization of the processing pipeline
        
        Args:
            image_path: Input image path
            output_dir: Output directory
            color_prototypes: Color prototypes
        
        Returns:
            Final semantic segmentation map
        """
        image_name = Path(image_path).stem
        step_output_dir = Path(output_dir) / f"{image_name}_steps"
        step_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== Starting step-by-step processing: {image_path} ===")
        
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 0: Save original image
        original_path = step_output_dir / f"step0_original.png"
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title("Step 0: Original Satellite Image", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(original_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Step 0: Original image saved to {original_path}")
        
        # Step 1: Generate masks
        print("Step 1: Generating masks using SAM...")
        masks = self.generate_masks(image)
        
        if len(masks) == 0:
            print("Warning: No masks generated")
            return np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        
        # Visualize generated masks
        masks_path = step_output_dir / f"step1_generated_masks.png"
        self.visualize_masks(image, masks, str(masks_path), 
                           f"Step 1: SAM Generated Masks ({len(masks)} masks)")
        
        # Step 2: Classify masks
        print("Step 2: Classifying masks using color prototypes...")
        classified_masks = self.classify_masks(image, masks, color_prototypes)
        
        # Visualize classified masks
        classified_path = step_output_dir / f"step2_classified_masks.png"
        self.visualize_classified_masks(image, classified_masks, str(classified_path),
                                      "Step 2: Classified Masks (by Color Prototypes)")
        
        # Step 3: Merge masks
        print("Step 3: Merging masks to generate final semantic map...")
        semantic_map = self.merge_masks_to_semantic_map(image.shape, classified_masks)
        
        # Visualize final result
        final_path = step_output_dir / f"step3_final_semantic_map.png"
        plt.figure(figsize=(10, 8))
        plt.imshow(semantic_map)
        plt.title("Step 3: Final Semantic Segmentation Map", fontsize=16)
        plt.axis('off')
        
        # Add legend
        class_names = ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"]
        legend_elements = []
        for class_id, color in self.class_color_map.items():
            legend_elements.append(patches.Patch(color=np.array(color)/255.0, 
                                                label=f"Class {class_id}: {class_names[class_id]}"))
        
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Step 3: Final semantic map saved to {final_path}")
        
        # Step 4: Create comparison plot
        comparison_path = step_output_dir / f"step4_comparison.png"
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Satellite Image", fontsize=14)
        axes[0, 0].axis('off')
        
        # SAM masks (simplified display of first 10)
        axes[0, 1].imshow(image)
        for i, mask in enumerate(masks[:10]):  # Only show first 10 masks to avoid complexity
            segmentation = mask['segmentation']
            color = [random.random(), random.random(), random.random(), 0.5]
            contours, _ = cv2.findContours(segmentation.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) > 2:
                    contour = contour.reshape(-1, 2)
                    polygon = patches.Polygon(contour, closed=True, 
                                            facecolor=color, edgecolor='white', linewidth=0.5)
                    axes[0, 1].add_patch(polygon)
        axes[0, 1].set_title(f"SAM Generated Masks (showing first 10 of {len(masks)})", fontsize=14)
        axes[0, 1].axis('off')
        
        # Classified masks
        axes[1, 0].imshow(image)
        class_colors = {
            0: [1.0, 0.0, 1.0, 0.6], 1: [1.0, 1.0, 0.0, 0.6], 2: [1.0, 0.5, 0.0, 0.6],
            3: [0.0, 1.0, 0.0, 0.6], 4: [0.0, 0.0, 1.0, 0.6], 5: [0.5, 0.5, 0.5, 0.6], 6: [0.0, 0.0, 0.0, 0.6]
        }
        for mask in classified_masks:
            segmentation = mask['segmentation']
            class_id = mask['class_id']
            color = class_colors.get(class_id, [0.5, 0.5, 0.5, 0.6])
            contours, _ = cv2.findContours(segmentation.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) > 2:
                    contour = contour.reshape(-1, 2)
                    polygon = patches.Polygon(contour, closed=True, 
                                            facecolor=color, edgecolor='white', linewidth=0.5)
                    axes[1, 0].add_patch(polygon)
        axes[1, 0].set_title("Classified Masks", fontsize=14)
        axes[1, 0].axis('off')
        
        # Final semantic map
        axes[1, 1].imshow(semantic_map)
        axes[1, 1].set_title("Final Semantic Segmentation Map", fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Step 4: Comparison plot saved to {comparison_path}")
        
        print(f"\n=== All step images saved to: {step_output_dir} ===")
        return semantic_map
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         color_prototypes: Dict[int, np.ndarray], visualize_steps: bool = True):
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output semantic maps
            color_prototypes: Color prototypes for classification
            visualize_steps: Whether to save step-by-step visualization
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        image_files = sorted(image_files)
        
        print(f"Found {len(image_files)} images to process")
        
        for i, img_file in enumerate(image_files):
            print(f"\nProcessing {i+1}/{len(image_files)}: {img_file.name}")
            
            try:
                if visualize_steps:
                    # Use step-by-step visualization processing
                    semantic_map = self.visualize_step_by_step(str(img_file), str(output_path), color_prototypes)
                else:
                    # Standard processing (without visualization)
                    semantic_map = self.process_image_standard(str(img_file), color_prototypes)
                
                # Save final result
                output_file = output_path / (img_file.stem + "_semantic.png")
                semantic_map_bgr = cv2.cvtColor(semantic_map, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_file), semantic_map_bgr)
                
                print(f"Final result saved to: {output_file}")
                
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                continue
    
    def process_image_standard(self, image_path: str, color_prototypes: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Standard image processing without visualization (original process_image method)
        
        Args:
            image_path: Path to input image
            color_prototypes: Color prototypes for classification
        
        Returns:
            Final semantic segmentation map
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 1: Generate masks
        masks = self.generate_masks(image)
        
        if len(masks) == 0:
            print("Warning: No masks generated, returning empty semantic map")
            return np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        
        # Step 2: Classify masks
        classified_masks = self.classify_masks(image, masks, color_prototypes)
        
        # Step 3: Merge masks
        semantic_map = self.merge_masks_to_semantic_map(image.shape, classified_masks)
        
        return semantic_map


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="SAM-based Semantic Segmentation")
    parser.add_argument("--input_dir", required=True, 
                       help="Path to directory containing validation images")
    parser.add_argument("--output_dir", required=True,
                       help="Path to save predicted semantic masks")
    parser.add_argument("--train_data_dir", required=True,
                       help="Path to training data directory (containing XXXX_sat.jpg and XXXX_mask.png files)")
    parser.add_argument("--sam_checkpoint", required=True,
                       help="Path to SAM model checkpoint file")
    parser.add_argument("--model_type", default="vit_b", 
                       choices=["vit_h", "vit_l", "vit_b"],
                       help="SAM model type")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cuda", "cpu"],
                       help="Device to run on")
    parser.add_argument("--max_train_images", type=int, default=100,
                       help="Maximum number of training images to use for prototype creation")
    parser.add_argument("--visualize_steps", action="store_true",
                       help="Save step-by-step visualization images for each processing stage")
    parser.add_argument("--no_visualize", action="store_true",
                       help="Disable step-by-step visualization (faster processing)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    if not os.path.exists(args.sam_checkpoint):
        print(f"Error: SAM checkpoint not found: {args.sam_checkpoint}")
        return
    
    if not os.path.exists(args.train_data_dir):
        print(f"Error: Training data directory not found: {args.train_data_dir}")
        return
    
    # visualize the steps
    visualize_steps = not args.no_visualize  # Enable visualization by default, unless --no_visualize is used
    if args.visualize_steps:
        visualize_steps = True
    
    try:
        # Initialize SAM segmentation system
        print("Initializing SAM semantic segmentation system...")
        sam_seg = SAMSemanticSegmentation(
            sam_checkpoint_path=args.sam_checkpoint,
            model_type=args.model_type,
            device=args.device
        )
        
        # Create color prototypes from training data
        print(f"\nCreating color prototypes from training data in {args.train_data_dir}...")
        color_prototypes = sam_seg.create_color_prototypes(
            args.train_data_dir, args.max_train_images
        )
        
        # Process all images
        print(f"\nProcessing images from {args.input_dir}...")
        if visualize_steps:
            print("🎨 Step-by-step visualization enabled - will save intermediate results")
        else:
            print("⚡ Fast mode - visualization disabled")
            
        sam_seg.process_directory(args.input_dir, args.output_dir, color_prototypes, visualize_steps)
        
        print(f"\nProcessing complete! Results saved to: {args.output_dir}")
        if visualize_steps:
            print("📊 Check the '*_steps' folders for detailed step-by-step visualizations!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
