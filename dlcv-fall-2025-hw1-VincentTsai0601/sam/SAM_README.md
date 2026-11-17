# SAM Semantic Segmentation Script

This script implements a three-step pipeline for semantic segmentation on satellite images using the Segment Anything Model (SAM).

## Overview

The pipeline consists of:
1. **Automatic Mask Generation** - Uses SAM to generate object masks
2. **Mask Classification by Color** - Classifies masks based on color similarity to training data prototypes
3. **Final Mask Merging** - Combines classified masks using painter's algorithm

## Installation

1. Install the required dependencies:
```bash
pip install -r sam_requirements.txt
```

2. Install the Segment Anything library:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

3. Download a SAM model checkpoint:
- Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints
- Recommended: `sam_vit_h_4b8939.pth` (ViT-H SAM model)

## Usage

### Basic Usage

```bash
python sam_semantic_segmentation.py --input_dir /home/d13944024/kai/dlcv-fall-2025-hw1-snooow1029/sam/images --output_dir /home/d13944024/kai/dlcv-fall-2025-hw1-snooow1029/sam/output --train_data_dir /home/d13944024/kai/data_2025/p2_data/train --sam_checkpoint /home/d13944024/kai/dlcv-fall-2025-hw1-snooow1029/sam/sam_vit_b_01ec64.pth```

### Full Arguments

```bash
python sam_semantic_segmentation.py \
    --input_dir /home/d13944024/kai/dlcv-fall-2025-hw1-snooow1029/sam/images \
    --output_dir /home/d13944024/kai/dlcv-fall-2025-hw1-snooow1029/sam/output \
    --train_data_dir /path/to/training/data \
    --sam_checkpoint /home/d13944024/kai/dlcv-fall-2025-hw1-snooow1029/sam/sam_vit_b_01ec64.pth \
    --model_type vit_h \
    --device auto \
    --max_train_images 10
```

### Arguments

- `--input_dir`: Directory containing images to process
- `--output_dir`: Directory to save semantic segmentation results
- `--train_data_dir`: Training data directory (should contain `images/` and `masks/` subdirectories)
- `--sam_checkpoint`: Path to SAM model checkpoint file
- `--model_type`: SAM model type (`vit_h`, `vit_l`, `vit_b`) - default: `vit_h`
- `--device`: Device to use (`auto`, `cuda`, `cpu`) - default: `auto`
- `--max_train_images`: Max training images for prototype creation - default: 10

## Directory Structure

Your training data directory should be organized as:
```
train_data_dir/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

## Class Color Mapping

The script assumes 7 semantic classes with the following color mapping:
- Class 0 (Urban land): Cyan (0, 255, 255)
- Class 1 (Agriculture land): Yellow (255, 255, 0)
- Class 2 (Rangeland): Magenta (255, 0, 255)
- Class 3 (Forest land): Green (0, 255, 0)
- Class 4 (Water): Blue (0, 0, 255)
- Class 5 (Barren land): White (255, 255, 255)
- Class 6 (Unknown): Black (0, 0, 0)

## Output

The script will generate semantic segmentation masks with the filename pattern:
`{original_name}_semantic.png`

## Performance Notes

- The script uses CUDA if available for faster processing
- Mask generation parameters are optimized for satellite imagery
- The painter's algorithm ensures higher confidence masks override lower confidence ones
- Color prototype creation uses a subset of training data for efficiency

## Troubleshooting

1. **CUDA out of memory**: Use `--device cpu` or reduce image resolution
2. **No masks generated**: Check image quality and SAM parameters
3. **Poor classification**: Ensure training data quality and increase `--max_train_images`
4. **Missing segment_anything**: Install using the pip command above

## Example

```bash
# Example with actual paths
python sam_semantic_segmentation.py \
    --input_dir ./validation_images \
    --output_dir ./segmentation_results \
    --train_data_dir ./training_data \
    --sam_checkpoint ./models/sam_vit_h_4b8939.pth \
    --max_train_images 15
```

This will process all images in `./validation_images` and save the results to `./segmentation_results`.
