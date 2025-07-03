# V-JEPA2 Fine-tuning on MERL Shopping Dataset

This directory contains the configuration and scripts for fine-tuning V-JEPA2 on the MERL Shopping Dataset for video action recognition.

## Dataset Overview

The MERL Shopping Dataset contains 106 videos (~2 minutes each) with 5 action classes:
1. **Reach To Shelf** (Class 0)
2. **Retract From Shelf** (Class 1) 
3. **Hand In Shelf** (Class 2)
4. **Inspect Product** (Class 3)
5. **Inspect Shelf** (Class 4)

## Setup Instructions

### 1. Data Preparation

First, convert the MERL dataset from JSON format to CSV format:

```bash
python convert_merl_to_csv.py
```

This will create CSV files in `MERL/csv_format/` with the following structure:
- `train_paths.csv` - Training data
- `val_paths.csv` - Validation data  
- `test_paths.csv` - Test data

Each CSV file contains rows with format: `video_path start_frame end_frame label`

### 2. Download Pre-trained Checkpoint

Download the V-JEPA2 ViT-L/16 pre-trained checkpoint:

```bash
# Create checkpoint directory
mkdir -p /mnt/data/ckpts/

# Download the checkpoint (you'll need to get this from the official source)
# The checkpoint should be saved as /mnt/data/ckpts/vitl.pt
```

### 3. Run Fine-tuning

Use the evaluation framework to fine-tune V-JEPA2 on MERL:

```bash
# Single GPU training
python evals/main.py --fname configs/eval/vitl/merl.yaml --devices cuda:0 --debugmode True

# Multi-GPU training (example with 4 GPUs)
python evals/main.py --fname configs/eval/vitl/merl.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3
```

## Configuration Details

The configuration file `configs/eval/vitl/merl.yaml` includes:

### Model Settings
- **Model**: V-JEPA2 ViT-L/16 (300M parameters)
- **Input Resolution**: 256×256
- **Frames per Clip**: 16
- **Patch Size**: 16×16
- **Tubelet Size**: 2

### Training Settings
- **Batch Size**: 8 (adjusted for smaller dataset)
- **Epochs**: 30
- **Learning Rates**: Multiple heads with different learning rates (0.0001 to 0.005)
- **Weight Decay**: Multiple values (0.01, 0.1, 0.4)
- **Mixed Precision**: bfloat16

### Data Settings
- **Dataset**: MERL Shopping Dataset
- **Classes**: 5 action classes
- **Segments**: 2 segments per video
- **Views per Segment**: 3 views for data augmentation

## Expected Performance

Based on the MERL dataset documentation:
- **Previous Best**: 81.87% mAP with Bidirectional LSTM
- **Per-frame Accuracy**: 76.4%
- **Action-wise AP**: 70-93% depending on action type

## Output

The fine-tuning will save:
- **Logs**: `/home/chanyong/workspace/vjepa2-cursor/evals/vitl/merl/log_r*.csv`
- **Checkpoints**: `/home/chanyong/workspace/vjepa2-cursor/evals/vitl/merl/latest.pt`
- **Best Model**: The best performing classifier head will be selected

## Evaluation

After training, you can evaluate the model:

```bash
python evals/main.py --fname configs/eval/vitl/merl.yaml --val_only --devices cuda:0 --debugmode True
```

## Troubleshooting

1. **Memory Issues**: Reduce batch size if you encounter GPU memory issues
2. **Data Loading**: Ensure video paths in CSV files are correct
3. **Checkpoint**: Verify the pre-trained checkpoint path is correct
4. **Dependencies**: Make sure all required packages are installed

## Files Created

- `convert_merl_to_csv.py` - Converts MERL JSON to CSV format
- `configs/eval/vitl/merl.yaml` - Fine-tuning configuration
- `MERL/csv_format/` - Converted CSV files (created after running conversion script) 