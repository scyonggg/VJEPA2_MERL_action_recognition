#!/bin/bash

# V-JEPA2 Fine-tuning Script for MERL Shopping Dataset
# This script activates the conda environment and runs the fine-tuning process

set -e  # Exit on any error

echo "=== V-JEPA2 Fine-tuning for MERL Shopping Dataset ==="
echo "Starting fine-tuning process..."

# Activate conda environment
echo "Activating vjepa2 conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vjepa2

# Check if CUDA is available
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "Error: CUDA not available or PyTorch not installed"
    exit 1
fi

# Check if the config file exists
if [ ! -f "configs/eval/vitl/merl.yaml" ]; then
    echo "Error: Configuration file configs/eval/vitl/merl.yaml not found"
    exit 1
fi

# Check if the checkpoint directory exists
if [ ! -d "ckpts" ]; then
    echo "Warning: Checkpoint directory 'ckpts' not found. Creating it..."
    mkdir -p ckpts
fi

# Check if the pre-trained model exists
if [ ! -f "ckpts/vjepa2_vitl.pt" ]; then
    echo "Warning: Pre-trained model 'ckpts/vjepa2_vitl.pt' not found"
    echo "Please download the V-JEPA2 ViT-L/16 checkpoint to ckpts/vjepa2_vitl.pt"
    echo "You can download it from the official V-JEPA2 repository"
fi

echo "Running fine-tuning with the following parameters:"
echo "- Config file: configs/eval/vitl/merl.yaml"
echo "- Device: cuda:0"
echo "- Debug mode: True"
echo ""

# Run the fine-tuning command
python -m evals.main --fname configs/eval/vitl/merl.yaml --devices cuda:0 --debugmode True

echo ""
echo "=== Fine-tuning completed ==="
