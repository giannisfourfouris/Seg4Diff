#!/bin/bash

set -e  # stop on error

echo "===== Environment Setup Started ====="

# Load module
echo "[1/6] Loading Python module..."
module load Python/3.10.4-GCCcore-11.3.0

# Create virtual environment
echo "[2/6] Creating virtual environment (.woska_venv_sample)..."
python3 -m venv .woska_venv_sample

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source .woska_venv_sample/bin/activate

# Install PyTorch
echo "[4/6] Installing PyTorch (CUDA 13.0)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install requirements
echo "[5/6] Installing project requirements..."
pip install -r requirements.txt

# Install local libraries
echo "[6/6] Installing local editable packages..."
pip install -e ./libraries/detectron2
pip install -e ./diffusers

# Ask for Hugging Face token
echo ""
echo "===== Hugging Face Authentication ====="
read -p "Enter your Hugging Face token: " HF_TOKEN

# Export token
export HF_TOKEN=$HF_TOKEN
echo "HF_TOKEN has been set for this session."

echo ""
echo "===== Setup Completed Successfully ====="
echo "To reuse the environment later, run:"
echo "  module load Python/3.10.4-GCCcore-11.3.0"
echo "  source .woska_venv_sample/bin/activate"