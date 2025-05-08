#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate Python environment if you're using conda or venv
# Uncomment and modify the following line if you're using conda
# source activate your_env_name
# Or if you're using venv
# source /path/to/your/venv/bin/activate

# Change to the script directory
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p output

# Step 1: Generate LSTM dataset
# echo "Generating LSTM dataset..."
# cd watermarking/utils
# python generate_lstm_dataset.py

# # Step 2: Train type predictor
# echo "Training type predictor..."
# python train_type_predictor.py

# Step 3: Go back to main directory and run watermarking
echo "Running watermarking..."
cd "$SCRIPT_DIR"
python run_wm.py --language python --save_path ./output/result_python_humaneval.json --model_name Qwen/Qwen2.5-Coder-7B-Instruct