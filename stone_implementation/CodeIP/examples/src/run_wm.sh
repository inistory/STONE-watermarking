#!/bin/bash

# Activate Python environment if you're using conda or venv
# Uncomment and modify the following line if you're using conda
# source activate your_env_name
# Or if you're using venv
# source /path/to/your/venv/bin/activate

# Create necessary directories
mkdir -p output

# # Step 1: Generate LSTM dataset
# echo "Generating LSTM dataset..."
# cd watermarking/utils
# python generate_lstm_dataset.py

# # Step 2: Train type predictor
# echo "Training type predictor..."
# python train_type_predictor.py

# Step 3: Run watermarking for HumanEval+
echo "Running watermarking for HumanEval+..."
python run_wm.py --language python --dataset_type humaneval --save_path ./output/result_python_humaneval.json --model_name Qwen/Qwen2.5-Coder-7B-Instruct

# Step 4: Run watermarking for MBPP+
# echo "Running watermarking for MBPP+..."
# python run_wm.py --language python --dataset_type mbpp --save_path ./output/result_python_mbpp.json --model_name Qwen/Qwen2.5-Coder-7B-Instruct