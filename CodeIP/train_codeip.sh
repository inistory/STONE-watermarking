#!/bin/bash

# Set the base directory to current directory
BASE_DIR="$(pwd)"

# Change to the utils directory
cd "$BASE_DIR/src/watermarking/utils"
export CUDA_LAUNCH_BLOCKING=1
# # Generate LSTM dataset for each language
# echo "Generating LSTM dataset for Python..."
# python generate_lstm_dataset.py --language python


# echo "Generating LSTM dataset for Java..."
# python generate_lstm_dataset.py --language java

# echo "Generating LSTM dataset for C++..."
# python generate_lstm_dataset.py --language cpp

# # Train type predictor for each language
# echo "Training type predictor for Python..."
# python train_type_predictor.py --language python

# echo "Training type predictor for Java..."
# python train_type_predictor.py --language java

# echo "Training type predictor for C++..."
# python train_type_predictor.py --language cpp

#Run watermarking for HumanEval
echo "Running watermarking for HumanEval..."
cd "$BASE_DIR/src"
python run_wm.py --language python --dataset_type humaneval --save_path ./output/result_python_humaneval.json --model_name Qwen/Qwen2.5-Coder-7B-Instruct

# Run watermarking for MBPP+
# echo "Running watermarking for MBPP+..."
# cd "$BASE_DIR/src"
# python run_wm.py --language python --dataset_type mbpp --save_path ./output/result_python_mbpp.json --model_name Qwen/Qwen2.5-Coder-7B-Instruct


# # Run watermarking for HumanEvalPack Java
# echo "Running watermarking for HumanEvalPack Java..."
# cd "$BASE_DIR/src"
# python run_wm.py --language java --dataset_type humanevalpack --save_path ./output/result_java_humanevalpack.json

# # Run watermarking for HumanEvalPack C++
# echo "Running watermarking for HumanEvalPack C++..."
# cd "$BASE_DIR/src"
# python run_wm.py --language cpp --dataset_type humanevalpack --save_path ./output/result_cpp_humanevalpack.json


echo "Dataset generation, type predictor training, and watermarking completed!" 