#!/bin/bash

# Activate Python environment if you're using conda or venv
# Uncomment and modify the following line if you're using conda
# source activate your_env_name
# Or if you're using venv
# source /path/to/your/venv/bin/activate

# Create necessary directories
mkdir -p output

# ########## Step 1: Generate LSTM datasets for each type
# echo "Generating LSTM datasets..."
cd watermarking/utils

# # Generate for HumanEval+, MBPP
# # echo "Generating LSTM dataset for HumanEval+..."
# python generate_lstm_dataset.py --language python

# # Generate for HumanEvalPack Java
# # echo "Generating LSTM dataset for HumanEvalPack Java..."
# python generate_lstm_dataset.py --dataset_type humanevalpack --language java

###### Step 2: Train type predictor for each language
# echo "Training type predictor for Python..."
# python train_type_predictor.py --language python

# echo "Training type predictor for Java..."
# python train_type_predictor.py --language java

# echo "Training random type predictor for C++..."
# python train_type_predictor.py --language cpp

cd ../../

######## Step 3: Run watermarking for HumanEval+(7b)
# echo "Running watermarking for HumanEval+..."
# python run_wm.py --dataset_type humaneval --save_path ./output/result_python_humaneval.json --model_name Qwen/Qwen2.5-Coder-7B-Instruct

# ###### Step 4: Run watermarking for MBPP+
# echo "Running watermarking for MBPP+..."
# python run_wm.py --dataset_type mbpp --save_path ./output/result_python_mbpp.json --model_name Qwen/Qwen2.5-Coder-7B-Instruct

# ###### Step 5: Run watermarking for HumanEvalPack C++
# echo "Running watermarking for HumanEvalPack C++..."
python run_wm.py --dataset_type humanevalpack --language cpp --save_path ./output/result_cpp_humanevalpack.json --model_name Qwen/Qwen2.5-Coder-7B-Instruct

# # Step 6: Run watermarking for HumanEvalPack Java
# echo "Running watermarking for HumanEvalPack Java..."
# python run_wm.py --dataset_type humanevalpack --language java --save_path ./output/result_java_humanevalpack.json --model_name Qwen/Qwen2.5-Coder-7B-Instruct


# ######## Step 3: Run watermarking for HumanEval+(3b)
# echo "Running watermarking for HumanEval+..."
# python run_wm.py --dataset_type humaneval --save_path ./output/result_python_humaneval_3b.json --model_name Qwen/Qwen2.5-Coder-3B-Instruct

# ###### Step 4: Run watermarking for MBPP+
# echo "Running watermarking for MBPP+..."
# python run_wm.py --dataset_type mbpp --save_path ./output/result_python_mbpp_3b.json --model_name Qwen/Qwen2.5-Coder-3B-Instruct

# ###### Step 5: Run watermarking for HumanEvalPack C++
# echo "Running watermarking for HumanEvalPack C++..."
# python run_wm.py --dataset_type humanevalpack --language cpp --save_path ./output/result_cpp_humanevalpack_3b.json --model_name Qwen/Qwen2.5-Coder-3B-Instruct

# # Step 6: Run watermarking for HumanEvalPack Java
# echo "Running watermarking for HumanEvalPack Java..."
# python run_wm.py --dataset_type humanevalpack --language java --save_path ./output/result_java_humanevalpack_3b.json --model_name Qwen/Qwen2.5-Coder-3B-Instruct
