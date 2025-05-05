#!/bin/bash

# Set the base directory
BASE_DIR="/home/jungin/workspace/STONE-watermarking/CodeIP"

# Change to the utils directory
cd "$BASE_DIR/src/watermarking/utils"

# Generate LSTM dataset
echo "Generating LSTM dataset..."
python generate_lstm_dataset.py

# Train type predictor
echo "Training type predictor..."
python train_type_predictor.py

echo "Dataset generation and type predictor training completed!" 