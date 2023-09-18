#!/bin/bash

# Get the PROJECT_ROOT environment variable
PROJECT_ROOT="$PROJECT_ROOT"

# Construct BASE_DIR using PROJECT_ROOT
BASE_DIR="$PROJECT_ROOT/data/natural_questions"

# Ensure the directory exists
mkdir -p $BASE_DIR

# URL for the dataset
URL="https://storage.googleapis.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz"

# Path to save the file
SAVE_PATH="$BASE_DIR/simplified-nq-train.jsonl.gz"

# Download the file using wget or curl
if command -v wget &> /dev/null; then
    wget -O $SAVE_PATH $URL
elif command -v curl &> /dev/null; then
    curl -o $SAVE_PATH $URL
else
    echo "Both wget and curl are not available. Please install one of them and try again."
    exit 1
fi

echo "Downloaded the dataset to $SAVE_PATH"
