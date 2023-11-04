#!/bin/bash

# Get the PROJECT_ROOT environment variable
PROJECT_ROOT="$PROJECT_ROOT"

# Construct BASE_DIR using PROJECT_ROOT
BASE_DIR="$PROJECT_ROOT/data/natural_questions"

# Ensure the directory exists
mkdir -p $BASE_DIR

# Declare an associative array with URLs
declare -A urls
urls["wikidata5m_alias.tar.gz"]="https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1"
urls["wikidata5m_text.txt.gz"]="https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1"
urls["wikidata5m_all_triplet.txt.gz"]="https://www.dropbox.com/s/563omb11cxaqr83/wikidata5m_all_triplet.txt.gz?dl=1"
urls["wikidata5m_inductive.tar.gz"]="https://www.dropbox.com/s/csed3cgal3m7rzo/wikidata5m_inductive.tar.gz?dl=1"
urls["wikidata5m_transductive.tar.gz"]="https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1"

# Iterate over the array and download files
for k in "${!urls[@]}"; do
    SAVE_PATH="$BASE_DIR$k"
    if command -v wget &> /dev/null; then
        wget --output-document=$SAVE_PATH "${urls[$k]}"
    elif command -v curl &> /dev/null; then
        curl -L -o $SAVE_PATH "${urls[$k]}"
    else
        echo "Both wget and curl are not available. Please install one of them and try again."
        exit 1
    fi
    echo "Downloaded the dataset to $SAVE_PATH"
done
