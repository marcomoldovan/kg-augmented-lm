#!/bin/bash

# Get the PROJECT_ROOT environment variable
PROJECT_ROOT="$PROJECT_ROOT"

if [ -z "$PROJECT_ROOT" ]; then
    echo "PROJECT_ROOT environment variable is not set."
    exit 1
fi

# Construct BASE_DIR using PROJECT_ROOT
BASE_DIR="$PROJECT_ROOT/data/wikigraphs"

# wikitext-103
TARGET_DIR=${BASE_DIR}/wikitext-103
mkdir -p ${TARGET_DIR}
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -P ${TARGET_DIR}
unzip ${TARGET_DIR}/wikitext-103-v1.zip -d ${TARGET_DIR}
mv ${TARGET_DIR}/wikitext-103/* ${TARGET_DIR}
rm -rf ${TARGET_DIR}/wikitext-103 ${TARGET_DIR}/wikitext-103-v1.zip

# wikitext-103-raw
TARGET_DIR=${BASE_DIR}/wikitext-103-raw
mkdir -p ${TARGET_DIR}
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip -P ${TARGET_DIR}
unzip ${TARGET_DIR}/wikitext-103-raw-v1.zip -d ${TARGET_DIR}
mv ${TARGET_DIR}/wikitext-103-raw/* ${TARGET_DIR}
rm -rf ${TARGET_DIR}/wikitext-103-raw ${TARGET_DIR}/wikitext-103-raw-v1.zip


# processed freebase graphs
FREEBASE_TARGET_DIR=/tmp/data
mkdir -p ${FREEBASE_TARGET_DIR}/packaged/
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uuSS2o72dUCJrcLff6NBiLJuTgSU-uRo' -O ${FREEBASE_TARGET_DIR}/packaged/max256.tar
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nOfUq3RUoPEWNZa2QHXl2q-1gA5F6kYh' -O ${FREEBASE_TARGET_DIR}/packaged/max512.tar
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uuJwkocJXG1UcQ-RCH3JU96VsDvi7UD2' -O ${FREEBASE_TARGET_DIR}/packaged/max1024.tar

for version in max1024 max512 max256
do
  output_dir=${FREEBASE_TARGET_DIR}/freebase/${version}/
  mkdir -p ${output_dir}
  tar -xvf ${FREEBASE_TARGET_DIR}/packaged/${version}.tar -C ${output_dir}
done
rm -rf ${FREEBASE_TARGET_DIR}/packaged
