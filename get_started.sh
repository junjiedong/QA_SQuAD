#!/usr/bin/env bash

# Get directory containing this script
HEAD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CODE_DIR=$HEAD_DIR/code
DATA_DIR=$HEAD_DIR/data
EXP_DIR=$HEAD_DIR/experiments

mkdir -p $EXP_DIR

# Creates the environment
conda create -n squad3 python=3.5

# Activates the environment
source activate squad3

# pip install into environment
pip install -r requirements.txt

# download punkt and perluniprops
python -m nltk.downloader punkt
python -m nltk.downloader perluniprops
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader maxent_ne_chunker
python -m nltk.downloader words

# Download and preprocess SQuAD data and save in data/
mkdir -p "$DATA_DIR"
rm -rf "$DATA_DIR"
python "$CODE_DIR/preprocessing/squad_preprocess.py" --data_dir "$DATA_DIR"

# Download GloVe vectors to data/
python "$CODE_DIR/preprocessing/download_wordvecs.py" --download_dir "$DATA_DIR"

cd code/preprocessing/
# preprocess for elmo
python "elmo_preprocess.py"

# preprocessing for pos/ne
# python "pos_ne_preprocessing.py"
# python "pos_ne_fast_validate.py"

cd ../../
# Install elmo
cd bilm-tf
python setup.py install
