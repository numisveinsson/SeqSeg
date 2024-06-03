#!/bin/bash

# Greetings
echo "Running tests..."

# Activate virtual environment
conda activate ml_data

# CD to the directory of the script /Users/numisveins/Documents/SeqSeg
cd /Users/numisveins/Documents/SeqSeg

# Run the tests
python3 auto_centerline.py \
    -data_dir /Users/numisveins/Documents/SeqSeg/tests/test_data/ \
    -config_name global_test \
    -img_ext .nii.gz \
    -outdir /Users/numisveins/Documents/SeqSeg/tests/test_output/ \
    -start 0 \
    -stop 1 \
    -max_n_steps 150 \