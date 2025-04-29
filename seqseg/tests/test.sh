#!/bin/bash

# Greetings
echo "Running tests..."

# Activate virtual environment
conda activate seqseg

# CD to the directory of the script if you need
# cd /Users/numisveins/Documents/SeqSeg

# Run the tests
python3 auto_centerline.py \
    -data_dir tests/test_data/ \
    -config_name global_test \
    -img_ext .nii.gz \
    -outdir tests/test_out/ \
    -start 0 \
    -stop 1 \
    -max_n_steps 10 \