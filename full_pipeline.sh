#!/bin/bash

set -e
eval "$(conda shell.bash hook)"

# Read user inputs
INPATH=$1
echo "INPUT FOLDER PATH: $INPATH"
FOLDER_NAME=$2
echo "COLLECTION NAME: $FOLDER_NAME"
S=$3
echo "DIFFUSION IMAGES: $S"
I=$4
echo "INTERPOLATION STEPS: $I"
F=$5
echo "FREEZE FRAME SECONDS: $F"

# Generate dissusion images
conda activate diffusers
python unclip.py --input_path $INPATH --folder_name $FOLDER_NAME --interpolation_steps $S
conda deactivate

# interpolate between images and generate video
conda activate film
python generate_videos.py --input_path output --folder_name $FOLDER_NAME --frames $I --sec_freeze $F
conda deactivate