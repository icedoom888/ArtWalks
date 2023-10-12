#!/bin/bash

set -e
eval "$(conda shell.bash hook)"


# Read user inputs
echo ""
INPATH=$1
echo "INPUT FOLDER PATH: $INPATH"

FOLDER_NAME=$2
echo "COLLECTION NAME: $FOLDER_NAME"

S=$3
echo "NUMBER OF DIFFUSION IMAGES: $S"

I=$4
echo "SECONDS BETWEEN IMAGES: $I"

F=$5
echo "SECONDS OF FREEZING ORIGINAL FRAMES: $F"

M=$6
echo "MODEL CHOSEN: $M"

echo ""

# Generate dissusion images
conda activate diff
python diffusion_models/diffusion.py --input_path $INPATH --folder_name $FOLDER_NAME --model $M --interpolation_steps $S 
conda deactivate

# interpolate between images and generate video
conda activate film
python generate_videos.py --input_path output --folder_name $FOLDER_NAME --sec_interpolation $I --sec_freeze $F --clean
conda deactivate

# Super resolution on Video
conda activate diff
cd Video-Super-Resolution-ESRGAN
if [ "$M" = 'kandisnky' ]; then
    sr=2
else
    sr=8
fi
python inference_realesrgan_video.py -n RealESRGAN_x4plus -i ../output/$FOLDER_NAME/final_video.mp4 -o ../output/$FOLDER_NAME/ -s $sr
conda deactivate
