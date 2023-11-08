#!/bin/bash

set -e
eval "$(conda shell.bash hook)"


# Read user inputs
INPATH=$1
FOLDER_NAME=$2
OUTPATH=$3
S=$4
I=$5
F=$6
M=$7

echo ""
echo "INPUT FOLDER PATH: $INPATH"
echo "COLLECTION NAME: $FOLDER_NAME"
echo "OUTPUT FOLDER PATH: $OUTPATH"
echo "NUMBER OF DIFFUSION IMAGES: $S"
echo "SECONDS BETWEEN IMAGES: $I"
echo "SECONDS OF FREEZING ORIGINAL FRAMES: $F"
echo "MODEL CHOSEN: $M"
echo ""

python utils.py --input_path $INPATH --folder_name $FOLDER_NAME --s $S --i $I --f $F

# Generate dissusion images
conda activate diff
python diffusion_models/diffusion.py --input_path $INPATH --folder_name $FOLDER_NAME --output_path $OUTPATH --model $M --interpolation_steps $S 
conda deactivate

# interpolate between images and generate video
conda activate film
# python generate_videos.py --input_path output --folder_name $FOLDER_NAME --sec_interpolation $I --sec_freeze $F #--clean
python generate_videos_alpha.py --input_path $OUTPATH --folder_name $FOLDER_NAME --sec_interpolation $I --sec_freeze $F --clean
conda deactivate

# Super resolution on Video
conda activate diff
cd Video-Super-Resolution-ESRGAN
if [ "$M" = 'unclip' ]; then
    sr=8
else
    sr=2
fi

python inference_realesrgan_video.py -n RealESRGAN_x4plus -i $OUTPATH/$FOLDER_NAME/final_video.mp4 -o $OUTPATH/$FOLDER_NAME/ -s $sr
conda deactivate

mv $OUTPATH/$FOLDER_NAME/final_video_out.mp4 $OUTPATH/$FOLDER_NAME/$FOLDER_NAME.mp4