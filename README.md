# Diffusion-Videos

[[_TOC_]]

## Architecture

Generate intermidiate frames with a generative diffusion model:
![](assets/sd_interpolation.jpg)

Interpolate between all frames (original and generated) with FILM:
![](assets/frame_interpolation.jpg)



## Preparations

Install ffmpeg and av dev libs

```bash
sudo apt install ffmpeg libavformat-dev libavdevice-dev
```

1. Clone repo:
```bash
git clone git@gitlab.ethz.ch:mtc/special_project.git
cd frame-interpolation
git submodule init
git submodule update
cd ..
```

2. Create diffusers environemnt
```bash
conda create -n diffusers python=3.10 -y
conda activate diffusers
pip install -r requirements.txt
conda deactivate
```

3. Create FILM environemnt
```bash
conda create -n film python=3.9 -y
conda activate film
cd frame-interpolation
pip install -r requirements.txt
pip install tensorflow
conda deactivate
```

4. Create SR environment
```bash
conda create -n sr python=3.9
conda activate sr
pip install basicsr
# facexlib and gfpgan are for face enhancement
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
conda deactivate
```

5. (Optional) Create Music environment
```bash
conda create -n music python=3.8
conda activate srmusic
pip install torch torchvideo torchaudio
pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs
pip install tqdm
pip install matplotlib opencv-python librosa Ipython
conda deactivate
```

6. Complete environment (no tf)
```bash
conda create -p /projects/Anaconda/envs/diff python=3.8 -y
conda activate diff
conda install cudnn -y
pip install -r requirements_total.txt --no-cache-dir
pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/Anaconda/envs/diff/lib/python3.8/site-packages/tensorrt_libs
pip install "git+https://github.com/ai-forever/Kandinsky-2.git"
pip install git+https://github.com/openai/CLIP.git
conda deactivate
```

## Full Pipeline

You can run the full pipeline using the following command:
```bash
bash full_pipeline.sh $INPATH $FOLDER_NAME $S $I $F $MODEL
```
Where:
 - *INPATH* : Path to the input data folder
 - *FOLDER_NAME* : Name of the folder containing the images
 - *S* : Number of images to generate with diffusion between each pair of images
 - *I* : Number of interpolation images to generate between each pair of generated images
 - *F* : Number of seconds to freeze on each original image during the video 
 - *MODEL* : Name of the diffusion model to use (unclip/kandinsky)

## Single components
### Story Generator

Generate a random story by using a collection of immages:

```bash
python random_story_generation.py --input_path inputdata --img_num 10
```
Where:
 - *input_path* : Path to the folder containing all subfolders of images
 - *img_num* : Number of images to randomly sample
### Diffusion 

Run diffusion pipeline to interpolate between every pair of images in `input_path` (It takes roughly ~10s per pair):

```bash
python diffusion_models/diffusion.py --input_path $INPATH --folder_name $FOLDER_NAME --model $M --interpolation_steps $S 
```

This script makes use of either the [UnCLIP Image Interpolation pipeline](https://github.com/huggingface/diffusers/tree/main/examples/community#unclip-image-interpolation-pipeline) or the [Kandisnky 2 model](https://github.com/ai-forever/Kandinsky-2).
Script arguments:
```python
parser.add_argument("--input_path", help="Path to folder with images",
                        type=str)
parser.add_argument("--folder_name", help="Name of the folder to read",
                    type=str)
parser.add_argument("--output_path", help="Path to the output folder",
                    type=str, default="output")
parser.add_argument("--model", help="Choose between kandinsky/unclip model", type=str, default='unclip')
parser.add_argument("--glob_pattern", help="Pattern to find files",
                    type=str, default="**/*.") 
parser.add_argument("--interpolation_steps", help="Number of generated frames between a pair of images",
                    type=int, default=5)
parser.add_argument("--square_crop", help="If active, crops the images in a square.", action="store_true")
parser.add_argument("--no_originals", help="If active, don't save original images.", action="store_true")
```

### Frame interpolation
Generate videos interpolating between diffusion frames with (It takes roughly ~2 mins per pair):
```bash
python generate_videos.py --input_path output --folder_name $FOLDER_NAME --sec_interpolation $I --sec_freeze $F --clean
```

Script arguments:
```python
parser.add_argument("--input_path", help="Path to folder with images", default='output',
                    type=str)
parser.add_argument("--folder_name", help="Name of the folder to read",
                    type=str)
parser.add_argument("--sec_interpolation", help="Number of seconds to interpolate between images", type=int, default=10)
parser.add_argument("--sec_freeze", help="Number of seconds to freeze per original image", type=int, default=20)
parser.add_argument("--clean", help="Delete everything but the final video", action='store_true')
```

### Video Super Resolution
Generate up to 4K video by using [Video Super Resolution ESRGAN](https://github.com/saba99/Video-Super-Resolution-ESRGAN/tree/master)

```bash
python inference_realesrgan_video.py -n RealESRGAN_x4plus -i ../output/$FOLDER_NAME/final_video.mp4 -o ../output/$FOLDER_NAME/ -s $sr
```

Script arguments:
```python
parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
parser.add_argument(
    '-n',
    '--model_name',
    type=str,
    default='realesr-animevideov3',
    help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
            ' RealESRGAN_x2plus | realesr-general-x4v3'
            'Default:realesr-animevideov3'))
parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
parser.add_argument(
    '-dn',
    '--denoise_strength',
    type=float,
    default=0.5,
    help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
            'Only used for the realesr-general-x4v3 model'))
parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
parser.add_argument(
    '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')
parser.add_argument('--extract_frame_first', action='store_true')
parser.add_argument('--num_process_per_gpu', type=int, default=1)

parser.add_argument(
    '--alpha_upsampler',
    type=str,
    default='realesrgan',
    help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
parser.add_argument(
    '--ext',
    type=str,
    default='auto',
    help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
```