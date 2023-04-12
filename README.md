# Diffusion-Videos
> Note: I have not gone through all these steps, so I might have missed something or there might be typos. Please, update this README if something is not correct.

## Steps

0. Clone repo:
```bash
git clone git@gitlab.ethz.ch:mtc/special_project.git
cd frame-interpolation
git submodule init
git submodule update
cd ..
```

1. Create diffusers environemnt
```bash
conda create -n diffusers python=3.10 -y
pip install -r requirements.txt
conda activate diffusers
```

2. Run unCLIP pipeline to interpolate between every pair of images in `input_path`:

> It takes roughly ~10s per pair

```bash
python unclip.py --input_path ../mtc-gtc/
```

This script makes use of the [UnCLIP Image Interpolation pipeline](https://github.com/huggingface/diffusers/tree/main/examples/community#unclip-image-interpolation-pipeline). It takes every image in the input_path, [natsort](https://github.com/SethMMorton/natsort/wiki/How-Does-Natsort-Work%3F) it and create n interpolations between every pair of images. Script arguments:
```python
parser.add_argument("--input_path", help="Path to folder with images",
                    type=str)
parser.add_argument("--output_path", help="Outputs path",
                    type=str, default="output")
parser.add_argument("--glob_pattern", help="Pattern to find files",
                    type=str, default="**/*.") 
parser.add_argument("--interpolation_steps", help="Number of generated frames between a pair of images",
                    type=int, default=5)
parser.add_argument("--max_image_size", help="Max image size",
                    type=int, default=256) # This needs to be fixed to 256 because the model outputs are fixed to 256x256
parser.add_argument("-D", help="Debug",
                    action="store_true")
```

3. Create FILM environemnt
```bash
conda create -n film python=3.9 -y
cd frame-interpolation
pip install -r requirements.txt
pip install tensorflow
conda activate film
```

3.1. Download the pretrained models: https://github.com/google-research/frame-interpolation#pre-trained-models

4. Generate videos with:

> It takes roughly ~2 mins per pair

```bash
python generate_videos.py --input_path output/20230412-151520
```
Script arguments:
```python
parser.add_argument("--input_path", help="Path to folder with images",
                    type=str)
parser.add_argument("--frames", help="Number of frames to interpolate between images",
                    type=int, default=360) # This is an approximation, because the number of frames N is: N=(2^times_to_interpolate+1). times_to_interpolate is the argument to the script, which must be an int (so there will be a bit more/less number of frames probably.)
```

## `variations.py`

This is another pipeline I've been experimenting with is based on the ImageVariationPipeline([more info here](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)) and the [Stable Diffusion Interpolation](https://github.com/huggingface/diffusers/tree/main/examples/community#stable-diffusion-interpolation) (only text based, which was adapted to accept two images as input and interpolate between them). It has everything already integrated to go from frames to videos, but without inter-frame interpolation using FILM. Results are pretty cool but I found that they are a bit worse than the unclip pipeline.

## Future steps

There are different lines to follow and finish the project:
- Currently, generated images are fixed to 256x256. This should be rather easy to change to scale up the images, and then apply superresolution.
- We can experiemnt doing inter-frame interpolation with FILM before or after superresolution.
- Fine-tuning stable diffusion model on the style of the artist: https://huggingface.co/docs/diffusers/training/dreambooth
- Explore other methods like prompt-to-prompt: https://github.com/google/prompt-to-prompt/
