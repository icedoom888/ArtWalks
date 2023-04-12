# Diffusion-Videos

## Steps

1. Create diffusers environemnt
```bash
conda create -n diffusers python=3.10 -y
pip install -r requirements.txt
conda activate diffusers
```

2. Run unCLIP pipeline to interpolate between every pair of images in `input_path`:

```bash
python unclip.py --input_path ../mtc-gtc/
```

3. Create film environemnt
```bash
conda create -n film python=3.9 -y
cd frame-interpolation
pip install -r requirements.txt
pip install tensorflow
conda activate film
```

3.1. Download the pretrained models: https://github.com/google-research/frame-interpolation#pre-trained-models

4. Generate videos with:
```bash
python generate_videos.py --input_path output/20230412-151520
```