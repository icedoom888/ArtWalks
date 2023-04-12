from stable_diffusion_videos import StableDiffusionImageVariationVideoPipeline
import os
import torch
import glob 
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", help="Path to folder with images",
                    type=str)
parser.add_argument("--glob_pattern", help="Pattern to find files",
                    type=str, default="**/*.") 
parser.add_argument("--interpolation_steps", help="Number of generated frames between a pair of images",
                    type=int, default=5)
parser.add_argument("--max_image_size", help="Max image size",
                    type=int, default=512) 
parser.add_argument("-D", help="Debug",
                    action="store_true")
args = parser.parse_args()

device = "cuda:0"
sd_pipe = StableDiffusionImageVariationVideoPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  torch_dtype=torch.float16,
  )
sd_pipe = sd_pipe.to(device)

extensions = ["png", "jpg", "jpeg"]
path = args.input_path
print(f"Looking in {path=} for files with {extensions=}")
artwork = [glob.glob(os.path.join(path, f"{args.glob_pattern}{extension}"), recursive=True) for extension in extensions]

print(f"Creating video with {len(artwork)} paintings")

seeds = [round(random.random()*100) for i in range(len(artwork))]
print(f"{seeds=}")

video_path = sd_pipe.walk(
    artwork,
    list(range(len(artwork))),
    fps=2,                      # use 5 for testing, 25 or 30 for better quality
    num_interpolation_steps=args.interpolation_steps,  # use 3-5 for testing, 30 or more for better results
    height=args.max_image_size,                 # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=args.max_image_size,                  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    guidance_scale=2,
    upsample=False, 
    include_prompts=True, #Whether to include the original paintings
    n_propmt_frames=10, #Number of frames for each original painting
    make_video=True,
)
