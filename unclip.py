import torch
from diffusers import DiffusionPipeline
from PIL import Image
import argparse
import glob
import os
import natsort
import time

from stable_diffusion_videos.utils import prepare_prompt_frame
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", help="Path to folder with images",
                    type=str)
parser.add_argument("--output_path", help="Outputs path",
                    type=str, default="output")
parser.add_argument("--glob_pattern", help="Pattern to find files",
                    type=str, default="**/*.")
parser.add_argument("--interpolation_steps", help="Number of generated frames between a pair of images",
                    type=int, default=5)
parser.add_argument("--max_image_size", help="Max image size",
                    type=int, default=256)
parser.add_argument("-D", help="Debug",
                    action="store_true")
args = parser.parse_args()

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16

print(F"Using {device=} and {dtype=}")

pipe = DiffusionPipeline.from_pretrained(
    "kakaobrain/karlo-v1-alpha-image-variations",
    torch_dtype=dtype,
    custom_pipeline="unclip_image_interpolation"
)
pipe.to(device)

extensions = ["png", "jpg", "jpeg"]
path = args.input_path
print(f"Looking in {path=} for files with {extensions=}")
artwork = [glob.glob(os.path.join(path, f"{args.glob_pattern}{extension}"), recursive=True) for extension in extensions]
artwork = natsort.natsorted([image for images in artwork for image in images])

if args.D:
    artwork = artwork[:3]

print()
print(f"Found {len(artwork)} files:")
print(artwork)
print()


generator = torch.Generator(device=device).manual_seed(42)
output_name = time.strftime("%Y%m%d-%H%M%S")
save_path = os.path.join(args.output_path, output_name)
os.makedirs(save_path)


def save_originals(images, save_path, idx):
    img1, img2 = images
    resized_img1 = prepare_prompt_frame(pil_to_tensor(img1).unsqueeze(0), args.max_image_size).squeeze(0)
    pil_img = T.ToPILImage()(resized_img1)
    pil_img.save(os.path.join(save_path, f"{idx}-{idx}.png"))

    resized_img2 = prepare_prompt_frame(pil_to_tensor(img2).unsqueeze(0),args.max_image_size).squeeze(0)
    pil_img = T.ToPILImage()(resized_img2)
    pil_img.save(os.path.join(save_path, f"{idx+1}-{idx+1}.png"))


for idx, (img1, img2) in enumerate(zip(artwork, artwork[1:])):
    
    images = [Image.open(img1).convert('RGB'), Image.open(img2).convert('RGB')]

    generated_frames = pipe(image = images ,steps = args.interpolation_steps, generator = generator)

    interpolation_save_path = os.path.join(save_path, f"{idx}-{idx+1}")
    os.makedirs(interpolation_save_path)
    save_originals(images, interpolation_save_path, idx)

    for i,image in enumerate(generated_frames.images):
        image.save(os.path.join(interpolation_save_path, f"{idx}-{idx+1}-{i}.png"))