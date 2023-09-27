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
import warnings
warnings.filterwarnings("ignore")


def save_originals(images, save_path, idx):
    img1, img2 = images
    resized_img1 = prepare_prompt_frame(pil_to_tensor(img1).unsqueeze(0), args.max_image_size).squeeze(0)
    pil_img = T.ToPILImage()(resized_img1)
    pil_img.save(os.path.join(save_path, f"{idx}-{idx}.png"))

    resized_img2 = prepare_prompt_frame(pil_to_tensor(img2).unsqueeze(0),args.max_image_size).squeeze(0)
    pil_img = T.ToPILImage()(resized_img2)
    pil_img.save(os.path.join(save_path, f"{idx+1}-{idx+1}.png"))


def image_generation(args):

    # get available device
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    print(F"Using {device=} and {dtype=}")

    # Load pretrained model
    pipe = DiffusionPipeline.from_pretrained(
        "kakaobrain/karlo-v1-alpha-image-variations",
        torch_dtype=dtype,
        custom_pipeline="unclip_image_interpolation"
    )
    print(pipe)
    pipe.to(device)

    # Get all files
    extensions = ["png", "jpg", "jpeg"]
    path = os.path.join(args.input_path, args.folder_name)
    print(f"Looking in {path=} for files with {extensions=}")
    artworks = [glob.glob(os.path.join(path, f"{args.glob_pattern}{extension}"), recursive=True) for extension in extensions]
    artworks = natsort.natsorted([image for images in artworks for image in images])

    print()
    print(f"Found {len(artworks)} images:")
    for aw in artworks:
        print(aw)
    print()

    # Make outfolder
    generator = torch.Generator(device=device)#.manual_seed(42)
    output_name = args.folder_name
    save_path = os.path.join(args.output_path, output_name)
    os.makedirs(save_path, exist_ok=True)

    # generate images for each couple
    for idx, (img1, img2) in enumerate(zip(artworks, artworks[1:])):
        
        images = [Image.open(img1).convert('RGB'), Image.open(img2).convert('RGB')]

        generated_frames = pipe(image=images,
                                steps=args.interpolation_steps, 
                                generator=generator)

        interpolation_save_path = os.path.join(save_path, f"{idx}-{idx+1}")
        os.makedirs(interpolation_save_path, exist_ok=True)
        save_originals(images, interpolation_save_path, idx)

        for i,image in enumerate(generated_frames.images):
            # if True:
            #     image.save(os.path.join(interpolation_save_path, f"{idx}-{idx+1}-{i}-0.png"))
            #     image.save(os.path.join(interpolation_save_path, f"{idx}-{idx+1}-{i}-1.png"))
            image.save(os.path.join(interpolation_save_path, f"{idx}-{idx+1}-{i}.png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to folder with images",
                        type=str)
    parser.add_argument("--folder_name", help="Name of the folder to read",
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
    args = parser.parse_args()

    image_generation(args)