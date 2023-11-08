import torch
from diffusers import DiffusionPipeline
from PIL import Image
import argparse
import glob
import os
import natsort
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore")

def prepare_prompt_frame(
        frame,
        max_size
) -> torch:
    """
    Function to reshape input prompt as a padded square preserving aspect ratio
    """
    _,_, H, W =  frame.shape
    image_max_size = H if H>W else W
    # pad to square
    if W > H:
        padding = int((image_max_size - H)/2)
        padded_frame = T.Pad((0,padding,0,padding))(frame)
    else:
        padding = int((image_max_size - W)/2)
        padded_frame = T.Pad((padding,0))(frame)
    
    #Resize to max_size
    resized_frame = T.Resize(size=(max_size,max_size))(padded_frame)
    return resized_frame

def save_originals(images, save_path, idx, square_crop=False):
    img1, img2 = images
    
    # crop images from the center
    if square_crop: 
        width, height = img1.size   

        left = (width - min(width, height))/2
        top = (height - min(width, height))/2
        right = (width + min(width, height))/2
        bottom = (height + min(width, height))/2

        img1 = img1.crop((left, top, right, bottom))

        width, height = img2.size   # Get dimensions

        left = (width - min(width, height))/2
        top = (height - min(width, height))/2
        right = (width + min(width, height))/2
        bottom = (height + min(width, height))/2

        img2 = img2.crop((left, top, right, bottom))

    resized_img1 = prepare_prompt_frame(pil_to_tensor(img1).unsqueeze(0), 256).squeeze(0)
    pil_img = T.ToPILImage()(resized_img1)
    pil_img.save(os.path.join(save_path, "%02d-%02d.png"%(idx, idx)))

    resized_img2 = prepare_prompt_frame(pil_to_tensor(img2).unsqueeze(0), 256).squeeze(0)
    pil_img = T.ToPILImage()(resized_img2)
    pil_img.save(os.path.join(save_path, "%02d-%02d.png"%(idx+1, idx+1)))

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
    # print(pipe)
    pipe.to(device)

    # Get all files
    extensions = ["png", "jpg", "jpeg", "JPG"]
    path = os.path.join(args.input_path, args.folder_name)
    # print(f"Looking in {path=} for files with {extensions=}")
    artworks = [glob.glob(os.path.join(path, f"{args.glob_pattern}{extension}"), recursive=True) for extension in extensions]
    artworks = natsort.natsorted([image for images in artworks for image in images])

    # print()
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

        interpolation_save_path = os.path.join(save_path, "%02d-%02d"%(idx, idx+1))
        os.makedirs(interpolation_save_path, exist_ok=True)
        if not args.no_originals:
            save_originals(images, interpolation_save_path, idx, square_crop=args.square_crop)

        for i,image in enumerate(generated_frames.images):
            # if True:
            #     image.save(os.path.join(interpolation_save_path, f"{idx}-{idx+1}-{i}-0.png"))
            #     image.save(os.path.join(interpolation_save_path, f"{idx}-{idx+1}-{i}-1.png"))
            image.save(os.path.join(interpolation_save_path, "%02d-%02d-%02d.png"%(idx, idx+1, i)))


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
    parser.add_argument("--square_crop", help="If active, crops the images in a square.", action="store_true")
    parser.add_argument("--no_originals", help="If active, don't save original images.", action="store_true")

    args = parser.parse_args()

    image_generation(args)