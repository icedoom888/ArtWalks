from PIL import Image
import argparse
import glob
import os
import natsort
import math
from kandinsky2 import get_kandinsky2
import warnings
warnings.filterwarnings("ignore")

def save_originals(images, save_path, idx, square_crop=False, max_size=None):
    from unclip import prepare_prompt_frame
    from torchvision.transforms.functional import pil_to_tensor 
    import torchvision.transforms as T

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
    
    resized_img1 = prepare_prompt_frame(pil_to_tensor(img1).unsqueeze(0), max_size=max_size).squeeze(0)
    pil_img1 = T.ToPILImage()(resized_img1)
    pil_img1.save(os.path.join(save_path, "%02d-%02d.png"%(idx, idx)))

    resized_img2 = prepare_prompt_frame(pil_to_tensor(img2).unsqueeze(0), max_size=max_size).squeeze(0)
    pil_img2 = T.ToPILImage()(resized_img2)
    pil_img2.save(os.path.join(save_path, "%02d-%02d.png"%(idx+1, idx+1)))

def incremental_interpolation(model, img1, img2, interpolation_steps, h, w):

    incr = 1/interpolation_steps
    generated_images = []
    images_texts = ['', img1, img2, '']
    
    for i in range(interpolation_steps):
        weights = [0.0, 1-(i*incr), (i*incr), 0.0]
        images = model.mix_images(
            images_texts, 
            weights, 
            decoder_steps=50,
            batch_size=1, 
            h=h, w=w,
        )

        gen_img = images[0]
        gen_w, gen_h = gen_img.size

        # Fix errors in generation size
        left = (gen_w - w)/2
        top = (gen_h - h)/2
        right = (gen_w + w)/2
        bottom = (gen_h + h)/2

        gen_img = gen_img.crop((left, top, right, bottom))
        generated_images.append(gen_img)

    return generated_images

def recursive_interpolation(model, img1, img2, interpolation_steps):
    # interpolation_steps had to be: (2^n) - 1

    # recursion close
    if interpolation_steps == 0:
        return []
    
    images_texts = ['', img1, img2, '']
    weights = [0.0, 0.5, 0.5, 0.0]
    images = model.mix_images(
        images_texts, 
        weights, 
        decoder_steps=50,
        batch_size=1, 
        #guidance_scale=5,
        h=960, w=540,
        #sampler='p_sampler', 
        #prior_cf_scale=4,
        #prior_steps="5"
    )

    gen_img = images[0]
    
    return recursive_interpolation(model, img1, gen_img, int(interpolation_steps // 2)) +\
           [gen_img] +\
           recursive_interpolation(model, gen_img, img2, int(interpolation_steps // 2))

def size_interpolation(model, img1, img2, interpolation_steps):
    incr = 1/interpolation_steps
    generated_images = []
    images_texts = ['', img1, img2, '']

    w_diff = (img2.size[0] - img1.size[0])/interpolation_steps
    h_diff = (img2.size[1] - img1.size[1])/interpolation_steps
    
    for i in range(interpolation_steps):
        # interpolation
        weights = [0.0, 1-(i*incr), (i*incr), 0.0]
        w = math.floor(img1.size[0] + i*w_diff)
        h = math.floor(img1.size[1] + i*h_diff)

        # mix images
        images = model.mix_images(
            images_texts, 
            weights, 
            decoder_steps=50,
            batch_size=1, 
            h=h, w=w,
        )

        gen_img = images[0]
        gen_w, gen_h = gen_img.size

        # Fix errors in generation size
        left = (gen_w - w)/2
        top = (gen_h - h)/2
        right = (gen_w + w)/2
        bottom = (gen_h + h)/2

        gen_img = gen_img.crop((left, top, right, bottom))
        generated_images.append(gen_img)

    return generated_images

def image_generation(args):

    # Get all files
    extensions = ["png", "jpg", "jpeg"]
    path = os.path.join(args.input_path, args.folder_name)
    artworks = [glob.glob(os.path.join(path, f"{args.glob_pattern}{extension}"), recursive=True) for extension in extensions]
    artworks = natsort.natsorted([image for images in artworks for image in images])

    # Print artworks
    print(f"Found {len(artworks)} images:")
    for aw in artworks:
        print(aw)
    print()

    # Make outfolder
    output_name = args.folder_name
    save_path = os.path.join(args.output_path, output_name)
    os.makedirs(save_path, exist_ok=True)

    # Get model
    model = get_kandinsky2('cuda', task_type='text2img', model_version='2.2', use_flash_attention=False)

    # generate images for each couple
    for idx, (img1_path, img2_path) in enumerate(zip(artworks, artworks[1:])):

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # TODO: remove
        img1 = img1.resize((math.floor(img1.size[0]/2), math.floor(img1.size[1]/2)), Image.Resampling.LANCZOS)
        img2 = img2.resize((math.floor(img2.size[0]/2), math.floor(img2.size[1]/2)), Image.Resampling.LANCZOS)

        # Load og images
        images = [img1.convert('RGB'), img2.convert('RGB')]

        # Generate interpolation frames
        generated_frames = incremental_interpolation(model, img1, img2, args.interpolation_steps, args.height, args.width)
        
        # Save generated images
        interpolation_save_path = os.path.join(save_path, "%02d-%02d"%(idx, idx+1))
        os.makedirs(interpolation_save_path, exist_ok=True)
        if not args.no_originals:
            save_originals(images, interpolation_save_path, idx, square_crop=args.square_crop, max_size=max(args.height, args.width))

        for i,image in enumerate(generated_frames):
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
    parser.add_argument("--height", help="Image height", type=int, required=False)
    parser.add_argument("--width", help="Image width", type=int, required=False)
    parser.add_argument("--square_crop", help="If active, crops the images in a square.", action="store_true")
    parser.add_argument("--no_originals", help="If active, don't save original images.", action="store_true")

    args = parser.parse_args()

    image_generation(args)