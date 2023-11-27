from PIL import Image
import argparse
import glob
import os
import natsort
import math
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor 
import torchvision.transforms as T
from tqdm import tqdm
from kandinsky2 import get_kandinsky2
import sys
sys.path.append("/projects/iaivc/special_project/diffusion_models")
from weight_samplers import get_weight_sampler
from PIL import Image, ImageOps
from diffusers import AutoPipelineForInpainting, KandinskyV22InpaintPipeline
from diffusers.models import UNet2DConditionModel

import matplotlib.pyplot as plt

def make_masks(image, target_img_size):

    img_size = image.size
    mask_size = (min(target_img_size), min(target_img_size))
    image = np.asarray(image)

    delta_width = target_img_size[0] - img_size[0]
    delta_height = target_img_size[1] - img_size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2

    mask_1 = Image.new("RGB", mask_size, color=(255,255,255))
    init_img_1 = Image.new("RGB", mask_size, color=(255,255,255))
    for i in range(pad_width, mask_size[0]):
        for j in range(pad_height, mask_size[1]):
            mask_1.putpixel((i,j), (0, 0, 0))
            init_img_1.putpixel((i,j), tuple(image[j-pad_height, i-pad_width]))

    mask_2 = Image.new("RGB", mask_size, color=(255,255,255))
    init_img_2 = Image.new("RGB", mask_size, color=(255,255,255))
    for i in range(0, mask_size[0]- pad_width):
        for j in range(0, mask_size[1] - pad_height):
            mask_2.putpixel((i,j), (0, 0, 0))
            init_img_2.putpixel((i,j), tuple(image[j+target_img_size[1]-mask_size[1]-pad_height-1, i+target_img_size[0]-mask_size[0]-pad_width-1]))

    return [init_img_1, init_img_2], [mask_1, mask_2]

def outpaint(model, image, target_img_size):

    image.thumbnail(target_img_size)

    unet = UNet2DConditionModel.from_pretrained('kandinsky-community/kandinsky-2-2-decoder-inpaint', subfolder='unet').to(torch.float16).to('cuda')
    decoder = KandinskyV22InpaintPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-decoder-inpaint', unet=unet, torch_dtype=torch.float16).to('cuda')

    # embedd image
    image_emb = model.prior.interpolate(images_and_prompts=['a neutral background', image], 
                                      weights=[0.32, 0.68],
                                      num_inference_steps=25, 
                                      num_images_per_prompt=1,
                                      guidance_scale=4, 
                                      negative_prompt='')
    negative_emb = model.prior(prompt='', num_inference_steps=25, num_images_per_prompt=1, guidance_scale=4)

    # load base and mask image
    init_images, mask_images = make_masks(image, target_img_size)
    
    gen_1 = decoder(image_embeds=image_emb.image_embeds,
                    negative_image_embeds=negative_emb.negative_image_embeds,
                    num_inference_steps=50, 
                    height=512,
                    width=512, 
                    guidance_scale=4,
                    image=init_images[0], 
                    mask_image=mask_images[0]).images[0].resize(mask_images[0].size)
    
    gen_2 = decoder(image_embeds=image_emb.image_embeds, 
                    negative_image_embeds=negative_emb.negative_image_embeds,
                    num_inference_steps=50, 
                    height=512,
                    width=512, 
                    guidance_scale=4,
                    image=init_images[1], 
                    mask_image=mask_images[1]).images[0].resize(mask_images[1].size)

    gen_images = [gen_1, gen_2]

    # outpainted image
    combined_image = Image.new("RGB", target_img_size)

    # Calculate the position to paste the first image
    x_offset = 0
    y_offset = 0

    # Paste the first image onto the combined image
    combined_image.paste(gen_images[0], (x_offset, y_offset))

    # Calculate the position to paste the second image
    x_offset = combined_image.width - gen_images[1].width
    y_offset = combined_image.height - gen_images[1].height

    # Paste the second image onto the combined image
    combined_image.paste(gen_images[1], (x_offset, y_offset))

    return combined_image


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def save_originals(model, images, save_path, idx, same_ratio=False, square_crop=False, target_img_size=None, outpaint=True):
  
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
    
    if not same_ratio:
        if outpaint:
            pil_img1 = outpaint(model, img1, target_img_size)
            pil_img1.save(os.path.join(save_path, "%02d-%02d.png"%(idx, idx)))

            pil_img2 = outpaint(model, img2, target_img_size)
            pil_img2.save(os.path.join(save_path, "%02d-%02d.png"%(idx+1, idx+1)))
        
        else:
            pil_img1 = resize_with_padding(img1, expected_size=target_img_size)
            pil_img1.save(os.path.join(save_path, "%02d-%02d.png"%(idx, idx)))

            pil_img2 = resize_with_padding(img2, expected_size=target_img_size)
            pil_img2.save(os.path.join(save_path, "%02d-%02d.png"%(idx+1, idx+1)))
    
    else:
        img1.save(os.path.join(save_path, "%02d-%02d.png"%(idx, idx)))
        img2.save(os.path.join(save_path, "%02d-%02d.png"%(idx+1, idx+1)))

def prompt2image(model, style_prompt, content_prompt, w=512, h=512, num_imgs=5, decoder_steps=50, prior_steps=25, decoder_guidance_scale=4, prior_guidance_scale=4):
    if style_prompt != "":
        prompt = content_prompt + f', {style_prompt}'
    else:
        prompt = content_prompt 
    print('Generating: ', prompt, '...')
    print(f'with the following configuration: decoder_steps={decoder_steps}, prior_steps={prior_steps}, decoder_guidance_scale={decoder_guidance_scale}, prior_guidance_scale={prior_guidance_scale}')
    
    images = model.generate_text2img(prompt, batch_size=num_imgs, h=h, w=w, 
                                     decoder_steps=decoder_steps, 
                                     prior_steps=prior_steps, 
                                     decoder_guidance_scale=decoder_guidance_scale, 
                                     prior_guidance_scale=prior_guidance_scale)
    
    gen_img = images[0]
    gen_w, gen_h = gen_img.size

    # Fix errors in generation size
    left = (gen_w - w)/2
    top = (gen_h - h)/2
    right = (gen_w + w)/2
    bottom = (gen_h + h)/2

    gen_img = gen_img.crop((left, top, right, bottom))
    return [gen_img]

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


def incremental_interpolation(model, img1, img2, interpolation_steps, h, w, weight_sampler_name='squeezed_sinh', decoder_guidance_scale=4, use_neg_prompts=False):
    # weight_sampler_name = 'linear' 
    weigth_sampler = get_weight_sampler(weight_sampler_name)
    incr = 1/(interpolation_steps)
    generated_images = []
    images_texts = ['', img1, img2, '']

    if use_neg_prompts:
        neg_prompt = 'writing,artists_name,name,russian,tag,watermark,signature,words,sentences,text,blurry,cropped,error,extra_structure,extra_frame,jpeg_artifacts,low_quality,black_borders,out_of_frame,lowres,ugly,username,uta,worst_quality'
    else:
        neg_prompt = ''

    for i in range(interpolation_steps):
        curr_weight = round(weigth_sampler(i*incr), 3)
        weights = [0.0, 1-curr_weight, curr_weight, 0.0]
        print(f'\nMixture at {i*incr} with weights: ', weights)
        images = model.mix_images(
            images_texts, 
            weights, 
            decoder_steps=50,
            decoder_guidance_scale=decoder_guidance_scale,
            batch_size=1, 
            h=h, w=w,
            negative_prior_prompt=neg_prompt,
        )

        gen_img = images[0]
        gen_w, gen_h = gen_img.size

        # Fix errors in generation size
        # gen_img = gen_img.resize((w, h), Image.Resampling.LANCZOS)
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

    if args.folder_name.split('_')[-1] == 'cropped':
        path = os.path.join(args.input_path, '_'.join(args.folder_name.split('_')[:-1]))
        og_artworks = [glob.glob(os.path.join(path, f"{args.glob_pattern}{extension}"), recursive=True) for extension in extensions]
        og_artworks = natsort.natsorted([image for images in og_artworks for image in images])
    
    else:
        og_artworks = artworks

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
    for idx, (img1_path, img2_path, og_img1_path, og_img2_path) in tqdm(enumerate(zip(artworks, artworks[1:], og_artworks, og_artworks[1:])), total=len(artworks)):
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        og_img1 = Image.open(og_img1_path)
        og_img2 = Image.open(og_img2_path)

        # Load og images
        images = [img1.convert('RGB'), img2.convert('RGB')]

        # Generate interpolation frames
        if idx == len(artworks) - 2:
            generated_frames = incremental_interpolation(model, 
                                                         og_img1, 
                                                         og_img2, 
                                                         args.interpolation_steps + args.accumulated, 
                                                         args.height, 
                                                         args.width, 
                                                         args.weight_sampler,
                                                         args.decoder_guidance_scale,
                                                         args.use_neg_prompts)
        else:
            generated_frames = incremental_interpolation(model, 
                                                         og_img1, 
                                                         og_img2, 
                                                         args.interpolation_steps, 
                                                         args.height, 
                                                         args.width, 
                                                         args.weight_sampler,
                                                         args.decoder_guidance_scale,
                                                         args.use_neg_prompts)
        
        # Save generated images
        interpolation_save_path = os.path.join(save_path, "%02d-%02d"%(idx, idx+1))
        os.makedirs(interpolation_save_path, exist_ok=True)

        if not args.no_originals:
            # remove the reconstructed original images, exept Burki
            if not args.output_path == 'output' and not args.output_path == '/media/data-storage/iaivc':
                del generated_frames[0]
                del generated_frames[-1]
                
            gen_size = generated_frames[0].size
            same_ratio = gen_size == img1.size
            save_originals(model, images, interpolation_save_path, idx, same_ratio=same_ratio, square_crop=args.square_crop, target_img_size=(args.width, args.height), outpaint=args.outpaint)

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
    parser.add_argument("--accumulated", help="Extra images to be added in the last step.", 
                        type=int, default=0)
    parser.add_argument("--weight_sampler", help="name of the weight sampler to use (linear/sinh/squeezed_sinh/logit/squeezed_logit)",
                        type=str, default='sinh')
    parser.add_argument("--decoder_guidance_scale", help="decoder guidance scale: how strong is affected by image (1 to 10)",
                        type=int, default=4)
    parser.add_argument("--use_neg_prompts", help="If active, then use negative prompts", action="store_true")
    parser.add_argument("--height", help="Image height", type=int, required=False)
    parser.add_argument("--width", help="Image width", type=int, required=False)
    parser.add_argument("--square_crop", help="If active, crops the images in a square.", action="store_true")
    parser.add_argument("--no_originals", help="If active, don't save original images.", action="store_true")
    parser.add_argument("--outpaint", help="If active, outpaints the image using generative model.", action="store_true")


    args = parser.parse_args()

    image_generation(args)