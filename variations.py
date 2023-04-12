from stable_diffusion_videos import StableDiffusionImageVariationVideoPipeline
from PIL import Image
from torchvision import transforms
import torch
import glob 
import random

device = "cuda:0"
sd_pipe = StableDiffusionImageVariationVideoPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  torch_dtype=torch.float16,
  )
sd_pipe = sd_pipe.to(device)

# artwork = glob.glob("/home/comp/Burki/Artworks/**/*.jpg", recursive=True)
# artwork = glob.glob("/home/comp/Burki/mtc-gtc/**/*.png", recursive=True)
# artwork = glob.glob("/home/comp/Burki/sd-video-variations/training_images/**/*.jpg", recursive=True)

path_1 = "../Artworks/Portrait Female/1.jpg"
path_2 = "../Artworks/Portrait Female/106.jpg"
path_3 = "../Artworks/Portrait Female/118.jpg"
# path_3 = "/home/comp/Burki/Artworks/DonQuijote/Don Quijote/41.jpg"
# path_4 = "/home/comp/Burki/Artworks/DonQuijote/Don Quijote/151.jpg"

artwork = [path_1, path_2, path_3]
print(f"Creating video with {len(artwork)} paintings")

seeds = [round(random.random()*100) for i in range(len(artwork))]
print(f"{seeds=}")

video_path = sd_pipe.walk(
    artwork,
    list(range(len(artwork))),
    fps=2,                      # use 5 for testing, 25 or 30 for better quality
    num_interpolation_steps=5,  # use 3-5 for testing, 30 or more for better results
    height=512,                 # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=512,                  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    guidance_scale=2,
    upsample=False, 
    include_prompts=True, #Whether to include the original paintings
    n_propmt_frames=10, #Number of frames for each original painting
    make_video=False,
)
