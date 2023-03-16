from stable_diffusion_videos import StableDiffusionImageVariationVideoPipeline
from PIL import Image
from torchvision import transforms
import torch
import glob 

device = "cuda:0"
sd_pipe = StableDiffusionImageVariationVideoPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  torch_dtype=torch.float16,
  )
sd_pipe = sd_pipe.to(device)


###### GENERATE ONE IMAGE FROM PROPMT
# im = Image.open("/home/comp/Burki/Artworks/PortraitFemale/Portrait Female/8.jpg")

# tform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize(
#         (224, 224),
#         interpolation=transforms.InterpolationMode.BICUBIC,
#         antialias=False,
#         ),
#     transforms.Normalize(
#       [0.48145466, 0.4578275, 0.40821073],
#       [0.26862954, 0.26130258, 0.27577711]),
# ])
# inp = tform(im).to(device).unsqueeze(0)

# out = sd_pipe(inp, guidance_scale=3, num_images_per_prompt=5)

# for i,im in enumerate(out["images"]):
#     im.save(f"result_{i}.jpg")

artwork = glob.glob("/home/comp/Burki/Artworks/**/*.jpg", recursive=True)[:5]
print(f"Creating video with {len(artwork)} paintings")
# path_1 = "/home/comp/Burki/Artworks/PortraitFemale/Portrait Female/8.jpg"
# path_2 = "/home/comp/Burki/Artworks/PortraitFemale/Portrait Female/106.jpg"
# path_3 = "/home/comp/Burki/Artworks/DonQuijote/Don Quijote/41.jpg"
# path_4 = "/home/comp/Burki/Artworks/DonQuijote/Don Quijote/151.jpg"

video_path = sd_pipe.walk(
    artwork,
    list(range(len(artwork))),
    fps=5,                      # use 5 for testing, 25 or 30 for better quality
    num_interpolation_steps=5,  # use 3-5 for testing, 30 or more for better results
    height=540,                 # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=960,                  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    guidance_scale=4,
    upsample=True
)
