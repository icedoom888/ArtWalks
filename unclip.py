import torch
from diffusers import DiffusionPipeline
from PIL import Image

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16

pipe = DiffusionPipeline.from_pretrained(
    "kakaobrain/karlo-v1-alpha-image-variations",
    torch_dtype=dtype,
    custom_pipeline="unclip_image_interpolation"
)
pipe.to(device)
path_1 = "/home/comp/Burki/Artworks/PortraitFemale/Portrait Female/1.jpg"
path_2 = "/home/comp/Burki/Artworks/PortraitFemale/Portrait Female/106.jpg"
path_3 = "/home/comp/Burki/Artworks/PortraitFemale/Portrait Female/118.jpg"
# path_3 = "/home/comp/Burki/Artworks/DonQuijote/Don Quijote/41.jpg"
# path_4 = "/home/comp/Burki/Artworks/DonQuijote/Don Quijote/151.jpg"
artwork = [path_1, path_2]
# images = [Image.open('./starry_night.jpg'), Image.open('./flowers.jpg')]
images = [Image.open(art) for art in artwork]
#For best results keep the prompts close in length to each other. Of course, feel free to try out with differing lengths.
generator = torch.Generator(device=device).manual_seed(42)

output = pipe(image = images ,steps = 5, generator = generator)

for i,image in enumerate(output.images):
    image.save('test/pf1-2_%s.jpg' % i)

artwork = [path_2, path_3]
# images = [Image.open('./starry_night.jpg'), Image.open('./flowers.jpg')]
images = [Image.open(art) for art in artwork]
#For best results keep the prompts close in length to each other. Of course, feel free to try out with differing lengths.
generator = torch.Generator(device=device).manual_seed(42)

output = pipe(image = images ,steps = 5, generator = generator)

for i,image in enumerate(output.images):
    image.save('test/pf2-3_%s.jpg' % i)