import glob
import argparse
import os
import subprocess
import math
import natsort
from PIL import Image
import ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", help="Path to folder with images",
                    type=str)
parser.add_argument("--frames", help="Number of frames to interpolate between images",
                    type=int, default=360)
args = parser.parse_args()

directories = natsort.natsorted(glob.glob(os.path.join(args.input_path, "*")))
print(directories)

# Number of frames N is: N=(2^times_to_interpolate+1)
times_to_interpolate = int(math.log2(args.frames - 1))

# for directory in directories:
#     cmd_str = f'python3 -m frame-interpolation.eval.interpolator_cli --pattern "{directory}"    --model_path frame-interpolation/pretrained_models/film_net/Style/saved_model    --times_to_interpolate {times_to_interpolate}    --output_video'
#     subprocess.run(cmd_str, shell=True)

# for directory in directories:
#     files = natsort.natsorted(glob.glob(os.path.join(directory, "*")))
#     print(directory)
#     for file in files:
#         print(file)
#         im = Image.open(file).convert('RGB')
#         print(im.size)
#         print(im.mode)
#     print()

videos = [ffmpeg.input(os.path.join(directory, "interpolated.mp4")) for directory in directories]
(
    ffmpeg
    .concat(*videos)
    .output('out.mp4')
    .run()
)