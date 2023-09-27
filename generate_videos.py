import glob
import argparse
import os
import subprocess
import math
import natsort
from PIL import Image
import ffmpeg
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def make_video_from_image(video_path, sec_freeze, fps=30, first=False):

    # Make output path
    out_path = os.path.join(os.path.basename(video_path), 'freeze_frame.mp4')

    # Open the input video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame count and frame rate of the input video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame number of the last frame
    last_frame_number = frame_count - 1

    # Read the first/last frame
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, last_frame = cap.read()

    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_number)
        ret, last_frame = cap.read()


    if not ret:
        raise Exception("Failed to read the last frame of the input video.")

    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    # Write the last frame to the output video multiple times to reach the desired duration
    for _ in range(int(sec_freeze * fps)):
        out.write(last_frame)

    # Release video objects
    cap.release()
    out.release()

    return out_path

def interpolate_images(args):

    folder_path = os.path.join(args.input_path, args.folder_name)

    # Load all available directories
    directories = natsort.natsorted(glob.glob(os.path.join(folder_path, "*")))
    print(f"Found the following directories to interpolate: \n{directories}")

    # Number of frames N is: N=(2^times_to_interpolate+1)
    times_to_interpolate = int(math.log2(args.frames - 1))

    # Run interpolation for all directories
    for directory in directories:
        cmd_str = f'python3 -m frame-interpolation.eval.interpolator_cli --pattern "{directory}"    --model_path frame-interpolation/pretrained_models/film_net/Style/saved_model    --times_to_interpolate {times_to_interpolate}    --output_video'
        subprocess.run(cmd_str, shell=True)

    # list all videos
    videos = []
    for directory in directories:
        vid_path = os.path.join(directory, "interpolated.mp4")
        if os.path.exists(vid_path):
            if not videos:
                pass
            videos.append(ffmpeg.input(vid_path))

            # Add still image
            frame_vid = make_video_from_image(vid_path, args.sec_freeze, first=False)
            videos.append(ffmpeg.input(frame_vid))

    
    (
        ffmpeg
        .concat(*videos)
        .output(os.path.join(folder_path, 'final_video.mp4'))
        .run()
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to folder with images", default='output',
                        type=str)
    parser.add_argument("--folder_name", help="Name of the folder to read",
                        type=str)
    parser.add_argument("--frames", help="Number of frames to interpolate between images", type=int, default=30)
    parser.add_argument("--sec_freeze", help="Number of seconds to freeze per image", type=int, default=20)

    args = parser.parse_args()

    interpolate_images(args)