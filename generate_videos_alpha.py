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
import shutil
import warnings

warnings.filterwarnings("ignore")

def find_mp4_files(directory):
    videos = []

    mp4_files = glob.glob(os.path.join(directory, '**/*.mp4'), recursive=True)
    for f in sorted(mp4_files):
        videos.append(ffmpeg.input(f))
    
    return videos


# def concatenate_videos(video_list, output_filename):
#     clips = [VideoFileClip(video) for video in video_list]
#     final_clip = concatenate_videoclips(clips)
#     final_clip.write_videofile(output_filename, codec='libx264')
#     final_clip.close()

def make_video_from_image(video_path, sec_freeze, fps=30, first=False):

    # Open the input video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame count and frame rate of the input video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame number of the last frame
    last_frame_number = frame_count - 1

    # Read the first/last frame
    if first:
        # Make output path
        out_path = os.path.join(os.path.dirname(video_path), 'freeze_frame_first.mp4')
        ret, frame = cap.read()

    else:
        # Make output path
        out_path = os.path.join(os.path.dirname(video_path), 'freeze_frame.mp4')
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_number)
        ret, frame = cap.read()

    if not ret:
        raise Exception("Failed to read the last frame of the input video.")

    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    # Write the last frame to the output video multiple times to reach the desired duration
    for _ in range(int(sec_freeze * fps)):
        out.write(frame)

    # Release video objects
    cap.release()
    out.release()

    return out_path

def image_interpolation(args):

    folder_path = os.path.join(args.input_path, args.folder_name)

    # Load all available directories
    directories = natsort.natsorted(glob.glob(os.path.join(folder_path, "*")))
    print(f"Found the following directories to interpolate: \n{directories}\n")
    # need this to balance log2 
    carry_frames = 0

    # Run interpolation for all directories
    for directory in directories:
        artworks = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if os.path.isfile(os.path.join(directory, f))]
        vid_path = os.path.join(directory, "interpolated.mp4")
        if os.path.exists(vid_path):
            continue

        for idx, (img1, img2) in enumerate(zip(artworks, artworks[1:])):

            # Number of frames N is: N=(2^times_to_interpolate+1)
            frames = (args.sec_interpolation * 30) + carry_frames
            print(f'\nThere are {frames} frames, to interpolate between {img1} and {img2}\n')
            carry_frames = 0

            save_path = os.path.join(directory, "%02d-%02d"%(idx, idx+1))
            os.makedirs(save_path, exist_ok=True)

            shutil.copyfile(img1, os.path.join(save_path, "0.png"))
            shutil.copyfile(img2, os.path.join(save_path, "3.png"))

            cmd = ["python3", "-m", "frame-interpolation.eval.interpolator_test",
                   "--frame1", img1,
                   "--frame2", img2,
                   "--model_path", "frame-interpolation/pretrained_models/film_net/Style/saved_model",
                   "--output_frame", os.path.join(save_path, 'mid.png')]
            subprocess.run(cmd, shell=False)

            cmd = ["python3", "-m", "frame-interpolation.eval.interpolator_test",
                   "--frame1", img1,
                   "--frame2", os.path.join(save_path, 'mid.png'),
                   "--model_path", "frame-interpolation/pretrained_models/film_net/Style/saved_model",
                   "--output_frame", os.path.join(save_path, '1.png')]
            subprocess.run(cmd, shell=False)

            cmd = ["python3", "-m", "frame-interpolation.eval.interpolator_test",
                   "--frame1", os.path.join(save_path, 'mid.png'),
                   "--frame2", img2,
                   "--model_path", "frame-interpolation/pretrained_models/film_net/Style/saved_model",
                   "--output_frame", os.path.join(save_path, '2.png')]
            subprocess.run(cmd, shell=False)

            # delete mid
            os.remove(os.path.join(save_path, 'mid.png'))

            # make videos
            tmp = [os.path.join(save_path, f) for f in sorted(os.listdir(save_path)) if os.path.isfile(os.path.join(save_path, f))]
            assigned_frames = [34/100, 32/100, 34/100]
            for j, (i1, i2) in enumerate(zip(tmp, tmp[1:])):
                num_frames = (frames * assigned_frames[j])
                print(f'{assigned_frames[j]}*{frames}: ',num_frames)
                times_to_interpolate = round(math.log2(num_frames-1))

                # Update carry frames
                this_frames = num_frames - math.pow(2, times_to_interpolate) + 1
                print(f'adding {this_frames} frames to carry...\n')
                carry_frames += this_frames

                new_path = os.path.join(save_path, "%02d-%02d"%(j, j+1))
                os.makedirs(new_path, exist_ok=True)

                if os.path.exists(os.path.join(new_path, 'interpolated.mp4')):
                    print(f"{os.path.join(new_path, 'interpolated.mp4')} exists, moving on..")
                    continue

                shutil.copyfile(i1, os.path.join(new_path, "0.png"))
                shutil.copyfile(i2, os.path.join(new_path, "1.png"))

                cmd = ["python3", "-m", "frame-interpolation.eval.interpolator_cli",
                       "--pattern", new_path,
                       "--model_path", "frame-interpolation/pretrained_models/film_net/Style/saved_model",
                       "--times_to_interpolate", str(times_to_interpolate),
                       "--output_video"]
                subprocess.run(cmd, shell=False)

        videos = find_mp4_files(directory)
        print(videos)

        (
        ffmpeg
        .concat(*videos)
        .output(vid_path)
        .run()
        )

    # list all videos
    videos = []
    for directory in directories:
        vid_path = os.path.join(directory, "interpolated.mp4")
        if os.path.exists(vid_path):
            if not videos:
                # Add freeze frame on first image
                frame_vid = make_video_from_image(vid_path, args.sec_freeze, first=True)
                videos.append(ffmpeg.input(frame_vid))

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

    # Clean all directories
    if args.clean:
        to_remove = [os.path.join(folder_path, dir) for dir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, dir))]
        for dir in to_remove:
            shutil.rmtree(dir, ignore_errors=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to folder with images", type=str)
    parser.add_argument("--folder_name", help="Name of the folder to read", type=str)
    parser.add_argument("--sec_interpolation", help="Number of seconds to interpolate between images", type=int, default=10)
    parser.add_argument("--sec_freeze", help="Number of seconds to freeze per original image", type=int, default=20)
    parser.add_argument("--clean", help="Delete everything but the final video", action='store_true')


    args = parser.parse_args()

    image_interpolation(args)