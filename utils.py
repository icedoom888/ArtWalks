import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import shutil
import argparse

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def list_jpg_files_in_subfolders(root_folder):
    jpg_files = []
    
    # Walk through the root folder and its subfolders
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".JPG", ".jpeg", ".PNG")):
                jpg_files.append(os.path.join(root, file))
    
    return jpg_files

def list_audio_files_in_subfolders(root_folder):
    audio_files = []
    
    # Walk through the root folder and its subfolders
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav')):
                audio_files.append(os.path.join(root, file))
    
    return audio_files

def add_audio_to_video(video_tmp_path, audio_path, output_path):
    cmd = 'ffmpeg -hide_banner -loglevel error -y -i "' + video_tmp_path + '" -i "' + \
        audio_path + '" "' + output_path + '"'
    subprocess.call(cmd, shell=True)
    os.remove(video_tmp_path)  # remove the template video

def stack2videos(path_vid1, path_vid2, audio_path, fps, out_path):

    # Read videos
    reader1 = cv2.VideoCapture(path_vid1)
    reader2 = cv2.VideoCapture(path_vid2)
    
    # Make video writer
    width = int(reader1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # get ratio of vid2
    w2 = int(reader2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(reader2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio = h2/w2
    new_h = int(ratio*width)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    tmp_path = os.path.join(os.path.dirname(out_path), 'tmp.mp4')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height + new_h)) 
    # print('t: ', (width, height + new_h, 3))
    # Stack images 
    limit = min(int(reader1.get(cv2.CAP_PROP_FRAME_COUNT)), int(reader2.get(cv2.CAP_PROP_FRAME_COUNT)))

    for i in tqdm(range(limit)):
        _, frame1 = reader1.read()
        _, frame2 = reader2.read()
        frame2 = cv2.resize(frame2, (width, new_h))
        # print(frame2.shape)
        img = np.vstack((frame1, frame2))
        cv2.waitKey(1)
        writer.write(img)
    
    writer.release()
    reader1.release()
    reader2.release()
    cv2.destroyAllWindows()

    # Add audio
    output_path = os.path.join(out_path)
    add_audio_to_video(tmp_path, audio_path, output_path)

def stack2video_horizontal(path_vid1, path_vid2, audio_path, fps, out_path):

    # Read videos
    reader1 = cv2.VideoCapture(path_vid1)
    reader2 = cv2.VideoCapture(path_vid2)
    
    # Make video writer
    width = int(reader1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # get ratio of vid2
    w2 = int(reader2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(reader2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio = w2/h2
    new_w = int(ratio*height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    tmp_path = os.path.join(os.path.dirname(out_path), 'tmp.mp4')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width + new_w, height)) 
    # print('t: ', (width, height + new_h, 3))
    # Stack images 
    limit = min(int(reader1.get(cv2.CAP_PROP_FRAME_COUNT)), int(reader2.get(cv2.CAP_PROP_FRAME_COUNT)))

    for i in tqdm(range(limit)):
        _, frame1 = reader1.read()
        _, frame2 = reader2.read()
        frame2 = cv2.resize(frame2, (new_w, height))
        # print(frame2.shape)
        img = np.hstack((frame1, frame2))
        cv2.waitKey(1)
        writer.write(img)
    
    writer.release()
    reader1.release()
    reader2.release()
    cv2.destroyAllWindows()

    os.path.join(out_path)
    # Add audio
    if audio_path is not None:
        add_audio_to_video(tmp_path, audio_path, out_path)
    
    else:
        print(tmp_path, out_path)
        shutil.copyfile(tmp_path, out_path)

def plot_images(images, prompt, n_images):
    from math import ceil
    rows, columns = 2, ceil(n_images/2)
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(4*columns,4*rows))
    fig.patch.set_facecolor('#1f1f1f')
    fig.suptitle(prompt.replace(',', '\n', 1), fontsize=12, color='white')
    img_count=0
    for i in range(rows):
        for j in range(columns):        
            if img_count < len(images):
                axes[i, j].set_title(f'Image {img_count+1}', fontsize=8, color='white')
                axes[i, j].axis('off')
                axes[i, j].imshow(images[img_count])
                img_count+=1
    plt.show()

def compute_total_time(n, s, i, f):
    total = n * f + (n-1)*(s+1)*i
    return total

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to folder with images", type=str)
    parser.add_argument("--folder_name", help="Name of the folder to read", type=str)    
    parser.add_argument("--s", help="Number of diffusion images", type=int, default=5)
    parser.add_argument("--i", help="Number of seconds to interpolate between images", type=int, default=10)
    parser.add_argument("--f", help="Number of seconds to freeze per original image", type=int, default=20)

    args = parser.parse_args()

    n = len(os.listdir(os.path.join(args.input_path, args.folder_name)))

    total = compute_total_time(n, args.s, args.i, args.f)

    print('\n')
    print(f'The video {args.folder_name} will be %02d:%02ds long'%(int(total//60), int(total%60)))
    print('\n')


