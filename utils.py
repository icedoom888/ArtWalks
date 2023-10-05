import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm

def list_jpg_files_in_subfolders(root_folder):
    jpg_files = []
    
    # Walk through the root folder and its subfolders
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".jpg"):
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

