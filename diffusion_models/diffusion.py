import subprocess
import argparse
from print_color import print

def generate_diffusion_images(args):

    # Run diffusion
    print(f'\nGenerating images with {args.model}..', color='g')
    if args.model == 'unclip':
        h, w = 256, 256
        cmd = ["python", "diffusion_models/unclip.py",
            "--input_path", args.input_path,
            "--folder_name", args.folder_name,
            "--output_path", args.output_path,
            "--interpolation_steps", str(args.interpolation_steps),
            ]
    
    else:
        # h, w = 1024, 1024
        h, w = 720, 1280
        cmd = ["python", "diffusion_models/kandinsky.py",
            "--input_path", args.input_path,
            "--folder_name", args.folder_name,
            "--output_path", args.output_path,
            "--interpolation_steps", str(args.interpolation_steps),
            "--height", str(h),
            "--width", str(w),
            ]
        
    if args.square_crop:
        cmd.append("--square_crop")
    if args.no_originals:
        cmd.append("--no_originals")

    subprocess.call(cmd, shell=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to folder with images",
                        type=str)
    parser.add_argument("--folder_name", help="Name of the folder to read",
                        type=str)
    parser.add_argument("--output_path", help="Outputs path",
                        type=str, default="output")
    parser.add_argument("--model", help="kandinsky/unclip", type=str, default='unclip')
    parser.add_argument("--glob_pattern", help="Pattern to find files",
                        type=str, default="**/*.") 
    parser.add_argument("--interpolation_steps", help="Number of generated frames between a pair of images",
                        type=int, default=5)
    parser.add_argument("--square_crop", help="If active, crops the images in a square.", action="store_true")
    parser.add_argument("--no_originals", help="If active, don't save original images.", action="store_true")

    args = parser.parse_args()

    generate_diffusion_images(args)