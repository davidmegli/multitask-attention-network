import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_normal_to_png(normal):
    """
    normal: (H, W, 3) or (3, H, W) with values in [-1, 1]
    returns: (H, W, 3) uint8 image in [0, 255]
    """
    if normal.shape[0] == 3:
        normal = np.transpose(normal, (1, 2, 0))  # (H, W, 3)
    
    normal = (normal + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    normal = np.clip(normal, 0, 1)
    normal = (normal * 255).astype(np.uint8)
    return normal

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    for f in tqdm(npy_files, desc="Converting normals to PNG"):
        normal_path = os.path.join(input_dir, f)
        normal = np.load(normal_path)  # shape: (3, H, W) or (H, W, 3)
        
        normal_img = convert_normal_to_png(normal)
        
        img_name = os.path.splitext(f)[0] + ".png"
        img_path = os.path.join(output_dir, img_name)
        Image.fromarray(normal_img).save(img_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with .npy normal files")
    parser.add_argument("--output_dir", required=True, help="Where to save .png images")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
