'''
File Name:  resize_nyuv2_dataset.py
Author:     David Megli
Created:    2025-03-22
Description: This file contains the function to resize the NYUv2 dataset
'''

import os
import numpy as np
from tqdm import tqdm
import argparse
import fnmatch
from scipy.ndimage import zoom

def center_crop_to_multiple(arr, multiple=32):
    h, w = arr.shape[:2]
    new_h = (h // multiple) * multiple
    new_w = (w // multiple) * multiple
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    if arr.ndim == 3:
        return arr[top:top+new_h, left:left+new_w, :]
    else:
        return arr[top:top+new_h, left:left+new_w]


def resize_and_save(src_path, dst_path, scale_factor):
    for split in ['train', 'val']:
        for subfolder in ['image', 'label', 'depth', 'normal']:
            src_folder = os.path.join(src_path, split, subfolder)
            dst_folder = os.path.join(dst_path, split, subfolder)
            os.makedirs(dst_folder, exist_ok=True)

            files = fnmatch.filter(os.listdir(src_folder), '*.npy')
            for fname in tqdm(files, desc=f"{split}/{subfolder}"):
                src_file = os.path.join(src_folder, fname)
                dst_file = os.path.join(dst_folder, fname)

                arr = np.load(src_file)

                # Determine zoom factors
                if subfolder == 'image' or subfolder == 'depth' or subfolder == 'normal':
                    zoom_factors = [scale_factor, scale_factor, 1.0]  # (H, W, C)
                else:  # label
                    zoom_factors = [scale_factor, scale_factor]  # (H, W)

                # Use nearest for label, bilinear for the rest
                order = 0 if subfolder == 'label' else 1
                resized_arr = zoom(arr, zoom_factors, order=order)
                resized_arr = center_crop_to_multiple(resized_arr, multiple=32)

                np.save(dst_file, resized_arr.astype(arr.dtype))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize NYUv2 dataset")
    parser.add_argument('--src', type=str, required=True, help='Path to original NYUv2 dataset')
    parser.add_argument('--dst', type=str, required=True, help='Path to save resized dataset')
    parser.add_argument('--scale', type=float, default=0.5, help='Resize factor (e.g., 0.5)')

    args = parser.parse_args()

    resize_and_save(args.src, args.dst, args.scale)
