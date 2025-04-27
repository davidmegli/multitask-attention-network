# convert_npy_to_png_dataset.py
import os
import argparse
import numpy as np
from PIL import Image
from matplotlib import cm
from tqdm import tqdm
from utils import normalize_array, depth_to_colormap, normal_to_rgb, segmentation_to_color, image_from_npy, nyu_palette

def main(input_dir, output_dir, task, palette=None):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(os.listdir(input_dir))

    for fname in tqdm(files):
        if not fname.endswith(".npy"):
            continue

        path = os.path.join(input_dir, fname)
        arr = np.load(path)

        # Se l'array ha dimensioni inutili (es: (1, H, W)), le rimuovo
        while arr.ndim > 3:
            arr = arr.squeeze(0)
            
        if task == "depth":
            img = depth_to_colormap(arr.squeeze())
            img = Image.fromarray(img)
        elif task == "normal":
            img = normal_to_rgb(arr)
            img = Image.fromarray(img)
        elif task == "segm":
            img = segmentation_to_color(arr.squeeze(), palette)
        elif task == "image":
            img = image_from_npy(arr)
            img = Image.fromarray(img)
        else:
            raise ValueError("Unknown task type")

        out_path = os.path.join(output_dir, fname.replace(".npy", ".png"))
        img.save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", choices=["depth", "normal", "segm", "image"], required=True)
    args = parser.parse_args()

    nyu_palette = nyu_palette()

    main(args.input_dir, args.output_dir, args.task, palette=nyu_palette if args.task == "segm" else None)
