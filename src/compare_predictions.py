# compare_predictions.py

import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

def concat_images(img1, img2):
    w, h = img1.size
    new_img = Image.new('RGB', (w * 2, h))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (w, 0))
    return new_img

def main(pred_dir, gt_dir, task, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".png") and f"_{task}" in f])

    for f in tqdm(pred_files):
        base = f.replace(f"_pred_{task}.png", "")
        pred_path = os.path.join(pred_dir, f)
        gt_path = os.path.join(gt_dir, base + ".png")

        if not os.path.exists(gt_path):
            base = str(int(base)) + ".png"
            gt_path = os.path.join(gt_dir, base)
            if not os.path.exists(gt_path):
                print(f"Missing GT for {f}")
                continue

        pred_img = Image.open(pred_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        comparison = concat_images(gt_img, pred_img)
        comparison.save(os.path.join(output_dir, f"compare_{task}_{base}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--task", required=True, choices=["seg", "depth", "normal"])
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    main(args.pred_dir, args.gt_dir, args.task, args.output_dir)
