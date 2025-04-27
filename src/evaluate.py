import os
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from segnet_mtan import SegNetMTAN
from create_dataset import NYUv2
from utils import nyu_palette, segmentation_to_color, depth_to_colormap, normal_to_rgb


def to_numpy_img(tensor, mode='rgb', palette=None):
    try:
        tensor = tensor.detach().cpu()
        if mode == 'rgb':
            if tensor.dim() == 3:
                img = (tensor * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                return img
            else:
                print("⚠️ RGB tensor non ha 3 dimensioni:", tensor.shape)
        elif mode == 'seg':
            if tensor.dim() == 2:
                seg = tensor.byte().numpy()
            else:
                seg = torch.argmax(tensor, dim=0).byte().numpy()
            return segmentation_to_color(seg, palette=palette)
        elif mode == 'depth':
            return depth_to_colormap(tensor.squeeze().numpy())
        elif mode == 'normal':
            return normal_to_rgb(tensor.numpy())
    except Exception as e:
        print(f"❌ Errore in to_numpy_img (mode={mode}): {e}")
        return None



def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"📌 Using device: {device}")

    model = SegNetMTAN().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Modello caricato da {args.model_path}")
    model.eval()

    test_set = NYUv2(root=args.input_dir, train=False)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)
    print(f"📦 Test set size: {len(test_set)}")

    os.makedirs(args.output_dir, exist_ok=True)

    palette = nyu_palette()
    saved = 0

    with torch.no_grad():
        for img, seg_gt, depth_gt, normal_gt in tqdm(test_loader, desc="📤 Predicting"):
            img = img.to(device)
            preds, _ = model(img)
            seg_pred, depth_pred, normal_pred = preds[0], preds[1], preds[2]

            batch_size = img.size(0)
            for i in range(batch_size):
                base_id = saved

                Image.fromarray(to_numpy_img(img[i], mode='rgb')).save(os.path.join(args.output_dir, f"{base_id:03d}_rgb.png"))
                to_numpy_img(seg_pred[i], mode='seg', palette=palette).save(os.path.join(args.output_dir, f"{base_id:03d}_pred_seg.png"))
                Image.fromarray(to_numpy_img(depth_pred[i], mode='depth')).save(os.path.join(args.output_dir, f"{base_id:03d}_pred_depth.png"))
                Image.fromarray(to_numpy_img(normal_pred[i], mode='normal')).save(os.path.join(args.output_dir, f"{base_id:03d}_pred_normal.png"))

                saved += 1

    print(f"\n✅ Inference completata su tutto il set. Predizioni salvate in '{args.output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path al checkpoint del modello (.pth)')
    parser.add_argument('--input_dir', type=str, required=True, help='Cartella root del dataset NYUv2')
    parser.add_argument('--output_dir', type=str, required=True, help='Cartella dove salvare le predizioni')

    args = parser.parse_args()
    main(args)
