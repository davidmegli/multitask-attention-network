import os
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from segnet import SegNet
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

    # === Inizializzazione modello ===
    task = 'semantic' if args.task == 'seg' else args.task
    model = SegNet(task=task).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"✅ Modello caricato da {args.model_path}")

    model.eval()

    # === Dataset ===
    test_set = NYUv2(root=args.input_dir, train=False)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)
    print(f"📦 Test set size: {len(test_set)}")

    os.makedirs(args.output_dir, exist_ok=True)

    palette = nyu_palette()
    saved = 0

    with torch.no_grad():
        for img, _, _, _ in tqdm(test_loader, desc="📤 Predicting"):
            img = img.to(device)
            pred = model(img)

            batch_size = img.size(0)
            for i in range(batch_size):
                base_id = saved

                Image.fromarray(to_numpy_img(img[i], mode='rgb')).save(
                    os.path.join(args.output_dir, f"{base_id:03d}_rgb.png")
                )

                if args.task == 'seg':
                    to_numpy_img(pred[i], mode='seg', palette=palette).save(
                        os.path.join(args.output_dir, f"{base_id:03d}_pred_seg.png")
                    )
                elif args.task == 'depth':
                    Image.fromarray(to_numpy_img(pred[i], mode='depth')).save(
                        os.path.join(args.output_dir, f"{base_id:03d}_pred_depth.png")
                    )
                elif args.task == 'normal':
                    Image.fromarray(to_numpy_img(pred[i], mode='normal')).save(
                        os.path.join(args.output_dir, f"{base_id:03d}_pred_normal.png")
                    )
                else:
                    raise ValueError(f"Task '{args.task}' non supportato. Usa 'seg', 'depth' o 'normal'.")

                saved += 1

    print(f"\n✅ Inference completata su tutto il set. Predizioni salvate in '{args.output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path al checkpoint del modello (.pth)')
    parser.add_argument('--input_dir', type=str, required=True, help='Cartella root del dataset NYUv2')
    parser.add_argument('--output_dir', type=str, required=True, help='Cartella dove salvare le predizioni')
    parser.add_argument('--task', type=str, choices=['seg', 'depth', 'normal'], required=True, help='Task da eseguire (seg, depth, normal)')

    args = parser.parse_args()
    main(args)
