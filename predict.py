import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from src.segnet_mtan import SegNetMTAN
from create_dataset import NYUv2
from utils import count_parameters  # solo se vuoi stampare il numero di parametri


# === Utility per convertire tensori in immagini salvabili ===
def to_numpy_img(tensor, mode='rgb'):
    tensor = tensor.detach().cpu()
    if mode == 'rgb':
        img = (tensor * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    elif mode == 'seg':
        if tensor.dim() == 2:
            img = tensor.byte().numpy()
        else:
            img = torch.argmax(tensor, dim=0).byte().numpy()
        img = (img * (255 // img.max())).astype(np.uint8)
        img = np.stack([img] * 3, axis=-1)
        return img
    elif mode == 'depth':
        tensor = tensor.squeeze()
        img = tensor.numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        img = np.stack([img] * 3, axis=-1)
    elif mode == 'normal':
        img = tensor.permute(1, 2, 0).numpy()
        img = ((img + 1) / 2 * 255).astype(np.uint8)
    return img


# === Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Carica modello e checkpoint ===
model = SegNetMTAN().to(device)
checkpoint_path = "checkpoint_latest.pth"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Modello caricato da checkpoint.")
else:
    raise FileNotFoundError("Checkpoint non trovato!")

model.eval()


# === Dataset e DataLoader ===
dataset_path = r"path/to/nyuv2"  # Cambia con il percorso corretto del dataset
downsample_ratio = 1
test_set = NYUv2(root=dataset_path, train=False, downsample_ratio=downsample_ratio)
test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

print(f"Test set size: {len(test_set)}")


# === Cartella per salvare le predizioni ===
SAVE_DIR = "predictions"
os.makedirs(SAVE_DIR, exist_ok=True)


# === Loop di predizione e salvataggio ===
NUM_PREDICTIONS = 10
saved = 0

with torch.no_grad():
    for idx, (img, seg_gt, depth_gt, normal_gt) in enumerate(test_loader):
        if saved >= NUM_PREDICTIONS:
            break

        img = img.to(device)
        seg_gt = seg_gt.to(device)
        depth_gt = depth_gt.to(device)
        normal_gt = normal_gt.to(device)

        preds, _ = model(img)
        seg_pred, depth_pred, normal_pred = preds[0], preds[1], preds[2]

        batch_size = img.size(0)
        for i in range(batch_size):
            if saved >= NUM_PREDICTIONS:
                break

            base_id = saved

            # === INPUT ===
            Image.fromarray(to_numpy_img(img[i], mode='rgb')).save(f"{SAVE_DIR}/{base_id:03d}_rgb.png")

            # === PREDICTIONS ===
            Image.fromarray(to_numpy_img(seg_pred[i], mode='seg')).save(f"{SAVE_DIR}/{base_id:03d}_pred_seg.png")
            Image.fromarray(to_numpy_img(depth_pred[i], mode='depth')).save(f"{SAVE_DIR}/{base_id:03d}_pred_depth.png")
            Image.fromarray(to_numpy_img(normal_pred[i], mode='normal')).save(f"{SAVE_DIR}/{base_id:03d}_pred_normal.png")

            # === GROUND TRUTH ===
            Image.fromarray(to_numpy_img(seg_gt[i], mode='seg')).save(f"{SAVE_DIR}/{base_id:03d}_gt_seg.png")
            Image.fromarray(to_numpy_img(depth_gt[i], mode='depth')).save(f"{SAVE_DIR}/{base_id:03d}_gt_depth.png")
            Image.fromarray(to_numpy_img(normal_gt[i], mode='normal')).save(f"{SAVE_DIR}/{base_id:03d}_gt_normal.png")

            saved += 1

print(f"\n✅ Predizioni salvate in '{SAVE_DIR}/' ({saved} immagini)")
