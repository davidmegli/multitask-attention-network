import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def depth_to_normals_single(depth):
    """
    depth: (1, H, W) tensor (no batch)
    returns: (3, H, W)
    """
    dzdx = depth[:, :, :-1] - depth[:, :, 1:]
    dzdy = depth[:, :-1, :] - depth[:, 1:, :]

    dzdx = F.pad(dzdx, (0, 1, 0, 0), mode='replicate')
    dzdy = F.pad(dzdy, (0, 0, 0, 1), mode='replicate')

    normal_x = -dzdx
    normal_y = -dzdy
    normal_z = torch.ones_like(dzdx)

    normal = torch.cat([normal_x, normal_y, normal_z], dim=0)  # (3, H, W)
    normal = F.normalize(normal, dim=0)
    return normal

def main(input_dir, output_dir, use_pt=False):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    for fname in tqdm(files):
        path = os.path.join(input_dir, fname)
        depth_np = np.load(path)
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np.squeeze(-1)  # Rimuove il canale extra

        depth_tensor = torch.from_numpy(depth_np).float().unsqueeze(0)  # (1, H, W)
        normal_tensor = depth_to_normals_single(depth_tensor)  # (3, H, W)
        normal_np = normal_tensor.numpy()

        out_path = os.path.join(output_dir, fname.replace(".npy", ".pt" if use_pt else ".npy"))
        if use_pt:
            torch.save(normal_tensor, out_path)
        else:
            np.save(out_path, normal_np)

    print("Done. Saved normals in:", output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with .npy depth files")
    parser.add_argument("--output_dir", required=True, help="Directory to save normals")
    parser.add_argument("--use_pt", action="store_true", help="Save normals as .pt instead of .npy")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.use_pt)