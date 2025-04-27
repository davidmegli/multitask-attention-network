import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
from glob import glob
from segnet_mtan import SegNetMTAN
import numpy as np

def normalize_tensor(tensor):
    tensor = tensor.detach().cpu()
    tensor -= tensor.min()
    tensor /= tensor.max() + 1e-5
    return tensor

def save_attention_images(input_image, enc_att, dec_att, task_name, img_name_base, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images_to_plot = []

    # Save input image
    img = normalize_tensor(input_image.squeeze(0))
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    images_to_plot.append(img)

    count = 1
    for layer in [1, 2, 3, 4, 5]:
        for source, name in zip([enc_att, dec_att], ["enc", "dec"]):
            att_map = source[layer][0][0]  # task-specific [0], first channel [0]
            att_img = normalize_tensor(att_map)

            # Assicura che sia (1, H, W)
            if att_img.dim() == 2:  # (H, W)
                att_img = att_img.unsqueeze(0)
            elif att_img.dim() == 3 and att_img.shape[0] != 1:
                att_img = att_img[0].unsqueeze(0)

            # Ripeti sui 3 canali → (3, H, W)
            att_img = att_img.repeat(3, 1, 1)
            #print(att_img.shape)


            # Resize attention map to match input image size
            att_img_resized = torch.nn.functional.interpolate(
                att_img.unsqueeze(0), size=input_image.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(0)

            images_to_plot.append(att_img_resized)


            # Save single attention map
            filename = f"{img_name_base}_{task_name}_map_{count}.png"
            vutils.save_image(att_img, os.path.join(output_dir, filename))
            count += 1

    # Collage
    grid = vutils.make_grid(images_to_plot, nrow=6, padding=2, normalize=False)
    np_img = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 6))
    plt.axis('off')
    plt.imshow(np_img)
    plt.title(f"{img_name_base} - {task_name}")
    plt.savefig(os.path.join(output_dir, f"{img_name_base}_{task_name}_maps.png"))
    plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegNetMTAN()
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image_paths = sorted(glob(os.path.join(args.input_dir, "*")))[:args.num_images]
    task_names = ["segmentation", "depth", "normal"]

    for img_path in image_paths:
        image_array = np.load(img_path)  # Carica array (H, W, C) o (C, H, W)
        if image_array.ndim == 3 and image_array.shape[0] in [1, 3]:  # (C, H, W)
            pass
        elif image_array.ndim == 3 and image_array.shape[2] in [1, 3]:  # (H, W, C)
            image_array = np.transpose(image_array, (2, 0, 1))
        else:
            raise ValueError(f"Formato non supportato per: {img_path}")

        image_tensor = torch.tensor(image_array, dtype=torch.float32) / 255.0
        input_tensor = transforms.Resize((224, 224))(image_tensor).unsqueeze(0).to(device)

        # Forward pass with attention output
        with torch.no_grad():
            outputs, logsigma, enc_att, dec_att = model(input_tensor, return_attentions=True)

        img_name_base = os.path.splitext(os.path.basename(img_path))[0]

        for task_id, task_name in enumerate(task_names):
            save_attention_images(
                input_tensor,
                {k: [enc_att[k][task_id]] for k in enc_att},
                {k: [dec_att[k][task_id]] for k in dec_att},
                task_name,
                img_name_base,
                args.output_dir
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to folder containing input images")
    parser.add_argument("--num_images", type=int, required=False, default=1, help="Number of images to process")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save output visualizations")
    args = parser.parse_args()

    main(args)
