import os
from torchviz import make_dot
import torch
from src.segnet_mtan import SegNetMTAN
from src.segnet import SegNet
from src.original_segnet_mtan import SegNet as OriginalSegNetMTAN
from create_dataset import *
from utils import *

# Se disponibile, usa la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_path = "C:\\Users\\verba\\Desktop\\UNIFI_AI\\ANNO 1\\B031278 (B241) - MODULO DEEP LEARNING\\Elaborato David\\NYUv2 preprocessed"
downsample_ratio = 1
apply_augmentation = True  # Set to True if you want to apply data augmentation
if not os.path.exists(dataset_path):
    raise Exception('Dataset path does not exist. Please check the path.')
if apply_augmentation:
    nyuv2_train_set = NYUv2(root=dataset_path, train=True, augmentation=True, downsample_ratio=downsample_ratio)
    print('Applying data augmentation on NYUv2.')
else:
    nyuv2_train_set = NYUv2(root=dataset_path, train=True, downsample_ratio=downsample_ratio)
    print('Standard training strategy without data augmentation.')
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=2,
    shuffle=True)
train_data, train_label, train_depth, train_normal = next(iter(nyuv2_train_loader))

# Lista di tuple (nome_modello, classe_modello)
models = [
    ("SegNetMTAN", SegNetMTAN),
    ("SegNet", SegNet),
    ("OriginalSegNetMTAN", OriginalSegNetMTAN)
]

for name, ModelClass in models:
    # Istanzia e sposta su GPU
    model = ModelClass().to(device)
    train_data = train_data.to(device)
    model.train()
    train_pred, logsigma = model(train_data) # Give dummy batch to forward().
    
    make_dot(train_pred[0], params=dict(list(model.named_parameters()))).render(f"{name}_torchviz", format="png")
