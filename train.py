import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
from src.segnet_mtan import SegNetMTAN
from src.utils import multi_task_trainer, count_parameters

from create_dataset import *
from utils import *

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--batch_size', default=2, type=int, help='batch size for training')
parser.add_argument('--epochs', default=20, type=int, help='number of epochs to train')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--downsample_ratio', default=1, type=float, help='downsample ratio for data augmentation')
parser.add_argument('--resume', default=False, action='store_true', help='resume training from the latest checkpoint')
opt = parser.parse_args()


# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))
SegNet_MTAN = SegNetMTAN().to(device)
optimizer = optim.Adam(SegNet_MTAN.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

'''import sys
print("Model Structure: {}".format(SegNet_MTAN))
sys.exit(0)'''

print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet_MTAN),
                                                         count_parameters(SegNet_MTAN) / 24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

# define dataset
dataset_path = opt.dataroot
downsample_ratio = opt.downsample_ratio
if not os.path.exists(dataset_path):
    raise Exception('Dataset path does not exist. Please check the path.')
if opt.apply_augmentation:
    nyuv2_train_set = NYUv2(root=dataset_path, train=True, augmentation=True, downsample_ratio=downsample_ratio)
    print('Applying data augmentation on NYUv2.')
else:
    nyuv2_train_set = NYUv2(root=dataset_path, train=True, downsample_ratio=downsample_ratio)
    print('Standard training strategy without data augmentation.')
# printing the images sizes (height, width)
print('Training data size: {}'.format(nyuv2_train_set[0][0].shape))
nyuv2_test_set = NYUv2(root=dataset_path, train=False, downsample_ratio=downsample_ratio)

batch_size = opt.batch_size
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False)

epochs = opt.epochs
# Train and evaluate multi-task network
multi_task_trainer(nyuv2_train_loader,
                   nyuv2_test_loader,
                   SegNet_MTAN,
                   device,
                   optimizer,
                   scheduler,
                   opt,
                   total_epoch=epochs,
                   resume=opt.resume)