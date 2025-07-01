import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
from src.segnet_mtan import SegNetMTAN
from src.segnet import SegNet
from src.original_segnet_mtan import SegNet as OriginalSegNetMTAN
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
parser.add_argument('--task', default='all', type=str, help='task to train: all, semantic, depth, normal')
parser.add_argument('--single_task', default=False, action='store_true', help='train single task network')
parser.add_argument('--segnet', default=False, action='store_true', help='train SegNet instead of SegNetMTAN')
parser.add_argument('--lambda_consistency', default=0.0, type=float, help='weight for inter-task normal-depth consistency loss')
opt = parser.parse_args()


# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))
model1 = SegNetMTAN().to(device)
model2 = SegNet(opt.task).to(device)
model3 = OriginalSegNetMTAN().to(device)  # Use the original SegNetMTAN for comparison
#optimizer = optim.Adam(model1.parameters(), lr=1e-4)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

'''import sys
print("Model Structure: {}".format(SegNet_MTAN))
sys.exit(0)'''

print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(model1),
                                                         count_parameters(model1) / 24981069))
print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(model2),
                                                         count_parameters(model2) / 24981069))
print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(model3),
                                                         count_parameters(model3) / 24981069))