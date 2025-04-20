import torch
import torch.nn as nn
import torch.nn.functional as F
from segnet_mtan import SegNetMTAN
from segnet import SegNet
from utils import count_parameters

segnetMTAN = SegNetMTAN()
segnet = SegNet(task='semantic')
print('SegNetMTAN parameter count: ', count_parameters(segnetMTAN))
print('SegNet parameter count: ', count_parameters(segnet))