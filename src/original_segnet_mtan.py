import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler

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
opt = parser.parse_args()


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512] # number of filters in each layer
        self.class_nb = 13 # number of classes in semantic segmentation

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])]) # input channels = 3 (RGB)
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(3):
            if j < 2:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, pred=False): # pred=False for encoder/decoder layers, pred means prediction layer
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for depth in range(5):
            g_encoder[depth], g_decoder[-depth - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for task in range(3):
            atten_encoder[task], atten_decoder[task] = ([0] * 5 for _ in range(2))
        for task in range(3):
            for depth in range(5):
                atten_encoder[task][depth], atten_decoder[task][depth] = ([0] * 3 for _ in range(2))

        # define global shared network
        for depth in range(5):
            if depth == 0:
                g_encoder[depth][0] = self.encoder_block[depth](x)
                g_encoder[depth][1] = self.conv_block_enc[depth](g_encoder[depth][0])
                g_maxpool[depth], indices[depth] = self.down_sampling(g_encoder[depth][1])
            else:
                g_encoder[depth][0] = self.encoder_block[depth](g_maxpool[depth - 1])
                g_encoder[depth][1] = self.conv_block_enc[depth](g_encoder[depth][0])
                g_maxpool[depth], indices[depth] = self.down_sampling(g_encoder[depth][1])

        for depth in range(5):
            if depth == 0:
                g_upsampl[depth] = self.up_sampling(g_maxpool[-1], indices[-depth - 1])
                g_decoder[depth][0] = self.decoder_block[-depth - 1](g_upsampl[depth])
                g_decoder[depth][1] = self.conv_block_dec[-depth - 1](g_decoder[depth][0])
            else:
                g_upsampl[depth] = self.up_sampling(g_decoder[depth - 1][-1], indices[-depth - 1])
                g_decoder[depth][0] = self.decoder_block[-depth - 1](g_upsampl[depth])
                g_decoder[depth][1] = self.conv_block_dec[-depth - 1](g_decoder[depth][0])

        # define task dependent attention module
        for task in range(3):
            for depth in range(5):
                if depth == 0:
                    atten_encoder[task][depth][0] = self.encoder_att[task][depth](g_encoder[depth][0])
                    atten_encoder[task][depth][1] = (atten_encoder[task][depth][0]) * g_encoder[depth][1]
                    atten_encoder[task][depth][2] = self.encoder_block_att[depth](atten_encoder[task][depth][1])
                    atten_encoder[task][depth][2] = F.max_pool2d(atten_encoder[task][depth][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[task][depth][0] = self.encoder_att[task][depth](torch.cat((g_encoder[depth][0], atten_encoder[task][depth - 1][2]), dim=1))
                    atten_encoder[task][depth][1] = (atten_encoder[task][depth][0]) * g_encoder[depth][1]
                    atten_encoder[task][depth][2] = self.encoder_block_att[depth](atten_encoder[task][depth][1])
                    atten_encoder[task][depth][2] = F.max_pool2d(atten_encoder[task][depth][2], kernel_size=2, stride=2)

            for depth in range(5):
                if depth == 0:
                    atten_decoder[task][depth][0] = F.interpolate(atten_encoder[task][-1][-1], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[task][depth][0] = self.decoder_block_att[-depth - 1](atten_decoder[task][depth][0])
                    atten_decoder[task][depth][1] = self.decoder_att[task][-depth - 1](torch.cat((g_upsampl[depth], atten_decoder[task][depth][0]), dim=1))
                    atten_decoder[task][depth][2] = (atten_decoder[task][depth][1]) * g_decoder[depth][-1]
                else:
                    atten_decoder[task][depth][0] = F.interpolate(atten_decoder[task][depth - 1][2], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[task][depth][0] = self.decoder_block_att[-depth - 1](atten_decoder[task][depth][0])
                    atten_decoder[task][depth][1] = self.decoder_att[task][-depth - 1](torch.cat((g_upsampl[depth], atten_decoder[task][depth][0]), dim=1))
                    atten_decoder[task][depth][2] = (atten_decoder[task][depth][1]) * g_decoder[depth][-1]

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma

