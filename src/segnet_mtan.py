'''
File Name:  segnet_mtan.py
Author:     David Megli
Created:    2025-03-22
Description: This file contains the MTAN implementation based on SegNet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SegNetMTAN(nn.Module):
    def __init__(self, pretrained=True):
        super(SegNetMTAN, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg16.features.children())
        self.class_nb = 13 # NYUv2 13 classes

        # VGG16_BN architecture:
        """ 
        VGG(
        (features): Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU(inplace=True)
            (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (9): ReLU(inplace=True)
            (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (12): ReLU(inplace=True)
            (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (16): ReLU(inplace=True)
            (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (19): ReLU(inplace=True)
            (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (22): ReLU(inplace=True)
            (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (26): ReLU(inplace=True)
            (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (29): ReLU(inplace=True)
            (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (32): ReLU(inplace=True)
            (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (36): ReLU(inplace=True)
            (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (39): ReLU(inplace=True)
            (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (42): ReLU(inplace=True)
            (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
        (classifier): Sequential(
            (0): Linear(in_features=25088, out_features=4096, bias=True)
            (1): ReLU(inplace=True)
            (2): Dropout(p=0.5, inplace=False)
            (3): Linear(in_features=4096, out_features=4096, bias=True)
            (4): ReLU(inplace=True)
            (5): Dropout(p=0.5, inplace=False)
            (6): Linear(in_features=4096, out_features=1000, bias=True)
        )
        ) """

        # DECODER
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.decoderBlock5 = self.decoderConvBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1, n_conv=3)
        self.decoderBlock4 = self.decoderConvBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1, n_conv=3)
        self.decoderBlock3 = self.decoderConvBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1, n_conv=3)
        self.decoderBlock2 = self.decoderConvBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1, n_conv=2)
        self.decoderBlock1 = self.decoderConvBlock(in_channels=64, out_channels=64, kernel_size=3, padding=1, n_conv=2)
        #self.softmax = nn.Softmax(dim=1)

        # ATTENTION MODULES
        n_tasks = 3
        self.encoderAttentionModule1 = nn.ModuleList()
        self.encoderAttentionModule2 = nn.ModuleList()
        self.encoderAttentionModule3 = nn.ModuleList()
        self.encoderAttentionModule4 = nn.ModuleList()
        self.encoderAttentionModule5 = nn.ModuleList()
        self.encoderAttentionConv1 = nn.ModuleList()
        self.encoderAttentionConv2 = nn.ModuleList()
        self.encoderAttentionConv3 = nn.ModuleList()
        self.encoderAttentionConv4 = nn.ModuleList()
        self.encoderAttentionConv5 = nn.ModuleList()
        for task in range(n_tasks):
            self.encoderAttentionModule1.append(self.attentionModule(64, 64, 64))
            self.encoderAttentionConv1.append(self.attentionConv(64, 128))
            self.encoderAttentionModule2.append(self.attentionModule(256, 128, 128))
            self.encoderAttentionConv2.append(self.attentionConv(128, 256))
            self.encoderAttentionModule3.append(self.attentionModule(512, 256, 256))
            self.encoderAttentionConv3.append(self.attentionConv(256, 512))
            self.encoderAttentionModule4.append(self.attentionModule(1024, 512, 512))
            self.encoderAttentionConv4.append(self.attentionConv(512, 512))
            self.encoderAttentionModule5.append(self.attentionModule(1024, 512, 512))
            self.encoderAttentionConv5.append(self.attentionConv(512, 512))
        self.decoderAttentionModule1 = nn.ModuleList()
        self.decoderAttentionModule2 = nn.ModuleList()
        self.decoderAttentionModule3 = nn.ModuleList()
        self.decoderAttentionModule4 = nn.ModuleList()
        self.decoderAttentionModule5 = nn.ModuleList()
        self.decoderAttentionConv1 = nn.ModuleList()
        self.decoderAttentionConv2 = nn.ModuleList()
        self.decoderAttentionConv3 = nn.ModuleList()
        self.decoderAttentionConv4 = nn.ModuleList()
        self.decoderAttentionConv5 = nn.ModuleList()
        for task in range(n_tasks):
            self.decoderAttentionModule1.append(self.attentionModule(128, 64, 64))
            self.decoderAttentionConv1.append(self.attentionConv(64, 64))
            self.decoderAttentionModule2.append(self.attentionModule(192, 64, 64))
            self.decoderAttentionConv2.append(self.attentionConv(128, 64))
            self.decoderAttentionModule3.append(self.attentionModule(384, 128, 128))
            self.decoderAttentionConv3.append(self.attentionConv(256, 128))
            self.decoderAttentionModule4.append(self.attentionModule(768, 256, 256))
            self.decoderAttentionConv4.append(self.attentionConv(512, 256))
            self.decoderAttentionModule5.append(self.attentionModule(1024, 512, 512))
            self.decoderAttentionConv5.append(self.attentionConv(512, 512))

        self.prediction_task1 = self.predictionConv(64, 13) # NYUv2 13 classes
        self.prediction_task2 = self.predictionConv(64,1) # Depth
        self.prediction_task3 = self.predictionConv(64,3) # Normal

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))
        if pretrained:
            self.init_weights()
        # ENCODER
        self.encoderBlock1A = nn.Sequential(*features[0:3])  # conv1
        self.encoderBlock1B = nn.Sequential(*features[3:6])  # conv1
        self.encoderBlock2A = nn.Sequential(*features[7:10])  # conv2
        self.encoderBlock2B = nn.Sequential(*features[10:13])  # conv2
        self.encoderBlock3A = nn.Sequential(*features[14:17])  # conv3
        self.encoderBlock3B = nn.Sequential(*features[17:23])  # conv3
        self.encoderBlock4A = nn.Sequential(*features[24:27])  # conv4
        self.encoderBlock4B = nn.Sequential(*features[27:33])  # conv4
        self.encoderBlock5A = nn.Sequential(*features[34:37])  # conv5
        self.encoderBlock5B = nn.Sequential(*features[37:43])  # conv5
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False)
        if not pretrained:
            self.init_weights()
        # TODO: use pretrained weights for the encoder

    def init_weights(self):
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

    def forward(self, x):
        # ENCODER
        enc_1_u = self.encoderBlock1A(x) # encoder block 1, conv layer 1
        enc_1_p = self.encoderBlock1B(enc_1_u) # encoder block 1, conv layer 2
        x1, idx1 = self.pool(enc_1_p) # encoder block 1 pooling, saving indices for unpooling

        enc_2_u = self.encoderBlock2A(x1)
        enc_2_p = self.encoderBlock2B(enc_2_u)
        x2, idx2 = self.pool(enc_2_p)

        enc_3_u = self.encoderBlock3A(x2)
        enc_3_p = self.encoderBlock3B(enc_3_u)
        x3, idx3 = self.pool(enc_3_p)

        enc_4_u = self.encoderBlock4A(x3)
        enc_4_p = self.encoderBlock4B(enc_4_u)
        x4, idx4 = self.pool(enc_4_p)

        enc_5_u = self.encoderBlock5A(x4)
        enc_5_p = self.encoderBlock5B(enc_5_u)
        x5, idx5 = self.pool(enc_5_p)

        # DECODER
        dec_5_u = self.unpool(x5, idx5, output_size=enc_5_p.size()) # decoder block 5 (1st), unpooling using indices from encoder block 5
        dec_5_p = self.decoderBlock5(dec_5_u) # decoder block 5, conv layers

        dec_4_u = self.unpool(dec_5_p, idx4, output_size=enc_4_p.size())
        dec_4_p = self.decoderBlock4(dec_4_u)

        dec_3_u = self.unpool(dec_4_p, idx3, output_size=enc_3_p.size())
        dec_3_p = self.decoderBlock3(dec_3_u)

        dec_2_u = self.unpool(dec_3_p, idx2, output_size=enc_2_p.size())
        dec_2_p = self.decoderBlock2(dec_2_u)

        dec_1_u = self.unpool(dec_2_p, idx1, output_size=enc_1_p.size())
        dec_1_p = self.decoderBlock1(dec_1_u)

        # ATTENTION MODULES
        enc_att = {i: [] for i in range(1, 6)} # encoder attention modules
        dec_att = {i: [] for i in range(1, 6)} # decoder attention modules
        for task in range(3):
            # Attention modules for encoder
            # conv -> bn -> ReLU -> conv -> bn -> sigmoid
            ea1 = self.encoderAttentionModule1[task](enc_1_u)
            # element-wise multiplication with encoder feature map
            ea1 = (ea1 * enc_1_p)
             # conv -> bn -> ReLU
            ea1 = self.encoderAttentionConv1[task](ea1)
            # max pooling
            ea1 = F.max_pool2d(ea1, kernel_size=2, stride=2)
            enc_att[1].append(ea1)

            ea2 = self.encoderAttentionModule2[task](torch.cat([enc_2_u, enc_att[1][task]], dim=1))
            ea2 = (ea2 * enc_2_p)
            ea2 = self.encoderAttentionConv2[task](ea2)
            ea2 = F.max_pool2d(ea2, kernel_size=2, stride=2)
            enc_att[2].append(ea2)

            ea3 = self.encoderAttentionModule3[task](torch.cat([enc_3_u, enc_att[2][task]], dim=1)) # merge
            ea3 = (ea3 * enc_3_p)
            ea3 = self.encoderAttentionConv3[task](ea3)
            ea3 = F.max_pool2d(ea3, kernel_size=2, stride=2)
            enc_att[3].append(ea3)

            ea4 = self.encoderAttentionModule4[task](torch.cat([enc_4_u, enc_att[3][task]], dim=1))
            ea4 = (ea4 * enc_4_p)
            ea4 = self.encoderAttentionConv4[task](ea4)
            ea4 = F.max_pool2d(ea4, kernel_size=2, stride=2)
            enc_att[4].append(ea4)

            ea5 = self.encoderAttentionModule5[task](torch.cat([enc_5_u, enc_att[4][task]], dim=1))
            ea5 = (ea5 * enc_5_p)
            ea5 = self.encoderAttentionConv5[task](ea5)
            ea5 = F.max_pool2d(ea5, kernel_size=2, stride=2)
            enc_att[5].append(ea5)

            # Attention modules for decoder
            # upsampling
            da5 = F.interpolate(enc_att[5][task], scale_factor=2, mode='bilinear', align_corners=True)
            # conv -> bn -> ReLU
            da5 = self.decoderAttentionConv5[task](da5)
            # merge -> conv -> bn -> ReLU -> conv -> bn -> sigmoid
            da5 = self.decoderAttentionModule5[task](torch.cat([dec_5_u, da5], dim=1))
            # element-wise multiplication with decoder feature map
            da5 = (da5 * dec_5_p)
            dec_att[5].append(da5)

            da4 = F.interpolate(dec_att[5][task], scale_factor=2, mode='bilinear', align_corners=True)
            da4 = self.decoderAttentionConv4[task](da4)
            da4 = self.decoderAttentionModule4[task](torch.cat([dec_4_u, da4], dim=1))
            da4 = (da4 * dec_4_p)
            dec_att[4].append(da4)

            da3 = F.interpolate(dec_att[4][task], scale_factor=2, mode='bilinear', align_corners=True)
            da3 = self.decoderAttentionConv3[task](da3)
            da3 = self.decoderAttentionModule3[task](torch.cat([dec_3_u, da3], dim=1))
            da3 = (da3 * dec_3_p)
            dec_att[3].append(da3)

            da2 = F.interpolate(dec_att[3][task], scale_factor=2, mode='bilinear', align_corners=True)
            da2 = self.decoderAttentionConv2[task](da2)
            da2 = self.decoderAttentionModule2[task](torch.cat([dec_2_u, da2], dim=1))
            da2 = (da2 * dec_2_p)
            dec_att[2].append(da2)

            da1 = F.interpolate(dec_att[2][task], scale_factor=2, mode='bilinear', align_corners=True)
            da1 = self.decoderAttentionConv1[task](da1)
            da1 = self.decoderAttentionModule1[task](torch.cat([dec_1_u, da1], dim=1))
            da1 = (da1 * dec_1_p)
            dec_att[1].append(da1)

        # Task prediction layers: 1: semantic segmentation, 2: depth estimation, 3: normal estimation
        task1 = F.log_softmax(self.prediction_task1(dec_att[1][0]), dim=1)
        task2 = self.prediction_task2(dec_att[1][1])
        task3 = self.prediction_task3(dec_att[1][2])
        task3 = task3 / torch.norm(task3, p=2, dim=1, keepdim=True)  # Normalization

        return [task1, task2, task3], self.logsigma

    def decoderConvBlock(self, in_channels, out_channels, kernel_size=3, padding=1, n_conv=2):
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding))
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, n_conv - 1):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
        
    
    def attentionModule(self, in_channels, int_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(int_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int_channels, out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
    
    def attentionConv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def predictionConv(self, in_channels=64, out_channels=3):
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0),
            )