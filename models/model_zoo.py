"""
This file (model_zoo.py) is designed for:
    models
Copyright (c) 2024, Zhiyu Pan. All rights reserved.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import copy

from . import resnet
from .inception import *
from .units import *

class NOP(nn.Module):
    def forward(self, x):
        return x  

class DensePrintB(nn.Module):
    def __init__(self, num_in=1, ndim_feat=6, pos_embed=True, tar_shape = (256, 256)):
        super().__init__()
        self.num_in = num_in  # number of input channel
        self.ndim_feat = ndim_feat  # number of latent dimension

        self.tar_shape = tar_shape
        layers = [3, 4, 6, 3]
        self.base_width = 64
        num_layers = [64, 128, 256, 512]
        block = resnet.BasicBlock

        self.inplanes = num_layers[0]
        self.layer0 = nn.Sequential(
            nn.Conv2d(num_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.LeakyReLU(inplace=True),
        )
        self.layer1 = self._make_layers(block, num_layers[0], layers[0])
        self.layer2 = self._make_layers(block, num_layers[1], layers[1], stride=2)
        self.layer3 = self._make_layers(block, num_layers[2], layers[2], stride=2)
        self.layer4 = self._make_layers(block, num_layers[3], layers[3], stride=2)

        self.texture3 = copy.deepcopy(self.layer3)
        self.texture4 = copy.deepcopy(self.layer4)

        self.minu_map = nn.Sequential(
            DoubleConv(num_layers[2] * block.expansion, 128),
            DoubleConv(128, 128),
            DoubleConv(128, 128),
            BasicDeConv2d(128, 128, kernel_size=4, stride=2, padding=1),
            BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BasicDeConv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )  # size=(128, 128)

        self.embedding = nn.Sequential(
            PositionEncoding2D((self.tar_shape[0]//16, self.tar_shape[1]//16), num_layers[3] * block.expansion) if pos_embed else NOP(),
            nn.Conv2d(num_layers[3] * block.expansion, num_layers[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(num_layers[3]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_layers[3], ndim_feat, kernel_size=1),
        )
        self.embedding_t = nn.Sequential(
            PositionEncoding2D((self.tar_shape[0]//16, self.tar_shape[1]//16), num_layers[3] * block.expansion) if pos_embed else NOP(),
            nn.Conv2d(num_layers[3] * block.expansion, num_layers[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(num_layers[3]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_layers[3], ndim_feat, kernel_size=1),
        )


        self.foreground = nn.Sequential(
            nn.Conv2d(num_layers[3] * block.expansion, num_layers[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(num_layers[3]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_layers[3], 1, kernel_size=1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=stride, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                base_width=self.base_width,
                downsample=downsample,
                norm_layer=norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def get_embedding(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        t_x3 = self.texture3(x2)
        t_x4 = self.texture4(t_x3)
        feature_t = self.embedding_t(t_x4)
        feature_m = self.embedding(x4)
        foreground = self.foreground(t_x4)
        feature = torch.cat((feature_t, feature_m), dim=1)

        return {
            "feature": feature.flatten(1),
            "feature_t": feature_t.flatten(1),
            "feature_m": feature_m.flatten(1),
            "mask": foreground.flatten(1),
        }

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        t_x3 = self.texture3(x2)
        t_x4 = self.texture4(t_x3)
        feature_t = self.embedding_t(t_x4)

        minu_map = self.minu_map(x3)
        feature = self.embedding(x4)
        foreground = self.foreground(t_x4) 

        return {
            "input": x,
            "feat_f": feature,
            "feat_t": feature_t,
            "mask_f": foreground,
            "minu_map": minu_map,
            "minu_lst": torch.split(minu_map.detach(), 3, dim=1),
            "feat_lst": torch.split(feature.detach(), 3, dim=1),
        }