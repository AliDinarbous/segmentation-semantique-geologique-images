# coding: utf-8 -*-
#
# This file is part of DeepGeol.
#
# DeepGeol is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# DeepGeol is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DeepGeol.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2024-2025 Anthony Larcher
"""


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2024-2025 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = "reS"


import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=None, batch_norm=False):
        """
        (convolution => [BN] => ReLU) * 2 + DropOut [optional]
    
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param mid_channels: number of middle channels, if None, mid-channels = out_channels, default is None
        :type mid_channels: int
        :param dropout: value of the DropOut probability, if None, Drop-Out is not applied, default is None
        :type dropout: float
        :param batch_norm: apply BatchNorm2d after each convolution layer, default is False
        :type batch_norm: bool
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        layers = []

        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down2C(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=False, batch_norm=False):
        """
        Downscaling with maxpool then double conv block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param mid_channels: number of middle channels, if None, mid-channels = out_channels, default is None
        :type mid_channels: int
        :param dropout: value of the DropOut probability, if None, Drop-Out is not applied, default is None
        :type dropout: float
        :param batch_norm: apply BatchNorm2d after each convolution layer, default is False
        :type batch_norm: bool
        """
        super().__init__()
        ...
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, mid_channels, dropout, batch_norm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2C(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        If bilinear is True:
            Upsample the first input by a factor 2 with bilinear mode
            Concatenate in the Channel dimension with the second input 
            (beware: make sure dimensions are equal apply padding if not
            Then apply a DoubleConv block on the result
        If bilinear is False:
            Upsample the first input using a ConvTranspose2d 
            that reduces the number of channels by a factor 2
            

        :param in_channels: number of input channels (sum of channels from the two inputs
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param bilinear: it True ,default is True
        :type bilinear: bool
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv =  DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate the two inputs (x1 has already been processed)
        x = torch.cat([x2, x1], dim=1)

        # Apply a 2D convolution
        return self.conv(x)
        

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Apply a last 2D convolution to produce the final output

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


