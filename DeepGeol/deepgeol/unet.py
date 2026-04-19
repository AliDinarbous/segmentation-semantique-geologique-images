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

from .unet_parts import DoubleConv
from .unet_parts import Down2C
from .unet_parts import Up2C
from .unet_parts import OutConv



class UNet(torch.nn.Module):
    """
    Flexible implementation of a U-Net architecture

    :param input_channels: int number of input channels, default is 1
    :param hidden_channels: list(int) number of convolution channels across the layers, default is [64, 128, 256, 512, 1024]
    :param n_classes: int number of output channels, default is 1
    :param dropout: bool if True, a dropout layer is added after each DoubleConv block, default is False
    :param batch_norm: bool if True a BatchNorm layer is added after each convolution, 
        in this case the bias of the convolutional layer is turned to False
    :param bilinear: bool
    """
    def __init__(self, 
                 input_channels=1,
                 hidden_channels=[64, 128, 256, 512, 1024],
                 n_classes=1, 
                 dropout=False,
                 batch_norm=False,
                 bilinear=True):
        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.n_classes = n_classes
        self.dropout= dropout
        self.batch_norm = batch_norm
        self.bilinear = bilinear
        self.innput_conv = None

        # We use a DoubleConv block before going down
        self.inc = DoubleConv(input_channels, hidden_channels[0], dropout=dropout, batch_norm=batch_norm)

        # Add Down blocks
        self.down1 = Down2C(hidden_channels[0], hidden_channels[1])
        self.down2 = Down2C(hidden_channels[1], hidden_channels[2])
        self.down3 = Down2C(hidden_channels[2], hidden_channels[3])
        self.down4 = Down2C(hidden_channels[3], hidden_channels[4])

        # Add Up blocks
        self.up1 = Up2C(hidden_channels[4] + hidden_channels[3], hidden_channels[3], bilinear)
        self.up2 = Up2C(hidden_channels[3] + hidden_channels[2], hidden_channels[2], bilinear)
        self.up3 = Up2C(hidden_channels[2] + hidden_channels[1], hidden_channels[1], bilinear)
        self.up4 = Up2C(hidden_channels[1] + hidden_channels[0], hidden_channels[0], bilinear)

        self.outc = OutConv(hidden_channels[0], n_classes)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        """
        """
        down_outputs = []
        down_outputs.append(self.inc(x))

        # Go down
        x1 = self.inc(x)
        down_outputs.append(x1)
        x2 = self.down1(x1)
        down_outputs.append(x2)
        x3 = self.down2(x2)
        down_outputs.append(x3)
        x4 = self.down3(x3)
        down_outputs.append(x4)
        x5 = self.down4(x4)

        # Go up
        x = self.up1(x5, down_outputs[-1])  # x4
        x = self.up2(x, down_outputs[-2])   # x3
        x = self.up3(x, down_outputs[-3])   # x2
        x = self.up4(x, down_outputs[-4])   # x1
        
        logits = self.outc(x)
       
        return logits


