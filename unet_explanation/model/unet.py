import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet_utils import *

class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (Ronneberger et al., 2015)
    https://arxiv.org/abs/1505.04597

    Contracting Path
        - Two 3x3 Conv2D (Unpadded Conv, i.e. no padding)
        - followed by a ReLU
        - A 2x2 MaxPooling (with stride 2)
    Expansive Path : sequence of "up-convolutions" and "concatenation" with high-resolution feature from contracting path
        - "2x2 up-convolution" that halves the number of feature channels
        - A "concatenation" with the correspondingly cropped feature map from the contracting path
        - Two 3x3 Conv2D
        - Followed by a ReLU

    Final Layer
        - "1x1 Conv2D" is used to map each 64 component feature vector to
        the desired number of classes
    """
    def __init__(self, n_channels, n_classes , bilinear = False):
        super(UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.in_conv = UNetConvBlock(self.n_channels , 64)
        self.Down1 = Down(64 , 128)
        self.Down2 = Down(128, 256)
        self.Down3 = Down(256, 512)
        self.Down4 = Down(512, 512)
        self.Up1 = Up(512 + 512, 256 , self.bilinear)
        self.Up2 = Up(256 + 256, 128 , self.bilinear)
        self.Up3 = Up(128 + 128 , 64 , self.bilinear)
        self.Up4 = Up(64 + 64, 64 , self.bilinear)
        self.out_conv = OutConv(64, n_classes)

    def forward(self,x):
        x1 = self.in_conv(x)
        x2 = self.Down1(x1)
        x3 = self.Down2(x2)
        x4 = self.Down3(x3)
        x5 = self.Down4(x4)
        x = self.Up1(x5,x4)
        x = self.Up2(x ,x3)
        x = self.Up3(x ,x2)
        x = self.Up4(x ,x1)
        out = self.out_conv(x)
        return out


if __name__ == '__main__':
    UNet(3,10)

