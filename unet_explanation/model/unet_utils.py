import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConvBlock(nn.Module):
    " [conv -> BN -> ReLU] -> [conv -> BN -> ReLU]"
    def __init__(self, in_channels, out_channels, kernel_size = 3 , padding = True):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param padding:

            The Original paper uses VALID padding (i.e. no padding)
            The main benefit of using SAME padding is that the output feature map will have the same spatial dimensions
            as the input feature map.
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            # Usually Conv -> BatchNormalization -> Activation
            nn.Conv2d(in_channels , out_channels , kernel_size= kernel_size , padding = int(padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=int(padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self,inp):
        return self.double_conv(inp)


class Down(nn.Module):
    """
    Downscaling with maxpool and then Double Conv
        - 3x3 Conv2D -> BN -> ReLU
        - 3X3 Conv2D -> BN -> ReLU
        - MaxPooling
    """
    def __init__(self, in_channels , out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UNetConvBlock(in_channels,out_channels)
        )

    def forward(self,x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
        - Upsampling Convolution ("Up-convolution")
        - 3x3 Conv2D -> BN -> ReLU
        - 3X3 Conv2D -> BN -> ReLU

        Upsampling vs Transposed convolutions

        Transposed convolution (a.k.a."up-convolution or fractionally-strided convolutions or deconvolutions")
            - The original paper uses this
            - detects more fine-grained detail

        Other implementation use bilinear upsampling, possibly followed by a 1x1 convolution.
        The benefit of using upsampling is that it has no parameters and if you include the 1x1 convolution,
        it will still have less parameters than the transposed convolution
    """
    def __init__(self,in_channels , out_channels , bilinear = False):
        super(Up,self).__init__()

        if bilinear: # use the normal conv to reduce the number of channels
            self.up = nn.Upsample(scale_factor=2, mode= 'bilinear', align_corners = True)
        else: # use Transpose convolution (the one that official UNet used)
            self.up = nn.ConvTranspose2d(in_channels//2 , in_channels // 2, kernel_size = 2,stride=2 )

        self.conv = UNetConvBlock(in_channels,out_channels)

    def forward(self,x1,x2):
        # input dim is CHW
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1 , [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1] , dim = 1)
        out = self.conv(x)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)