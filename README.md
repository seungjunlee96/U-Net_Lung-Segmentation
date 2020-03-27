# U-Net_Lung-Segmentation
Application of U-Net in Lung Segmentation

## U-Net: Convolutional Networks for Biomedical Image Segmentation 
Writers: Olaf Ronneberger, Philipp Fischer, and Thomas Brox <br>
Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015, available at https://arxiv.org/pdf/1505.04597.pdf <br>
<br>
U-Net is a convolutional neural network architecture for fast and precise segmentation of images.<br>
<br>
**U-Net Architecture**<br>
![u-net-architecture](./u-net-architecture.png)<br>
U-net architecture (example for 32x32 pixels in the lowest resolution). Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the box. The x-y-size is provided at the lower left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations.
- U-net **learns segmentation** in an end-to-end setting
- **Very few annotated images** (approx. 30 per applications)
- **Touching objects** of the same class

