# U-Net_Lung-Segmentation
Application of U-Net in Lung Segmentation

## U-Net: Convolutional Networks for Biomedical Image Segmentation 
Writers: Olaf Ronneberger, Philipp Fischer, and Thomas Brox <br>
Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015, available at https://arxiv.org/pdf/1505.04597.pdf <br>
<br>
U-Net is a convolutional neural network architecture for fast and precise segmentation of images.<br>
<br>
## U-Net Architecture
![u-net-architecture](./u-net-architecture.png)<br>
U-net architecture (example for 32x32 pixels in the lowest resolution). Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the box. The x-y-size is provided at the lower left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations.<br>
<br>
### Contracting path (Downsampling)
It consists of the repeated application of :<br>
- Two **3x3 convolutions** (unpadded convolutions)
- Followed by a **ReLU** (Rectified Linear Unit)
- A **2x2 max pooling** operation with stride 2 for downsampling.<br>
### Expansive path (Upsampling)
It consists of the repeated application of :<br>
- **2x2 convolution ("up-convolution")** that halves the number of feature channels
- A **concatenation** with the correspondingly cropped feature map from the contracting path
- Two **3x3 convolutions**
- Followed by a **ReLU**
At the final layer a **1x1 convolution** is used to map each 64 componet feature vector to the desired number of classes.<br>
## challenges
- Very few annotated Very few annotated images available (approx. 30 per applications)
- **Touching objects** of the same class<br>
Important trick:<br>
select the input tile size such that all 2x2 max-pooling operations are applied to a layer with an even x- and y-size
## Contributions
- U-net learns **segmentation** in an **end-to-end** setting (beats the prior best method, a sliding-window CNN, with large margin.) 
- **excessive data augmentation** by applying **elastic deformations** which used to be the most common variation in tissue and realistic deformations can be simulated efficiently.
- The use of **a weighted loss**, where the separating background labels between **touching cells** of the same class

