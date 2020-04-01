import torchvision.transforms as transforms
import random
import torch.nn.functional as F
import numpy as np
#transforms = {Resize, ToTensor, RandomCrop, ToPILImage}

class GrayScale(object):
    def __call__(self,sample):
        from torchvision.transforms import Grayscale
        Grayscale = Grayscale()
        sample['image'] = Grayscale(sample['image'])
        return sample

class Resize(object):
    """
    Resize the input PIL Image to the given size.
    """
    def __init__(self,img_size):
        assert isinstance(img_size , (int,tuple))
        self.img_size = img_size

    def __call__(self,sample):
        img , mask = sample['image'],sample['mask']
        Resize = transforms.Resize((self.img_size,self.img_size))
        sample['image'],sample['mask'] = Resize(img), Resize(mask)
        return sample
class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class ColorJitter(object):
    def __init__(self,brightness=0, contrast=0, saturation=0, hue=0):
        """

        :param brightness:
        :param contrast:
        :param saturation:
        :param hue:
        """
        from torchvision.transforms import ColorJitter
        self.ColorJitter = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self,sample):
        return {"image":self.ColorJitter(sample["image"]),
                "mask" :sample["mask"]}



# class RandomCrop(object):
#     """Crop randomly the image in a sample
#
#     Args:
#         output_size (tuple or int): Desired output size.
#         If int, square crop is made
#     """
#     def __init__(self,output_size):
#         assert isinstance(output_size, (int,tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
#
#     def __call__(self,sample):
#         img, mask = sample['image'], sample['mask']
#
#         # h,w = img.shape[:2] # numpy img : H X W X C
#         w,h = img.size
#         new_h , new_w = self.output_size
#
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0,w - new_w)
#
#         img = img[top:top + new_h,
#                   left: left + new_w]
#         mask = mask[top:top + new_h,
#                   left: left + new_w]
#
#         sample['image'], sample['mask'] = img, mask
#         return sample

class ToTensor(object):
    """convert ndarrays in sample to Tensors"""
    def __call__(self,sample):
        from torchvision.transforms import ToTensor
        ToTensor = ToTensor()
        img , mask = sample['image'],sample['mask']
        sample['image'],sample['mask'] = ToTensor(img) ,ToTensor(mask)
        return sample



# class Rescale(object):
#     """
#     Rescale the image in a sample to a given size
#     """
#     def __init__(self,scale):
#         self.scale = scale
#
#     def __call__(self,sample):
#         import torchvision.transforms as transforms
#         img , mask = sample['image'],sample['mask']
#         Scale = transforms.Scale()
#         resize = transforms.Resize((self.img_size,self.img_size))
#         sample['image'],sample['mask'] = resize(img), resize(mask)
#         return sample


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p =0.5):
        self.p = p
    def __call__(self,sample):
        from torchvision.transforms.functional import vflip as vertical_flip
        img , mask = sample['image'],sample['mask']
        if random.random() < self.p:
                sample['image'], sample['mask'] = vertical_flip(img), vertical_flip(mask)
        return sample

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, sample):
        from torchvision.transforms.functional import hflip as horizontal_flip
        img , mask = sample['image'],sample['mask']
        if random.random() < self.p:
                sample['image'], sample['mask'] = horizontal_flip(img), horizontal_flip(mask)
        return sample

class ToPILImage(object):
    def __call__(self,sample):
        from torchvision.transforms import ToPILImage
        img , mask = sample['image'],sample['mask']
        ToPILImage = ToPILImage()
        sample['image'], sample['mask'] = ToPILImage(img),ToPILImage(mask)
        return sample
