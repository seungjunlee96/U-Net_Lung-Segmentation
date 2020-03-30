from PIL import Image
import numpy as np

def mask_to_image(mask):
    return Image.fromarray((mask*255)).astype(np.uint8)