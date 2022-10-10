import numpy as np
from torch import nn
from PIL import Image, ImageChops
from torchvision import transforms
from torchvision.transforms import (Resize,
                                    Compose,
                                    RandomVerticalFlip,
                                    RandomHorizontalFlip,
                                    InterpolationMode)
from configs import config as cfg

class CustomAffine(nn.Module):
    def __init__(self,
                 x_offset_range: tuple,
                 y_offset_range: tuple,
                 thresh: float=.3):
        super(CustomAffine, self).__init__()
        assert x_offset_range.__len__() == 2
        assert y_offset_range.__len__() == 2
        self.x_offset_range = x_offset_range
        self.y_offset_range = y_offset_range
        self.thresh = np.minimum(np.maximum(thresh, 0.), 1.)

    def forward(self, image):

        offset_random, affine_random = random.random(), random.random()
        x_offset = random.choice(range(*self.x_offset_range)) if offset_random < self.thresh else 0
        y_offset = random.choice(range(*self.y_offset_range)) if offset_random < self.thresh else 0
        width, height = image.size
        offset_image = ImageChops.offset(image, x_offset, y_offset) \
            if affine_random >= 0.5 \
            else ImageChops.offset(image, width-x_offset, height-y_offset)

        offset_image.paste((0, 0, 0), (0, 0, x_offset, height)) \
            if affine_random >= 0.5 \
            else offset_image.paste((0, 0, 0), (width-x_offset, 0, width, height))

        offset_image.paste((0, 0, 0), (0, 0, width, y_offset)) \
            if affine_random >= 0.5 \
            else offset_image.paste((0, 0, 0), (0, height-y_offset, width, height))

        return offset_image


class CustomRotation(nn.Module):
    def __init__(self,
                 degrees: tuple,
                 thresh: float=.3):
        super(CustomRotation, self).__init__()
        assert degrees.__len__() == 2
        self.degrees = degrees
        self.thresh = np.minimum(np.maximum(thresh, 0.), 1.)

    def forward(self, image):

        rotate_random = random.random()
        if rotate_random < self.thresh:
            angle = random.choice(range(*self.degrees))
            image = image.rotate(angle, Image.BILINEAR)

        return image

transformer = Compose([
    RandomHorizontalFlip(p=cfg.flip_prob),
    RandomVerticalFlip(p=cfg.flip_prob),
    CustomRotation(degrees=cfg.rotate_degrees),
    CustomAffine(x_offset_range=cfg.x_offset_range,
                 y_offset_range=cfg.y_offset_range)])


def image_preprocess(image, transform: bool=False):

    if transform:
        image = transformer(image)

    image = np.array(image)
    image = image / 127.5 - 1.
    image = np.clip(image, -1., 1.)
    image = image.transpose([2, 0, 1])

    return image
