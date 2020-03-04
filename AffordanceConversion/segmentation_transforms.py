import numbers
import random

import numpy as np
from PIL import Image

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

# For Use with PIL Image Input and Target (ex. Auto Encoder)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=True, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be cropped.
            target (PIL Image): Target to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            target = F.pad(target, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(
                img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            target = F.pad(
                target, (self.size[1] - target.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(
                img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            target = F.pad(
                target, (0, self.size[0] - target.size[1]), self.fill, self.padding_mode)

        i, j, h, w = T.RandomCrop.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(target, i, j, h, w)

class RandomCutout(object):
    def __init__(self, size, fill=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to remove a chunk of size 'size'.
            target (PIL Image): Target to stay the same.

        Returns:
            img, target: image and modified target.
        """
        w, h = img.size
        th, tw = self.size

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        img.paste(0, box=(j, i, j+tw, i+th))
        return img, target

class CentralCutout(object):
    def __init__(self, size, fill=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to remove a chunk of size 'size'.
            target (PIL Image): Target to stay the same.

        Returns:
            img, target: image and modified target.
        """
        w, h = img.size
        th, tw = self.size

        i = int(h // 2 - th // 2)
        j = int(w // 2 - tw // 2)
        end_w = j + tw if j+tw < w else w
        end_h = i + th if i+th < h else h
        img.paste(0, box=(j, i, end_w, end_h))
        return img, target

class GaussianNoise(object):
    """
    Inputs are torch tensors, not PIL Image. Target is not affected
    """
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
    
    def __call__(self, image, target):
        image = image + torch.randn(image.shape) * self.std + self.mean
        # target = target + torch.randn(target.shape) * self.std + self.mean
        return image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target

class SingleToTensor(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        return image

class ToPIL(object):
    def __call__(self, image, target):
        image = F.to_pil_image(image)
        target = F.to_pil_image(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        target = F.normalize(target, mean=self.mean, std=self.std)
        return image, target


def img_norm(image):
    image = image.clone()

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))
    norm_range(image, None)
    return image


# TODO verify these normalizations work well for pixelated games
# TODO Denoising autoencoder
# DEFAULT_MEAN = [0.485, 0.456, 0.406]
# DEFAULT_STD = [0.229, 0.224, 0.225]
# Super Mario Bros world Mean and Std
DEFAULT_MEAN = [0.3711, 0.3652, 0.5469]
DEFAULT_STD = [0.2973, 0.2772, 0.4554]
BASE_SIZE = 224
CROP_SIZE = 180

def get_transform(train=False, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    transforms = []
    if train:
        # min_size = int(0.5 * BASE_SIZE)
        max_size = int(1.5 * BASE_SIZE)

        transforms.append(RandomResize(BASE_SIZE, max_size))
        
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))

        transforms.append(RandomCrop(CROP_SIZE))

    transforms.append(Resize(BASE_SIZE))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=mean,
                                  std=std))

    return Compose(transforms)

def get_inpaint_transform(train=False, mean=DEFAULT_MEAN, std=DEFAULT_STD, cutout_pixels=60):
    transforms = []
    if train:
        # min_size = int(0.5 * BASE_SIZE)
        max_size = int(1.5 * BASE_SIZE)

        transforms.append(RandomResize(BASE_SIZE, max_size))

        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))

        transforms.append(RandomCrop(CROP_SIZE))


    transforms.append(Resize(BASE_SIZE))
    transforms.append(CentralCutout(cutout_pixels))

    transforms.append(ToTensor())
    transforms.append(Normalize(mean=mean,
                                std=std))

    return Compose(transforms)
    
def get_noisy_transform(train=False, mean=DEFAULT_MEAN, std=DEFAULT_STD, noise_mean=0.0, noise_std=1.0):
    transforms = get_transform(train, mean, std)

    return Compose([transforms, GaussianNoise(mean=noise_mean, std=noise_std)])
