"""
transforms.py — paired image/mask augmentation.

Paper (Sec 4.2) uses: resize to 256x256, horizontal/vertical flip,
random rotation. We keep image and mask in sync by applying the
same random state to both.
"""
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


class PairedCompose:
    """Applies a list of paired (image, mask) transforms in sequence."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class PairedResize:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class PairedRandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class PairedRandomVFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask


class PairedRandomRotation:
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class PairedToTensor:
    """Image -> float tensor in [0,1], CxHxW. Mask -> float tensor in {0,1}, 1xHxW."""

    def __call__(self, image, mask):
        image = TF.to_tensor(image)  # C,H,W in [0,1]
        mask_np = np.array(mask, dtype=np.float32)
        if mask_np.max() > 1.0:
            mask_np = mask_np / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
        mask = TF.to_tensor(mask_np)
        return image, mask


def get_train_transforms(image_size=256, hflip=True, vflip=True, rotation_degrees=15):
    tfms = [PairedResize(image_size)]
    if hflip:
        tfms.append(PairedRandomHFlip(0.5))
    if vflip:
        tfms.append(PairedRandomVFlip(0.5))
    if rotation_degrees and rotation_degrees > 0:
        tfms.append(PairedRandomRotation(rotation_degrees))
    tfms.append(PairedToTensor())
    return PairedCompose(tfms)


def get_eval_transforms(image_size=256):
    return PairedCompose([PairedResize(image_size), PairedToTensor()])