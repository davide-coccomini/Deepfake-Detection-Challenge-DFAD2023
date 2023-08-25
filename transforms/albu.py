import random

import cv2
import numpy as np
import torch
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations.functional import crop

from torchsr.models import edsr_baseline

from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
from albumentations import RandomCrop
from scipy.fftpack import dct

def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]

    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down

    img = img.astype('uint8')
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


class Resize4xAndBack(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Resize4xAndBack, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = random.choice([2, 4])
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h),
                         interpolation=random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
        return img


class RandomSizedCropNonEmptyMaskIfExists(DualTransform):

    def __init__(self, min_max_height, w2h_ratio=[0.7, 1.3], always_apply=False, p=0.5):
        super(RandomSizedCropNonEmptyMaskIfExists, self).__init__(always_apply, p)

        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        cropped = crop(img, x_min, y_min, x_max, y_max)
        return cropped

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        mask_height, mask_width = mask.shape[:2]
        crop_height = int(mask_height * random.uniform(self.min_max_height[0], self.min_max_height[1]))
        w2h_ratio = random.uniform(*self.w2h_ratio)
        crop_width = min(int(crop_height * w2h_ratio), mask_width - 1)
        if mask.sum() == 0:
            x_min = random.randint(0, mask_width - crop_width + 1)
            y_min = random.randint(0, mask_height - crop_height + 1)
        else:
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, crop_width - 1)
            y_min = y - random.randint(0, crop_height - 1)
            x_min = np.clip(x_min, 0, mask_width - crop_width)
            y_min = np.clip(y_min, 0, mask_height - crop_height)

        x_max = x_min + crop_height
        y_max = y_min + crop_width
        y_max = min(mask_height, y_max)
        x_max = min(mask_width, x_max)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def get_transform_init_args_names(self):
        return "min_max_height", "height", "width", "w2h_ratio"

class CustomRandomCrop(DualTransform):
    def __init__(self, size, p=0.5) -> None:
        super(CustomRandomCrop, self).__init__()
        self.size = size
        self.prob = p

    def apply(self, img, copy=True, **params):
        if img.shape[0] < self.size or img.shape[1] < self.size:
            transform = IsotropicResize(max_side=self.size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR)
        else:
            transform = RandomCrop(self.size, self.size)
        return np.asarray(transform(image=img)["image"])

class FFT(DualTransform):
    def __init__(self, mode, p=0.5) -> None:
        super(FFT, self).__init__()
        self.prob = p
        self.mode = mode

    def apply(self, img, copy=True, **params):
        dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(rgb2gray(img)))
        mask = np.log(abs(dark_image_grey_fourier)).astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        if self.mode == 0:
            return np.asarray(cv2.bitwise_and(img, img, mask=mask))
        else:
            mask = np.asarray(mask)
            image =  cv2.merge((mask, mask, mask))
            return image

class SR(DualTransform):
    def __init__(self, model_sr, p=0.5) -> None:
        super(SR, self).__init__()
        self.prob = p
        self.model_sr = model_sr

    def apply(self, img, copy=True, **params):
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation = cv2.INTER_AREA)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float).unsqueeze(0).to(2)
        sr_img = self.model_sr(img)
        return sr_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


class DCT(DualTransform):
    def __init__(self, mode, p=0.5) -> None:
        super(DCT, self).__init__()
        self.prob = p
        self.mode = mode


    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def apply(self, img, copy=True, **params):

        gray_img = self.rgb2gray(img)
        dct_coefficients = dct(dct(gray_img, axis=0, norm='ortho'), axis=1, norm='ortho')
        epsilon = 1e-10
        mask = np.log(np.abs(dct_coefficients)+epsilon).astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        if self.mode == 0:
            return np.asarray(cv2.bitwise_and(img, img, mask=mask))
        else:
            mask = np.asarray(mask)
            image = cv2.merge((mask, mask, mask))
            return image




