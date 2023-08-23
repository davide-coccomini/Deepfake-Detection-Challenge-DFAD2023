import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np
import uuid
from albumentations import Compose, RandomBrightnessContrast, RandomCrop, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ISONoise, MultiplicativeNoise, Cutout, CoarseDropout, MedianBlur, Blur, GlassBlur, MotionBlur, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, ToSepia, RandomShadow, RandomGamma, Rotate, Resize, RandomContrast, RandomBrightness, RandomBrightnessContrast

from transforms.albu import IsotropicResize, FFT, SR, DCT, CustomRandomCrop


class DeepFakesDataset(Dataset):
    def __init__(self, images, labels, size,  methods = None, model_sr = None, pre_load_images = False, image_mode = 0, mode = 'train'):
        self.x = images
        self.y = labels
        self.methods = methods
        self.image_size = size
        self.pre_load_images = pre_load_images
        self.mode = mode
        self.image_mode = image_mode
        self.model_sr = model_sr
        self.n_samples = len(images)    
        
    def create_train_transforms(self, size = 224, image_mode = 0):
        if image_mode == 0:
            return Compose([
                    ImageCompression(quality_lower=40, quality_upper=100, p=0.2),
                    GaussNoise(p=0.3),
                    ISONoise(p=0.3),
                    MultiplicativeNoise(p=0.3),
                    HorizontalFlip(),
                    OneOf([
                        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                        CustomRandomCrop(size=size)
                    ], p=1),
                    Resize(height=size, width=size),
                    PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                    OneOf([RandomBrightnessContrast(), RandomContrast(), RandomBrightness(), FancyPCA(), HueSaturationValue()], p=0.5),
                    OneOf([GaussianBlur(blur_limit=3), MedianBlur(), GlassBlur(), MotionBlur(), Blur()], p=0.5),
                    OneOf([Cutout(), CoarseDropout()], p=0.05),
                    ToGray(p=0.1),
                    ToSepia(p=0.05),
                    RandomShadow(p=0.05),
                    RandomGamma(p=0.1),
                    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                    FFT(mode=0, p=0.05),
                    #SR(model_sr=self.model_sr, p=0.03)
                ]
                )
        else:
            return Compose([
                ImageCompression(quality_lower=40, quality_upper=100, p=0.1),
                HorizontalFlip(),
                GaussNoise(p=0.3),
                ISONoise(p=0.3),
                MultiplicativeNoise(p=0.3),
                OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                    CustomRandomCrop(size=size)
                ], p=1),
                Resize(height=size, width=size),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                OneOf([RandomBrightnessContrast(), RandomContrast(), RandomBrightness(), FancyPCA(), HueSaturationValue()], p=0.5),
                OneOf([Cutout(), CoarseDropout()], p=0.05),
                ToGray(p=0.1),
                ToSepia(p=0.05),
                RandomShadow(p=0.05),
                RandomGamma(p=0.1),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                DCT(mode=0, p=1)
            ])
            
    def create_val_transform(self, size, image_mode = 0):
        if image_mode == 0:
            return Compose([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            ])
        else:
            return Compose([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                DCT(mode=1, p=1)
            ])

    def __getitem__(self, index):
        if self.pre_load_images:
            image = self.x[index]
        else:
            image = cv2.imread(self.x[index])
        label = self.y[index]
        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size, self.image_mode)
        elif self.mode == "validation":
            transform = self.create_val_transform(self.image_size, self.image_mode)
        else:
            transform1 = self.create_val_transform(self.image_size, 0)
            transform2 = self.create_val_transform(self.image_size, 1)
        
        if self.mode != "test":
            image = transform(image=image)['image']
            return torch.tensor(image).float(), float(label)
        else:
            images = [transform1(image=image)['image'], transform2(image=image)['image']]
            return torch.tensor(images[0]).float(), torch.tensor(images[1]).float(), self.x[index], float(label), self.methods[i]




    def __len__(self):
        return self.n_samples

 