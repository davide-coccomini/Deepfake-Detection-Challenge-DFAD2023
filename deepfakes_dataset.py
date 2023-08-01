import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np
import uuid
from albumentations import Compose, RandomBrightnessContrast, RandomCrop, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ISONoise, MultiplicativeNoise, Cutout, CoarseDropout, MedianBlur, GlassBlur, MotionBlur, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, ToSepia, RandomShadow, RandomGamma, Rotate, Resize, RandomContrast, RandomBrightness, RandomBrightnessContrast

from transforms.albu import IsotropicResize, FFT, SR, CustomRandomCrop


class DeepFakesDataset(Dataset):
    def __init__(self, images, labels, size, model_sr = None, mode = 'train'):
        self.x = images
        self.y = labels
        self.image_size = size
        self.mode = mode
        self.model_sr = model_sr
        self.n_samples = len(images)    
        
    def create_train_transforms(self, size = 224, model = 0):
        if model == 0:
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
                    OneOf([GaussianBlur(blur_limit=3), MedianBlur(), GlassBlur(), MotionBlur()], p=0.1),
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
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                FFT(mode=1, p=1)
            ])
            
    def create_val_transform(self, size):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.x[index])
        if image is None:
            print(self.x[index].replace(" ", "we"))
        label = self.y[index]
        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transform(self.image_size)

        image = transform(image=image)['image']
        return torch.tensor(image).float(), float(label)




    def __len__(self):
        return self.n_samples

 