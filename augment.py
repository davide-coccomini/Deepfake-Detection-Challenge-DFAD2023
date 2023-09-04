
from albumentations import Compose, RandomBrightnessContrast, RandomCrop, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ISONoise, MultiplicativeNoise, Cutout, CoarseDropout, MedianBlur, Blur, GlassBlur, MotionBlur, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, ToSepia, RandomShadow, RandomGamma, Rotate, Resize, RandomContrast, RandomBrightness, RandomBrightnessContrast
from PIL import Image

from transforms.albu import IsotropicResize, FFT, SR, DCT, CustomRandomCrop
import cv2
import numpy as np
import os 
import imageio

size = 224
transform = Compose([
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
        FFT(mode=0, p=0.05),
        DCT(mode=1, p=0.5)
    ])



IMAGE_PATH = "images/face.png"

transformed_images = []
image = cv2.imread(IMAGE_PATH)
for i in range(200):
    transformed_image = transform(image=image)["image"]
    transformed_images.append(transformed_image)

num_rows = 10
num_cols = 20


# Create an empty canvas to arrange the images in a grid
canvas_height = num_rows * transformed_images[0].shape[0]
canvas_width = num_cols * transformed_images[0].shape[1]
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Initialize row and column counters
row, col = 0, 0


# Iterate through the transformed_images and place them on the canvas
for i, image in enumerate(transformed_images):
    canvas[row:row+image.shape[0], col:col+image.shape[1]] = image
    col += image.shape[1]
    if (i + 1) % num_cols == 0:
        row += image.shape[0]
        col = 0


output_path = "augmented_images.png"
cv2.imwrite(output_path, canvas)


# Specify the output GIF filename
output_gif_path = "augmented_images.gif"

# Create the GIF by saving each image in the list to a temporary file
with imageio.get_writer(output_gif_path, mode='I', duration=0.5) as writer:
    for image in transformed_images:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        writer.append_data(image_bgr)

    