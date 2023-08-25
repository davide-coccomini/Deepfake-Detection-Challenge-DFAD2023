

from datasets import load_dataset
import torch
from deepfakes_dataset import DeepFakesDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import yaml
import argparse
import math
from progress.bar import ChargingBar 
from albumentations import Cutout, CoarseDropout, RandomCrop, RandomGamma, MedianBlur, ISONoise, MultiplicativeNoise, ToSepia, RandomShadow, MultiplicativeNoise, RandomSunFlare, GlassBlur, RandomBrightness, MotionBlur, RandomRain, RGBShift, RandomFog, RandomContrast, Downscale, InvertImg, RandomContrast, ColorJitter, Compose, RandomBrightnessContrast, CLAHE, ISONoise, JpegCompression, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate, Normalize, Resize
from transforms.albu import IsotropicResize, FFT, SR
from timm.scheduler.cosine_lr import CosineLRScheduler
import cv2
import numpy as np
from cross_efficient_vit import CrossEfficientViT
import math
import random
import os
from utils import check_correct, unix_time_millis
from datetime import datetime, timedelta
from torchvision.models import resnet50, ResNet50_Weights



def create_train_transforms(model_sr, size = 224, model = 0):
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
                    RandomCrop(size, size)
                ], p=1),
                Resize(height=size, width=size),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                OneOf([RandomBrightnessContrast(), RandomContrast(), RandomBrightness(), FancyPCA(), HueSaturationValue()], p=0.5),
                OneOf([GaussianBlur(blur_limit=3), MedianBlur(), GlassBlur(), MotionBlur()], p=0.1),
                OneOf([Cutout(), CoarseDropout(), PixelDropout()], p=0.05),
                ToGray(p=0.1),
                ToSepia(p=0.05),
                RandomShadow(p=0.05),
                RandomGamma(p=0.1),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                FFT(p=0.05),
                SR(model_sr=model_sr, p=0.05)
            ]
            )
    else:
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            FFT(p=1)
        ])

def collate(batch, transform):
    images = []
    labels = []
    for data in batch:
        path = data["filepath"]
        if "fake" in path:
            label = 1
        else:
            label = 0

        images.append(transform(image=np.asarray(data.pop("image")))["image"])
        labels.append(label)
        
    return torch.tensor(np.asarray(images)).float(), torch.tensor(np.asarray(labels)).float()



estimated_images = 2000000
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--max_images', type=int, default=-1, 
                        help="Maximum number of images to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--gpu_id', default=5, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--model', default=0, type=int,
                        help='Model (0: Cross Efficient ViT; 1: FFT-Resnet50).')
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random state value')
    opt = parser.parse_args()
    print(opt)


    torch.backends.cudnn.deterministic = True
    random.seed(opt.random_state)
    torch.manual_seed(opt.random_state)
    torch.cuda.manual_seed(opt.random_state)
    np.random.seed(opt.random_state)


    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 
    if opt.model == 0:
        model = CrossEfficientViT(config=config)
    else:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(2048, config['model']['num-classes'])

    if opt.gpu_id == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = opt.gpu_id

    model = model.to(device)

    if opt.gpu_id == -1:
        model = torch.nn.DataParallel(model)


    elsa_data = load_dataset("rs9000/ELSA1M_track1", split="train", streaming=True)
    elsa_data = elsa_data.shuffle(buffer_size=2000000, seed=opt.random_state)
    elsa_data = elsa_data.with_format("torch")

    if opt.model == 0:
        model_edsr = edsr_baseline(scale=2, pretrained=True)
        model_edsr = model_edsr.eval()
        model_edsr = model_edsr.to(device)
        if opt.gpu_id == -1:
            model_edsr = torch.nn.DataParallel(model_edsr)

    transform = create_train_transforms(model_edsr, opt.model)


    dl = torch.utils.data.DataLoader(elsa_data, batch_size=config['training']['bs'], sampler=None,
                                    batch_sampler=None, num_workers=10, collate_fn=lambda batch: collate(batch, transform=transform),
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=1,
                                    persistent_workers=False)



    
     # Init optimizers
    parameters =  model.parameters()

    if config['training']['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    elif config['training']['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    elif config['training']['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    else:
        print("Error: Invalid optimizer specified in the config file.")
        exit()

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Init LR schedulers
    if config['training']['scheduler'].lower() == 'steplr':   
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    elif config['training']['scheduler'].lower() == 'cosinelr':
        num_steps = int(opt.num_epochs * 100000)
        lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=config['training']['lr'] * 1e-2,
                cycle_limit=1,
                t_in_epochs=False,
        )
    else:
        print("Warning: Invalid scheduler specified in the config file.")



    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf   
    for t in range(0, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break

        counter = 0
        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(estimated_images/config['training']['bs']))
        train_correct = 0
        positive = 0
        negative = 0 
        train_images = 0
        for index, (images, labels) in enumerate(dl):
            start_time = datetime.now()
            labels = labels.unsqueeze(1)
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.to(device)
            y_pred = model(images)
            y_pred = y_pred.cpu()
            loss = loss_fn(y_pred, labels)
        
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)
            
            bar.next()
            train_images += images.shape[0]
            time_diff = unix_time_millis(datetime.now() - start_time)

             
            # Print intermediate metrics
            if index%100 == 0:
                expected_time = str(datetime.fromtimestamp((time_diff)*((estimated_images/config['training']['bs'])-index)/1000).strftime('%H:%M:%S.%f'))
                print("\nLoss: ", total_loss/counter, "Accuracy: ", train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive, "Expected Time:", expected_time)

            '''
            # Save checkpoint if the model's validation loss is improving
            if previous_loss > total_val_loss:
                if opt.model != 2:
                    torch.save(features_extractor.state_dict(), os.path.join(opt.models_output_path,  "Extractor_checkpoint" + str(t)))
                torch.save(model.state_dict(), os.path.join(opt.models_output_path,  "Model_checkpoint" + str(t)))
            '''
        bar.finish()