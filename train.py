

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
import glob
import pandas as pd
import collections


ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif")

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=80, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--training_path', default='datasets/laion', type=str, metavar='PATH',
                        help='Path to the training images folder.')
    parser.add_argument('--validation_csv', default="../datasets/custom_validation_gan/validation_set.csv", type=str, metavar='PATH',
                        help='Path to the validation csv file.')
    parser.add_argument('--max_images', type=int, default=-1, 
                        help="Maximum number of images to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--gpu_id', default=2, type=int,
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


    train_paths = glob.glob(os.path.join(opt.training_path, "real-images", "*/*"), recursive = True)
    train_paths.extend(glob.glob(os.path.join(opt.training_path, "fake-images", "*/*"), recursive = True))
    train_paths = [path for path in train_paths if path.lower().endswith(ALLOWED_EXTENSIONS)]
    random.shuffle(train_paths)
    if opt.max_images > 0:
        train_paths = train_paths[:opt.max_images]
    train_labels = [1. if "fake" in path else 0. for path in train_paths]

    train_dataset = DeepFakesDataset(train_paths, train_labels, config['model']['image-size'])
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=1,
                                    persistent_workers=False)

    train_samples = len(train_dataset)
    val_df = pd.read_csv(opt.validation_csv)
    val_df = val_df.sample(frac = 1)
    val_paths = val_df["path"].tolist()
    val_labels = val_df['label'].tolist()

    validation_dataset = DeepFakesDataset(val_paths, val_labels, config['model']['image-size'], mode='validation')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=1,
                                    persistent_workers=False)
    
    validation_samples = len(validation_dataset)

    # Print some useful statistics
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(train_labels)
    print(train_counters)

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
        num_steps = int(opt.num_epochs * len(dl))
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
        
        bar = ChargingBar('EPOCH #' + str(t), max=(int(len(dl)/config["training"]["bs"])+int(len(val_dl)/config["training"]["bs"])))
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

            
        val_counter = 0
        val_correct = 0
        val_positive = 0
        val_negative = 0
    
        train_correct /= train_samples
        total_loss /= counter
        for index, (val_images, val_labels) in enumerate(val_dl):
    
            val_images = np.transpose(val_images, (0, 3, 1, 2))
            
            val_images = val_images.cuda()
            val_labels = val_labels.unsqueeze(1)
            val_pred = model(val_images)
            val_pred = val_pred.cpu()
            val_loss = loss_fn(val_pred, val_labels)
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
            val_correct += corrects
            val_positive += positive_class
            val_negative += negative_class
            val_counter += 1
            bar.next()
            
        scheduler.step()
        bar.finish()
        

        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0
            
            bar.next()
            train_images += images.shape[0]
            time_diff = unix_time_millis(datetime.now() - start_time)

        previous_loss = total_val_loss

        # Print intermediate metrics
        if index%100 == 0:
            expected_time = str(datetime.fromtimestamp((time_diff)*((len(dl)/config['training']['bs'])-index)/1000).strftime('%H:%M:%S.%f'))
            print("\nLoss: ", total_loss/counter, "Accuracy: ", train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive, "Expected Time:", expected_time)

        '''
        # Save checkpoint if the model's validation loss is improving
        if previous_loss > total_val_loss and t >= 20:
            if opt.model != 2:
                torch.save(features_extractor.state_dict(), os.path.join(opt.models_output_path,  "Extractor_checkpoint" + str(t)))
            torch.save(model.state_dict(), os.path.join(opt.models_output_path,  "Model_checkpoint" + str(t)))
        '''
    bar.finish()