

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
from multiprocessing import Manager
from multiprocessing.pool import Pool
from functools import partial
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
from torchsr.models import edsr_baseline
from sklearn.metrics import f1_score
import timm

def read_images(path, dataset):
    if "fake" in path:
        label = 1
    else:
        label = 0
    dataset.append([cv2.imread(path), label])
    


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=60, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=80, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--training_csv', default='datasets/laion/training_set_3_cleaned.csv', type=str, metavar='PATH',
                        help='Path to the training csv file.')
    parser.add_argument('--validation_csv', default="../datasets/custom_validation_gan/validation_set.csv", type=str, metavar='PATH',
                        help='Path to the validation csv file.')
    parser.add_argument('--models_output_path', default="models", type=str, metavar='PATH',
                        help='Path to the validation csv file.')
    parser.add_argument('--max_images', type=int, default=-1, 
                        help="Maximum number of images to use for training (default: all).")
    parser.add_argument('--pre_load_images', default=False, action="store_true",
                        help='Pre-load the images in memory (True) or load from path in the dataloader.')
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--gpu_id', default=4, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--model', default=0, type=int,
                        help='Model (0: Cross Efficient ViT; 1: Resnet50; 2: Swin).')
    parser.add_argument('--image_mode', default=0, type=int,
                        help='(0: Normal; 1: DCT)')
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=10, 
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
    elif opt.model == 1:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(2048, config['model']['num-classes'])
    elif opt.model == 2:
        model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', in_chans = 3, pretrained=True)
        model.head.fc = torch.nn.Linear(1024, config['model']['num-classes'])
        for index, (name, param) in enumerate(model.named_parameters()):
            param.requires_grad = True

    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1]) + 1 # The checkpoint's file name format should be "checkpoint_EPOCH"
        print("Weights loaded.")


    if opt.gpu_id == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = opt.gpu_id


    if opt.gpu_id == -1:
        model = torch.nn.DataParallel(model)


    model = model.to(device)
    '''
    model_sr = edsr_baseline(scale=2, pretrained=True)
    model_sr = model_sr.eval()
    model_sr = model_sr.to(device)
    if opt.gpu_id == -1:
        model_sr = torch.nn.DataParallel(model_sr)
    '''
    train_df = pd.read_csv(opt.training_csv, names=["path", "label"]) 
    train_df = train_df.sample(frac = 1)

    if opt.pre_load_images:
        train_paths = train_df["path"].tolist()
        if opt.max_images > 0:
            train_paths = train_paths[:opt.max_images]
        mgr = Manager()
        train_dataset = mgr.list()
        print("Reading training images...")
        with Pool(processes=opt.workers) as p:
            with tqdm(total=len(train_paths)) as pbar:
                for v in p.imap_unordered(partial(read_images, dataset=train_dataset),train_paths):
                    pbar.update()
        train_dataset = DeepFakesDataset([row[0] for row in train_dataset], [row[1] for row in train_dataset], config['model']['image-size'], pre_load_images = opt.pre_load_images, image_mode=opt.image_mode)
    else:
        train_paths = train_df["path"].tolist()
        train_labels = train_df['label'].tolist()
        if opt.max_images > 0:
            train_paths = train_paths[:opt.max_images]
            train_labels = train_labels[:opt.max_images]
        train_dataset = DeepFakesDataset(train_paths, train_labels, config['model']['image-size'], pre_load_images = opt.pre_load_images, image_mode=opt.image_mode)

    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=False, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=1,
                                    persistent_workers=False)

    train_samples = len(train_dataset)
    val_df = pd.read_csv(opt.validation_csv)
    val_df = val_df.sample(frac = 1)

    if opt.pre_load_images:
        val_paths = val_df["path"].tolist()
        
        validation_samples = len(val_paths)
        if opt.max_images > 0:
            val_paths = val_paths[:opt.max_images]

        validation_dataset = mgr.list()
        print("Reading validation images...")
        with Pool(processes=opt.workers) as p:
            with tqdm(total=len(val_paths)) as pbar:
                for v in p.imap_unordered(partial(read_images, dataset=validation_dataset),val_paths):
                    pbar.update()
        correct_val_labels = [row[1] for row in validation_dataset]
        validation_dataset = DeepFakesDataset([row[0] for row in validation_dataset], [row[1] for row in val_dataset], config['model']['image-size'], pre_load_images = opt.pre_load_images, mode='validation', image_mode = opt.image_mode)
    else:
        val_paths = val_df["path"].tolist()
        val_labels = val_df['label'].tolist()
        if opt.max_images > 0:
            val_paths = val_paths[:opt.max_images]
            val_labels = val_labels[:opt.max_images]
        validation_samples = len(val_paths)
        correct_val_labels = val_labels
        validation_dataset = DeepFakesDataset(val_paths, val_labels, config['model']['image-size'],  mode='validation',  image_mode = opt.image_mode)


    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=False, sampler=None,
                                batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                pin_memory=False, drop_last=False, timeout=0,
                                worker_init_fn=None, prefetch_factor=2,
                                persistent_workers=False)
                                
    # Print some useful statistics
    print("Train images:", train_samples, "Validation images:", validation_samples)
    print("__TRAINING STATS__")
    train_counters = collections.Counter(train_labels)
    print(train_counters)
    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(val_labels)
    print(val_counters)
    print("___________________")


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

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    # Init LR schedulers
    if config['training']['scheduler'].lower() == 'steplr':   
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    elif config['training']['scheduler'].lower() == 'cosinelr':
        num_steps = int(opt.num_epochs * len(dl))
        scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=config['training']['lr'] * 1e-2,
                cycle_limit=2,
                t_in_epochs=False,
        )
    else:
        print("Warning: Invalid scheduler specified in the config file.")



    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf   
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break

        counter = 0
        total_loss = 0
        total_val_loss = 0
        bar = ChargingBar('EPOCH #' + str(t), max=(int(len(dl))+int(len(val_dl))))
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
            if config['training']['scheduler'].lower() == 'cosinelr':
                scheduler.step_update((t * (len(dl)) + index))
            else:
                scheduler.step()
            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)
            
            # Update time per epoch
            time_diff = unix_time_millis(datetime.now() - start_time)

            bar.next()
            # Print intermediate metrics
            if index%5 == 0:
                expected_time = str(datetime.fromtimestamp((time_diff)*(len(dl)-index)/1000).strftime('%H:%M:%S.%f'))
                print("\nLoss: ", total_loss/counter, "Accuracy: ", train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive, "Expected Time:", expected_time)

            
        val_counter = 0
        val_correct = 0
        val_positive = 0
        val_negative = 0
    
        train_correct /= train_samples
        total_loss /= counter
        val_preds = []
        for index, (val_images, val_labels) in enumerate(val_dl):
            with torch.no_grad():
                val_images = np.transpose(val_images, (0, 3, 1, 2))
                val_images = val_images.to(device)
                val_labels = val_labels.unsqueeze(1)
                val_pred = model(val_images)
                val_pred = val_pred.cpu()
                val_preds.extend(val_pred)
                val_loss = loss_fn(val_pred, val_labels)
                total_val_loss += round(val_loss.item(), 2)
                corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
                val_correct += corrects
                val_positive += positive_class
                val_negative += negative_class
                val_counter += 1
                bar.next()
   
        bar.finish()
        

        total_val_loss /= val_counter
        val_correct /= validation_samples
        val_preds = [torch.sigmoid(torch.tensor(pred)) for pred in val_preds]

        f1 = f1_score(correct_val_labels, [round(pred.item()) for pred in val_preds])
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0
            
            bar.next()
            train_images += images.shape[0]


        # Print metrics
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" + str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_f1:", str(round(f1, 2)) + " val_0s:" + str(val_negative) + "/" + str(val_counters[0]) + " val_1s:" + str(val_positive) + "/" + str(val_counters[1]))
    

        # Save checkpoint if the model's validation loss is improving
        if previous_loss > total_val_loss and t >= 3:
            if opt.model == 0:
                model_name = "CrossViT"
            elif opt.model == 1:
                model_name = "Resnet50"
            elif opt.model == 2:
                model_name = "Swin"

            
            if opt.image_mode == 1:
                model_name += "DCTfull"
            elif opt.image_mode == 2:
                model_name += "noaug"
            torch.save(model.state_dict(), os.path.join(opt.models_output_path, model_name + "_checkpoint" + str(t)))
        
        
        previous_loss = total_val_loss

    bar.finish()