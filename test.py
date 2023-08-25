

from datasets import load_dataset
import torch
from deepfakes_dataset import DeepFakesDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import yaml
import argparse
import math
from statistics import mean
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
from utils import check_correct, unix_time_millis, custom_round
from datetime import datetime, timedelta
from torchvision.models import resnet50, ResNet50_Weights
import glob
import pandas as pd
import collections
from sklearn.metrics import f1_score, roc_curve, auc
import timm

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default=80, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--test_folder', default='datasets/test_set', type=str, metavar='PATH',
                        help='Path to the test images.')
    parser.add_argument('--correct_labels_csv', default='', type=str, metavar='PATH',
                        help='Path to the labels csv file if available.')
    parser.add_argument('--output_path', default="predictions.json", type=str, metavar='PATH',
                        help='Path to the output json file.')
    parser.add_argument('--max_images', type=int, default=-1, 
                        help="Maximum number of images to use for training (default: all).")
    parser.add_argument('--ensemble', default=False, action="store_true",
                        help='Use multiple models for prediction.')
    parser.add_argument('--error_analysis', default=False, action="store_true",
                        help='Perform error analysis for custom validation set.')
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--gpu_id', default=4, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--model', default=0, type=int,
                        help='Model (0: Cross Efficient ViT; 1: Resnet50; 2: Swin).')
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random state value')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='Fake/Pristine threshold')
    parser.add_argument('--model1_weights', type=str, default="",
                        help='Model 1 weights.')
    parser.add_argument('--model2_weights', type=str, default="",
                        help='Model 2 weights.')
    parser.add_argument('--model3_weights', type=str, default="",
                        help='Model 3 weights.')
    opt = parser.parse_args()
    print(opt)
    torch.backends.cudnn.deterministic = True
    random.seed(opt.random_state)
    torch.manual_seed(opt.random_state)
    torch.cuda.manual_seed(opt.random_state)
    np.random.seed(opt.random_state)



    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 
    if opt.gpu_id == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = opt.gpu_id

    
    if opt.error_analysis and config['test']['bs'] > 1:
        raise Exception("The error analysis is available only with batch size equal to 1.")

    if opt.ensemble:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(2048, config['model']['num-classes'])
        model.eval()
        model2 = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', in_chans = 3, pretrained=True)
        model2.head.fc = torch.nn.Linear(1024, config['model']['num-classes'])
        model2.eval()
        model3 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model3.fc = torch.nn.Linear(2048, config['model']['num-classes'])
        model3.eval()

        models = [model.to(device), model2.to(device), model3.to(device)]
        
        if os.path.exists(opt.model1_weights):
            models[0].load_state_dict(torch.load(opt.model1_weights))
        else:
            raise Exception("No checkpoint loaded for the model 1.")  

        
        if os.path.exists(opt.model2_weights):
            models[1].load_state_dict(torch.load(opt.model2_weights))
        else:
            raise Exception("No checkpoint loaded for the model 2.")  

         
        if os.path.exists(opt.model3_weights):
            models[2].load_state_dict(torch.load(opt.model3_weights))
        else:
            raise Exception("No checkpoint loaded for the model 3.")  

    else:            
        if opt.model == 0:
            model = CrossEfficientViT(config=config)
        elif opt.model == 1:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = torch.nn.Linear(2048, config['model']['num-classes'])
        elif opt.model == 2:
            model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', in_chans = 3, pretrained=True)
            model.head.fc = torch.nn.Linear(1024, config['model']['num-classes'])
        if os.path.exists(opt.model1_weights):
            model.load_state_dict(torch.load(opt.model1_weights))
        else:
            raise Exception("No checkpoint loaded for the model.")    

        model.eval()
    
    if opt.gpu_id == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = opt.gpu_id


    if opt.gpu_id == -1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    if opt.correct_labels_csv != "":
        test_df = pd.read_csv(opt.correct_labels_csv)
        test_paths = test_df["path"].tolist()
        test_labels = test_df['label'].tolist()
        test_methods = test_df["method"].tolist()
    else:
        test_paths = os.listdir(opt.test_folder)
        test_paths = sorted(test_paths, key=lambda path:int(path.split("_")[1].split(".")[0]), reverse=False)
        if opt.max_images != -1:
            test_paths = test_paths[:opt.max_images]
        test_paths = [os.path.join(opt.test_folder, filename) for filename in test_paths]
        test_labels = [1 for path in test_paths] # Unknown labels
        test_methods = ["Unknown" for path in test_paths]

    if opt.max_images != -1:
        test_paths = test_paths[:opt.max_images]
        test_labels = test_labels[:opt.max_images]
        test_methods = test_methods[:opt.max_images]
    test_samples = len(test_paths)
    correct_test_labels = test_labels
    
    test_counters = collections.Counter(test_labels)

    test_dataset = DeepFakesDataset(test_paths, test_labels, config['model']['image-size'], methods = test_methods,  mode='test')


    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config['test']['bs'], shuffle=False, sampler=None,
                                batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                pin_memory=False, drop_last=False, timeout=0,
                                worker_init_fn=None, prefetch_factor=2,
                                persistent_workers=False)
    if opt.error_analysis:
        errors = {}

    bar = ChargingBar('PREDICT', max=(len(test_dl)))
    preds = []
    names = []
    test_counter = 0
    test_correct = 0
    test_positive = 0
    test_negative = 0
    correct_test_labels = []
    for index, (images, images_dct, image_path, labels, methods) in enumerate(test_dl):
        with torch.no_grad():

            labels = labels.unsqueeze(1)
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.to(device)
            if opt.ensemble:
                images_dct = np.transpose(images_dct, (0, 3, 1, 2))
                images_dct = images_dct.to(device)
                tmp_preds = []
                #tmp_preds.append(mean([torch.sigmoid(models[0](images).cpu()).item(), torch.sigmoid(models[1](images).cpu()).item()]))
                #tmp_preds.append(torch.sigmoid(models[0](images).cpu()).item())
                tmp_preds.append(torch.sigmoid(models[1](images).cpu()).item())
                tmp_preds.append(torch.sigmoid(models[2](images).cpu()).item())
                y_pred = mean(tmp_preds)
                preds.extend([y_pred])
                correct_test_labels.extend(labels)

            else:
                y_pred = model(images)
                y_pred = y_pred.cpu()
                preds.extend(torch.sigmoid(torch.tensor(y_pred)))
                correct_test_labels.extend(labels)
            names.append(os.path.basename(image_path[0]))
            
            corrects, positive_class, negative_class = check_correct(y_pred, labels, opt.ensemble, opt.threshold)  

            if opt.error_analysis:
                if corrects == 0:
                    method = methods[0]
                    if method in errors:
                        errors[method] += 1
                    else:
                        errors[method] = 1

            test_correct += corrects
            test_positive += positive_class
            test_counter += 1
            test_negative += negative_class

            bar.next()

    if opt.error_analysis:
        print("__ERRORS PER METHOD__")
        print(errors)

    with open(opt.output_path, "w+") as f:
        f.write("{")
        for i in range(len(names)):
            if opt.ensemble:
                f.write("\n\"" + names[i] + "\": " + str(custom_round(preds[i], opt.threshold)) + ",")
            else:
                f.write("\n\"" + names[i] + "\": " + str(custom_round(preds[i].item(), opt.threshold)) + ",")
        f.write("\n}")
    f.close()
    
    test_preds_counter = collections.Counter(preds)
    correct_test_labels = [int(label.item()) for label in correct_test_labels]
    if opt.correct_labels_csv != "":
        if opt.ensemble:
            fpr, tpr, th = roc_curve(correct_test_labels, [custom_round(pred, opt.threshold) for pred in preds])
            auc = auc(fpr, tpr)
            f1 = f1_score(correct_test_labels,  [custom_round(pred, opt.threshold) for pred in preds])
        else:
            fpr, tpr, th = roc_curve(correct_test_labels, [custom_round(pred.item(), opt.threshold) for pred in preds])
            auc = auc(fpr, tpr)
            f1 = f1_score(correct_test_labels,  [custom_round(pred.item(), opt.threshold) for pred in preds])
        bar.finish()
        test_correct /= test_samples
        print("F1 score: " + str(f1) + " test accuracy:" + str(test_correct) + " test_0s:" + str(test_negative) + "/" + str(test_counters[0]) + " test_1s:" + str(test_positive) + "/" + str(test_counters[1]) + " AUC " + str(auc))
    
