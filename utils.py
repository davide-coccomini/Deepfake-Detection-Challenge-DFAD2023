
import numpy as np
import os
import cv2
import torch
from statistics import mean

def check_correct(preds, labels, ensemble=False, threshold = 0.5):
    labels = labels.cpu()
    if not ensemble:
        preds = preds.cpu()
        preds = [custom_round(torch.sigmoid(pred).detach().numpy(), threshold = threshold) for pred in preds]
    else:
        preds = [custom_round(preds, threshold = threshold)]
    correct = 0
    positive_class = 0
    negative_class = 0
    for i in range(len(labels)):
        pred = int(preds[i])
        if labels[i] == pred:
            correct += 1
        if pred == 1:
            positive_class += 1
        else:
            negative_class += 1
    return correct, positive_class, negative_class

    
def unix_time_millis(dt):
    return dt.total_seconds() * 1000.0

def custom_round(value, threshold):
    if type(value) == list:
        result = []
        for v in value:
            if v > threshold:
                result.append(1)
            else:
                result.append(0)
    else:
        if value > threshold:
            return 1
        else:
            return 0