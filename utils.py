
import numpy as np
import os
import cv2
import torch
from statistics import mean

def check_correct(preds, labels):
    preds = preds.cpu()
    labels = labels.cpu()
    preds = [np.asarray(torch.sigmoid(pred).detach().numpy()).round() for pred in preds]

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