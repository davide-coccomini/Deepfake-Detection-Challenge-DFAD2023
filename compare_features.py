

from sklearn.manifold import TSNE
import clip
import argparse
import torch
from datasets import load_dataset
import numpy as np
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate
import os
from transforms.albu import IsotropicResize
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def create_val_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--max_images', type=int, default=100000, 
                        help="Maximum number of images to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--gpu_id', default=5, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--comparing_set', default="datasets/test_set", type=str,
                        help='Set to compare')
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random state value')
    opt = parser.parse_args()
    print(opt)

    clip_model, preprocess = clip.load("RN50", device='cpu')
    clip_model.eval()
    clip_model = clip_model.to(opt.gpu_id)
 
    fake_embeddings = torch.zeros((0, 1024), dtype=torch.float32)
    real_embeddings = torch.zeros((0, 1024), dtype=torch.float32)
    test_embeddings = torch.zeros((0, 1024), dtype=torch.float32)
    elsa_data = load_dataset("rs9000/ELSA1M_track1", split="train", streaming=True)
    print("shuffling")
    elsa_data = elsa_data.shuffle(buffer_size=2000000, seed=opt.random_state)
    for index, data in enumerate(elsa_data):
        if index % 1000 == 0:
            print(index, fake_embeddings.shape[0], real_embeddings.shape[0])
        path = data["filepath"]
        if "fake" in path:
            label = 1
        else:
            label = 0
        
        if real_embeddings.shape[0] < opt.max_images and fake_embeddings.shape[0] == opt.max_images and label == 1:
            continue

        if fake_embeddings.shape[0] < opt.max_images and real_embeddings.shape[0] == opt.max_images and label == 0:
            continue
        
        if real_embeddings.shape[0] == opt.max_images and fake_embeddings.shape[0] == opt.max_images:
            break
        image = np.asarray(data.pop("image"))
        transform = create_val_transform(224)
        image = transform(image=image)['image']
        image = torch.tensor(image).unsqueeze(0).float()
        image = np.transpose(image, (0, 3, 1, 2))

        image = image.to(opt.gpu_id)
        
        image_features = clip_model.encode_image(image)

        path = data["filepath"]
        if "fake" in path:
            fake_embeddings = torch.cat((fake_embeddings, image_features.detach().cpu()), 0)
        else:
            real_embeddings = torch.cat((real_embeddings, image_features.detach().cpu()), 0)
        

    print("DONE2")
    for index, image_name in enumerate(os.listdir(opt.comparing_set)):
        if index > opt.max_images * 2:
            break
        image_path = os.path.join(opt.comparing_set, image_name)
        image = np.asarray(cv2.imread(image_path))
        transform = create_val_transform(224)
        image = transform(image=image)['image']
        image = torch.tensor(image).unsqueeze(0).float()
        image = np.transpose(image, (0, 3, 1, 2))
        image = image.to(opt.gpu_id)
        image_features = clip_model.encode_image(image)
        test_embeddings = torch.cat((test_embeddings, image_features.detach().cpu()), 0)
    print("DONE3")

    z = TSNE(n_components=2, init="pca").fit_transform(fake_embeddings)
    df = pd.DataFrame()
    df["y"] = 1
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 3),
                    data=df).set(title="Training Fake Data T-SNE projection") 
    plt.savefig('tsne-init.png')
    print("TSNE 2")
    z = TSNE(n_components=2, init="pca").fit_transform(real_embeddings)
    df = pd.DataFrame()
    df["y"] = 0
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 4),
                    data=df).set(title="Training Real Data T-SNE projection") 
    plt.savefig('tsne-half.png')
    print("TSNE LAST")
    z = TSNE(n_components=2, init="pca").fit_transform(test_embeddings)
    df = pd.DataFrame()
    df["y"] = 2
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 5),
                    data=df).set(title="Test Data T-SNE projection") 
    plt.savefig('tsne.png')