import os
import shutil
import random
import pandas as pd
from datasets import load_dataset
import cv2
REAL_IMAGES_PATH = "../datasets/custom_validation_gan/real_images"
FAKE_IMAGES_PATH = "../datasets/custom_validation_gan/fake_images"
DIFFUSED_PATHS = ["../datasets/diffused_coco/test", "../datasets/diffused_wikipedia/test", "../datasets/glide_diffused_coco/test", "../datasets/glide_diffused_wikipedia/test"]
FLICKR_IMAGES = "../datasets/flickr/flickr30k_images/flickr30k_images"
GAN_IMAGES = "../datasets/unina-gans"


random.seed(42)

df = pd.DataFrame(columns=["path", "label", "method", "dataset"])


# CREATE THE MAIN OUTPUT FOLDERS
os.makedirs(REAL_IMAGES_PATH, exist_ok=True)
os.makedirs(FAKE_IMAGES_PATH, exist_ok=True)
fake_image_index = 0
real_image_index = 0
# COPY SD AND GLIDE

already_seen_real_images = []

for diffused_path in DIFFUSED_PATHS:
    folders = [os.path.join(diffused_path, folder_name) for folder_name in os.listdir(diffused_path)]
    random.shuffle(folders)
    real_images = 0
    fake_images = 0
    for folder_path in folders:
        if "glide" in folder_path:
            method = "Glide"
        else:
            method = "Stable Diffusion"

        if "coco" in folder_path:
            dataset = "MSCOCO"
        else:
            dataset = "Wikimedia"
        
        if fake_images == 1500 and real_images == 1500:
            break
        paths = os.listdir(folder_path)
        random.shuffle(paths)
        ignored_images = 0
        fake_image_found = False
        for index, image_name in enumerate(paths):
            if "jpg" not in image_name:
                continue
            src_path = os.path.join(folder_path, image_name)
            if "real" in image_name:
                if src_path in already_seen_real_images or real_images == 1500:
                    continue
                try: 
                    img = cv2.imread(src_path)
                    if img is None:
                        print(src_path, real_image_index)
                        continue
                except:
                    print(src_path, real_image_index)
                    continue

                dst_path = os.path.join(REAL_IMAGES_PATH, str(real_image_index) + ".jpg")
                df = pd.concat([df, pd.DataFrame.from_records({"path": dst_path, "label": 0, "method": "N/A", "dataset": dataset}, index=[0])], ignore_index=True)
                already_seen_real_images.append(src_path)
                real_images += 1
                real_image_index += 1
            else:
                if fake_images == 1500 or fake_image_found:
                    continue
                dst_path = os.path.join(FAKE_IMAGES_PATH, str(fake_image_index) + ".jpg")
                df = pd.concat([df, pd.DataFrame.from_records({"path": dst_path, "label": 1, "method": method, "dataset": dataset}, index=[0])], ignore_index=True)
                fake_images += 1
                fake_image_index += 1
                fake_image_found = True
            shutil.copy(src_path, dst_path)



# COPY FLICKR
print(FLICKR_IMAGES, fake_image_index, real_image_index)
paths = os.listdir(FLICKR_IMAGES)
random.shuffle(paths)
for index, image_name in enumerate(paths):
    
    src_path = os.path.join(FLICKR_IMAGES, image_name)
    if "jpg" not in image_name:
        continue
    
    if index == 2500:
        break
    dst_path = os.path.join(REAL_IMAGES_PATH, str(real_image_index) + ".jpg")
    df = pd.concat([df, pd.DataFrame.from_records({"path": dst_path, "label": 0, "method": "N/A", "dataset": "Flickr50K"}, index=[0])], ignore_index=True)
    shutil.copy(src_path, dst_path)
    real_image_index += 1

    


# COPY GAN IMAGES

for folder in os.listdir(GAN_IMAGES):
    folder_path = os.path.join(GAN_IMAGES, folder)
    if not os.path.isdir(folder_path):
        continue
    splitted_folder = folder.split("_")
    method = splitted_folder[0]
    
    print(folder, fake_image_index, real_image_index)
    if len(splitted_folder) > 1:
        dataset = splitted_folder[1].split(".")[0]
    else:
        dataset = "CelebA"
    
    if not os.path.isdir(folder_path):
        continue
    paths = os.listdir(folder_path)
    random.shuffle(os.listdir(folder_path))
    for index, image_name in enumerate(paths):
        src_path = os.path.join(folder_path, image_name)
        _, ext =  os.path.splitext(src_path)
        if index == 500:
            break

        dst_path = os.path.join(folder_path, str(fake_image_index) + ext)
        df = pd.concat([df, pd.DataFrame.from_records({"path": dst_path, "label": 1, "method": method, "dataset": dataset}, index=[0])], ignore_index=True)
        shutil.copy(src_path, dst_path)
        fake_image_index += 1
        
print("DiffusionDB", fake_image_index, real_image_index)


# Load DiffusionDB
diffusion_dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k')
for index, data in enumerate(diffusion_dataset["train"]):
    if index == 1000:
        break

    image = data["image"]
    dst_path = os.path.join(FAKE_IMAGES_PATH, str(fake_image_index) + ".png")
    df = pd.concat([df, pd.DataFrame.from_records({"path": dst_path, "label": 1, "method": "Stable Diffusion", "dataset": "DiffusionDB"}, index=[0])], ignore_index=True)
    image.save(dst_path, "PNG")
    fake_image_index += 1


print(fake_image_index, real_image_index)

# Load DiffusionDB 2M
diffusion_dataset = load_dataset('poloclub/diffusiondb', "2m_random_1k")
for index, data in enumerate(diffusion_dataset["train"]):
    if index == 1000:
        break

    image = data["image"]
    dst_path = os.path.join(FAKE_IMAGES_PATH, str(fake_image_index) + ".png")
    df = pd.concat([df, pd.DataFrame.from_records({"path": dst_path, "label": 1, "method": "Stable Diffusion", "dataset": "DiffusionDB"}, index=[0])], ignore_index=True)
    image.save(dst_path, "PNG")
    fake_image_index += 1


print(fake_image_index, real_image_index)

df.to_csv("../datasets/custom_validation_gan/validation_set.csv", index=False)