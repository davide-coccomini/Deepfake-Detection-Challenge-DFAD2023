import os
import shutil
import random

REAL_IMAGES_PATH = "datasets/validation_set/real_images"
FAKE_IMAGES_PATH = "datasets/validation_set/fake_images"
DIFFUSED_PATHS = ["../datasets/diffused_coco/test", "../datasets/diffused_wikipedia/test", "../datasets/glide_diffused_coco/test", "../datasets/glide_diffused_wikipedia/test"]
FLICKR_IMAGES = "../datasets/flickr/flickr30k_images/flickr30k_images"


random.seed(42)

df = pd.DataFrame(columns=["path", "label", "method", "dataset"])


# CREATE THE MAIN OUTPUT FOLDERS
os.makedirs(REAL_IMAGES_PATH)
os.makedirs(FAKE_IMAGES_PATH)
image_index = 0

# COPY SD AND GLIDE
diffused_folders_paths = []
for diffused_path in DIFFUSED_PATHS:
    folders = [os.path.join(SD_COCO_PATH, folder_name) for folder_name in os.listdir(diffused_path)]
    diffused_folders_paths.extend(folders)

diffused_folders_paths = random.shuffle(diffused_folders_paths)
already_seen_real_images = []
for folder_path in diffused_folders_paths:
    if "glide" in folder_path:
        method = "Glide"
    else:
        method = "Stable Diffusion"

    if "coco" in folder_path:
        dataset = "MSCOCO"
    else:
        dataset = "Wikimedia"

    for image_name in os.listdir(folder_path):
        if "jpg" not in image_name:
            continue
        src_path = os.path.join(folder_path, image_name)
        if "real" in image_name:
            if src_path in already_seen_real_images:
                continue
            dst_path = os.path.join(REAL_IMAGES_PATH, image_index)
            df.append({"path": dst_path, "label": 0, "method": "N/A", "dataset": dataset})
            already_seen_real_images.append(src_path)
        else:
            dst_path = os.path.join(FAKE_IMAGES_PATH, image_index)
            df.append({"path": dst_path, "label": 1, "method": method, "dataset": dataset})

        shutil.copy(src_path, dst_path)

        image_index += 1



# COPY FLICKR
for index, image_name in enumerate(random.shuffle(os.listdir(FLICKR_IMAGES))):
    src_path = os.path.join(FLICKR_IMAGES, image_name)
    if "jpg" not in image_name:
        continue
    
    if index > 1500:
        break

    dst_path = os.path.join(REAL_IMAGES_PATH, image_index)
    df.append({"path": dst_path, "label": 0, "method": "N/A", "dataset": "Flickr50K"})
    shutil.copy(src_path, dst_path)
    image_index += 1

    


    