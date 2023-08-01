import glob
import pandas as pd
import cv2
import os
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from tqdm import tqdm

BASE_PATH = "datasets/laion"
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")
def check_paths(path, cleaned_train_paths, write_directly):

    try:
        image = cv2.imread(path)
        if image is None:
            return
    except:
        return
    if write_directly:
        if "fake" in path:
            label = 1
        else:
            label = 0

        with open("datasets/laion/training_set_3.csv", "a") as f:
            f.write("\n" + path + "," + str(label))
        f.close()
    else:
        cleaned_train_paths.append(path)

def step1(write_directly=False):
    train_paths = glob.glob(os.path.join(BASE_PATH, "real-images", "*/*"), recursive = True)
    train_paths.extend(glob.glob(os.path.join(BASE_PATH, "fake-images", "*/*"), recursive = True))
    
    mgr = Manager()
    cleaned_train_paths = mgr.list()
    with Pool(processes=250) as p:
            with tqdm(total=len(train_paths)) as pbar:
                for v in p.imap_unordered(partial(check_paths, cleaned_train_paths=cleaned_train_paths, write_directly=write_directly), train_paths):
                    pbar.update()

    if not write_directly:
        df = pd.DataFrame(columns=["path", "label"])
        for path in cleaned_train_paths:
            if "fake" in path:
                label = 1
            else:
                label = 0
            df = pd.concat([df, pd.DataFrame.from_records({"path": path, "label": label}, index=[0])], ignore_index=True)

        df.to_csv("datasets/laion/training_set.csv", index=False)

def step2(write_directly=True):
    print("Reading csv...")
    df = pd.read_csv("datasets/laion/training_set_3.csv")
    print("Readed.")
    found_ids = []
    new_df = pd.DataFrame(columns=["path", "label"])
    deleted = 0
    if write_directly:
        f = open("datasets/laion/training_set_3_cleaned.csv", "a")
    with tqdm(total=len(df)) as pbar:
        for index, row in df.iterrows():
            if index % 100000 == 0:
                print(deleted)
            '''
            filename = os.path.basename(row["path"])
            if filename in found_ids:
                continue
            
            found_ids.append(filename)
            if write_directly:
                f.write("\n" + row["path"] + "," + str(row["label"]))
            else:
                new_df = pd.concat([new_df, row], ignore_index=True)
            
            if not row["path"].endswith(ALLOWED_EXTENSIONS):
                deleted += 1
                continue
            '''
            if row["label"] == 0:
                if "part-000000" in row["path"]:
                    deleted += 1
                    pbar.update()
                    continue
            if write_directly:
                f.write("\n" + row["path"] + "," + str(row["label"]))
            
            pbar.update()
    if not write_directly:
        new_df.to_csv("datasets/laion/training_set_3_cleaned.csv", index=False)
    else:
        f.close()
        

#step1(write_directly=True)
step2(write_directly=True)