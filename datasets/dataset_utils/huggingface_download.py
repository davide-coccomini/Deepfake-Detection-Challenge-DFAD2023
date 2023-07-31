from datasets import load_dataset
import os

BASE_PATH = "../laion"

elsa_data = load_dataset("rs9000/ELSA1M_track1", split="train", streaming=True)

for index, sample in enumerate(elsa_data):
  image = sample.pop("image")
  metadata = sample
  if "fake" in metadata["filepath"]:
    image_path = os.path.join(BASE_PATH, metadata["filepath"])
    os.makedirs(os.path.dirname(image_path), exist_ok = True)
    image.save(image_path, "PNG")

