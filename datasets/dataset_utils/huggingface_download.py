from datasets import load_dataset
elsa_data = load_dataset("rs9000/ELSA1M_track1", split="train", streaming=True)

for index, sample in enumerate(elsa_data):
  image = sample.pop("image")
  metadata = sample
  if index % 100 == 0:
    print(metadata)