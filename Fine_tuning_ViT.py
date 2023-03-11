import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="../Data_to_use")

splits = dataset["train"].train_test_split(test_size=0.1)

test_ds = splits["test"]
splits_2 = splits["train"].train_test_split(test_size=0.1)
train_ds = splits_2["train"]
val_ds = splits_2["test"]

print(train_ds[0]['label'])
