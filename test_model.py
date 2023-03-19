






#!pip install transformers
import transformers
from transformers import AutoImageProcessor, ViTForImageClassification
import torch
import datasets
from datasets import load_dataset

import time
import json
import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, optim
from torch.optim import lr_scheduler


from torchvision import transforms

from torchvision import models
from torchvision import datasets


import kaggle
import os
from torch.utils.data import Dataset
from PIL import Image
import json
class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join("/shared/local_scratch/gpu1/c_mtmaxwel/ENETER/ILSRVC", "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join("/shared/local_scratch/gpu1/c_mtmaxwel/ENETER/ILSRVC", "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join("/shared/local_scratch/gpu1/c_mtmaxwel/ENETER/", "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]


from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision
from tqdm import tqdm
import transformers
from transformers import AutoImageProcessor, ViTForImageClassification
import torch

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

model.eval().cuda()  # Needs CUDA, don't bother on CPUs
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
val_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
dataset = ImageNetKaggle("", "val", val_transform)
dataloader = DataLoader(
            dataset,
            batch_size=64, # may need to reduce this depending on your GPU 
            num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
correct = 0
total = 0



with torch.no_grad():
    for x, y in tqdm(dataloader):
        output = model(x.cuda())
        _, preds = torch.max(output, 1)
        print("printing")
        print(output)
        print(preds)
        print(y)

