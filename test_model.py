






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

model.train().cuda()  # Needs CUDA, don't bother on CPUs
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


'''
with torch.no_grad():
    for x, y in tqdm(dataloader):
        logits = model(x.cuda()).logits
        _, preds = torch.max(logits, 1)
        # model predicts one of the 1000 ImageNet classes
        #preds = logits.argmax(-1).item()
        #_, preds = torch.max(outputs, 1)
        #loss = criteria(outputs, labels)
        #loss = criteria(logits, labels)
        print("printing")
        print(torch.max(logits, 1))
        print(logits)
        print(preds)
        print(y)'''
        
 
def train_model(model, criteria, optimizer, scheduler,    
                                      num_epochs=25, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, 15):#num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        # Each epoch has a training and validation phase
        for x, y in tqdm(dataloader):
            
            
            # Iterate over data.
            inputs = x
            labels = y
            inputs = inputs.to(device)
            labels = labels.to(device)

            #print(inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            #print(inputs.shape)
            # forward
            # track history if only in train

                #outputs = model(inputs)
            logits = model(inputs).logits
            _, preds = torch.max(logits, 1)
            # model predicts one of the 1000 ImageNet classes
            #preds = logits.argmax(-1).item()
            #_, preds = torch.max(outputs, 1)
            #loss = criteria(outputs, labels)
            loss = criteria(logits, labels)
            #print(loss)

                # backward + optimize only if in training phase

            loss.backward() #computes gradients
            optimizer.step() #updates weights

                # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            print(running_corrects)
            print(running_corrects)

            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / len(dataset)
            print(running_corrects / len(dataset))
            #epoch_acc = running_corrects/64

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                "train", epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
           

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

criteria = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# Number of epochs
eps=5

model = train_model(model, criteria, optimizer, scheduler, eps, 'cuda') #try to look at why val loss is lower


 

