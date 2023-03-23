#coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        #writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    #args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    #train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    #optimizer = torch.optim.SGD(model.parameters(),
                                #lr=args.learning_rate,
                                #momentum=0.9,
                                #weight_decay=args.weight_decay)
    #t_total = args.num_steps
    #if args.decay_type == "cosine":
       # scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    #else:
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    #if args.fp16:
        #model, optimizer = amp.initialize(models=model,
                                          #optimizers=optimizer,
                                          #opt_level=args.fp16_opt_level)
        #amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    #if args.local_rank != -1:
        #model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    #set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    #losses = AverageMeter()
    #global_step, best_acc = 0, 0
    '''
    while True:
        model.train()
        
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    '''



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

    
    #model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

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
    dataset = ImageNetKaggle("train", "", val_transform)
    dataloader = DataLoader(
                dataset,
                batch_size=64, # may need to reduce this depending on your GPU 
                num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                shuffle=False,
                drop_last=False,
                pin_memory=True
            )
    

    '''
    correct = 0
    total = 0
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
                scheduler.step()
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



def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
   
    config = CONFIGS[args.model_type]

    num_classes = 1000

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()