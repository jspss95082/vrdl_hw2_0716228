from __future__ import print_function
import numpy as np
import torchvision
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import antialiased_cnns
import os
import h5py
from engine import train_one_epoch, evaluate
import utils

f = h5py.File('train\digitStruct.mat', 'r')


def get_img_name(f, idx=0, names=f['digitStruct/name']):
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return(img_name)


bbox_prop = ['height', 'left', 'top', 'width', 'label']


def get_img_boxes(f, idx=0, bboxs=f['digitStruct/bbox']):
    meta = {key: [] for key in bbox_prop}

    box = f[bboxs[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))

    return meta


class SVHNDataset(torch.utils.data.Dataset):
    def __init__(self, mat_path, transforms=None):
        imgs = []
        bbox = []
        for i in range(33402):
            imgs.append('train/' + get_img_name(f, i))
            bbox.append(get_img_boxes(f, i))

        self.transforms = transforms
        self.imgs = imgs
        self.bbox = bbox

    def __getitem__(self, idx):

        fn = self.imgs[idx]
        img = Image.open(fn).convert("RGB")
        image_id = torch.tensor([idx])
        bbox_t = self.bbox[idx]
        boxes = []
        labels = []
        area = []
        for i in range(len(bbox_t['label'])):
            xmin = bbox_t['left'][i]
            xmax = bbox_t['left'][i] + bbox_t['width'][i]
            ymin = bbox_t['top'][i]
            ymax = bbox_t['top'][i] + bbox_t['height'][i]

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(bbox_t['label'][i])
            area.append(bbox_t['width'][i] * bbox_t['height'][i])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area)
        iscrowd = torch.zeros((len(bbox_t['label']),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        metavar="N",
        help="input patience for training (default: 10)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    load_model = input('Do you need to load model ? y/n :')
    if load_model[0] == 'y' or load_model[0] == 'Y':
        model_name = input('Enter your model filename : ')
        model = torch.load(model_name)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False,
            progress=True,
            num_classes=11,
            pretrained_backbone=True
        )
    model.to(device)

    # augmentation

    transform_blur_set = [
        transforms.GaussianBlur(9, sigma=(1, 7)),
        transforms.GaussianBlur(7, sigma=(1, 7)),
        transforms.GaussianBlur(5, sigma=(1, 7)),
        transforms.GaussianBlur(3, sigma=(1, 7))
    ]
    transform_erase_set = [transforms.RandomErasing(
        p=1, ratio=(1, 1), scale=(0.001, 0.001))]*30
    transform_blur = [transforms.RandomChoice(transform_blur_set)]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ColorJitter(
                brightness=(0.9, 1.1),
                contrast=(0.9, 1.1),
                saturation=(0.9, 1.1),
            ),
            transforms.RandomApply(transform_blur, p=0.75),
            # transforms.RandomApply(transform_erase_set, p=1),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),


        ]
    )
    train_data = SVHNDataset(
        mat_path="train\digitStruct.mat", transforms=transform)

    train_loader = DataLoader(
        train_data,
        args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=my_collate
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2)

    print('Train start')
    # train
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader,
                        device, epoch, print_freq=50)
        scheduler.step()
        print('')
        print('==================================================')
        print('')
        torch.save(model, 'model.pt')

    print("That's it!")


if __name__ == "__main__":
    main()
