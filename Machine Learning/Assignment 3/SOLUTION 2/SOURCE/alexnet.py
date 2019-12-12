# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 04:50:31 2019

@author: Ashima
"""
####Testing AlexNet Model of PyTorch Library
### https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
import os
import time
import torch
import data
import torch.nn as nn
import config
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import numpy as np

## Load the model 
alexnet = models.alexnet(pretrained=True)
print(alexnet)
alexnet.eval()

from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])


data = data.Data()
#Read train data
 
train_dataloader = data.read(config.TRAIN_FILE)
print("Train data read successfully")

i = 0
for data in train_dataloader:
    i += 1
    X, Y = data
    out = alexnet(X)
    print(out.shape)
    if i == 1:
        break
#
## Number of filters in the bottleneck layer
#num_ftrs = model_conv.classifier[6].in_features
#num_out = model_conv.classifier[6].out_features
## convert all the layers to list and remove the last one
#features = list(model_conv.classifier.children())[:-1]
#
##num_fc = model_conv.features[0].
#net = models.AlexNet()
#print(net)
#
#n_class = 10
### Lets freeze the first few layers. This is done in two stages 
## Stage-1 Freezing all the layers 
#freeze_layers = True
#if freeze_layers:
#  for i, param in model_conv.named_parameters():
#    param.requires_grad = False
##
### Since imagenet as 1000 classes , We need to change our last layer according to the number of classes we have,
##num_ftrs = model_conv.fc.in_features
##model_conv.fc = nn.Linear(num_ftrs, n_class)
##
### Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
##ct = []
##for name, child in model_conv.named_children():
##    if "Conv2d_4a_3x3" in ct:
##        for params in child.parameters():
##            params.requires_grad = True
##    ct.append(name)
##    
## To view which layers are freeze and which layers are not freezed:
#for name, child in model_conv.named_children():
#  for name_2, params in child.named_parameters():
#    print(name, name_2)
#
##model_conv = models.alexnet(pretrained = 'imagenet')
#    
#    
    
# =============================================================================
# ## Loading the dataloaders -- Make sure that the data is saved in following way
# """
# data/
#   - train/
#       - class_1 folder/
#           - img1.png
#           - img2.png
#       - class_2 folder/
#       .....
#       - class_n folder/
#   - val/
#       - class_1 folder/
#       - class_2 folder/
#       ......
#       - class_n folder/
# """
# 
# data_dir = "data/"
# input_shape = 299
# batch_size = 32
# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]
# scale = 360
# input_shape = 299 
# use_parallel = True
# use_gpu = True
# epochs = 100
# 
# data_transforms = {
#         'train': transforms.Compose([
#         transforms.Resize(scale),
#         transforms.RandomResizedCrop(input_shape),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(degrees=90),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)]),
#         'val': transforms.Compose([
#         transforms.Resize(scale),
#         transforms.CenterCrop(input_shape),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)]),}
# 
# 
# 
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                       data_transforms[x]) for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
#                                          shuffle=True, num_workers=4) for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes
# 
# if use_parallel:
#     print("[Using all the available GPUs]")
#     model_conv = nn.DataParallel(model_conv, device_ids=[0, 1])
# 
# print("[Using CrossEntropyLoss...]")
# criterion = nn.CrossEntropyLoss()
# 
# print("[Using small learning rate with momentum...]")
# optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr=0.001, momentum=0.9)
# 
# print("[Creating Learning rate scheduler...]")
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
# 
# print("[Training the model begun ....]")
# # train_model function is here: https://github.com/Prakashvanapalli/pytorch_classifiers/blob/master/tars/tars_training.py
# model_ft = train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, use_gpu,
#                      num_epochs=epochs)
# =============================================================================
