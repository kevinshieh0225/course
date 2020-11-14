import os
import numpy as np
import re
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import gc

import PIL
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

import torchvision.models as models
import torch.nn as nn
from ipywidgets import IntProgress

import json

class Dataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, self.y[idx]
   

def load_mango_csv(csv_path='./train.csv'):
    label2idx = {
    '不良-乳汁吸附': 0,
    '不良-機械傷害': 1,
    '不良-炭疽病': 2,
    '不良-著色不佳': 3,
    '不良-黑斑病': 4
    }
    path = []
    box = []
    label = []
    subdir = csv_path.split('/')[-1].split('.')[0].capitalize() #[-1]意思是倒數最後一col，.capitalize()將首英文字母大寫，其他小寫
    # subdir : Train , Dev
    # 此時我需要的輸出格式為：
    # path: 照片路徑 : ./Train/img.jpg
    # label: 標籤  : [1,0,0,0,0]
    with open(csv_path, 'r', encoding='utf8') as f:        
        for line in f:
            clean_line = re.sub(',+\n', '', line).replace('\n', '').replace('\ufeff', '').split(',')
            curr_img_path = f'./{subdir}/{clean_line[0]}'
            column = 5
            curr_label = [0,0,0,0,0]
            while column <= len(clean_line):
              symptom = clean_line[column]
              if symptom in label2idx:
                curr_label[label2idx[symptom]] = 1
              column += 5
            path.append(curr_img_path)
            label.append(curr_label)
    print('data size: ')
    print(len(path), len(label))
    print(path[:6])
    print(label[:6])
    count = np.zeros(5)
    for check in label:
      count += np.array(check)
    print('不良-乳汁吸附：'+str(count[0])+' '+str(count[0]/len(label)))
    print('不良-機械傷害：'+str(count[1])+' '+str(count[1]/len(label)))
    print('不良-炭疽病：'+str(count[2])+' '+str(count[2]/len(label)))
    print('不良-著色不佳：'+str(count[3])+' '+str(count[3]/len(label)))
    print('不良-黑斑病：'+str(count[4])+' '+str(count[4]/len(label)))
    print()
    return path, label

def dataloader_prepare(csv_path):
    """aug:
        csv_path:讀入資料之位置
    """
    transform_flip = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(p = 1),
        torchvision.transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    transform_rotation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomRotation((10,15), resample=PIL.Image.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])



    path, label = load_mango_csv(csv_path)

    dataloader_flip = DataLoader( Dataset(path, label, transform_flip) , batch_size=1, shuffle=True)
    print(len(dataloader_flip))

    dataloader_origin = DataLoader( Dataset(path, label, transform) , batch_size=1, shuffle=True)
    print(len(dataloader_origin))

    dataloader_rotation = DataLoader( Dataset(path, label, transform_rotation) , batch_size=1, shuffle=True)
    print(len(dataloader_rotation))
    
    return dataloader_origin, dataloader_flip, dataloader_rotation

def feature_extractor(dataloader):
    print(len(dataloader))
    x = []
    y = []
    
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    i = 0
    dh = display('Start',display_id=True)
    for data in iter(dataloader):
        i += 1
        dh.update(i)
        with torch.no_grad():
            alex_output = alexnet(data[0]).tolist()

        with torch.no_grad():
            vgg_output = vgg16(data[0]).tolist()
        
        x.append(alex_output[0]+vgg_output[0])
        
        new_data = []
        for data_ in data[1]:
            data_ =  int(data_)
            new_data.append(data_)
        y.append(new_data)
        
    print(np.array(x).shape)
    print(np.array(y).shape)
    return x, y

def prepare_new_data(csv_path):
    """aug:
        csv_path:讀入資料之位置
    """
    dataloader_origin, dataloader_flip, dataloader_rotation = dataloader_prepare(csv_path)
    
    x_origin, y_origin = feature_extractor(dataloader_origin)
    x_flip, y_flip = feature_extractor(dataloader_flip)
    x_rotation, y_rotation = feature_extractor(dataloader_rotation)
    x = x_origin + x_flip + x_rotation
    y = y_origin + y_flip + y_rotation
    
    return x, y 

def write_data(file_name):
    data, label = prepare_new_data(csv_path='./train.csv')
    with open(f'{file_name}/data.txt', 'w') as outfile:
        json.dump(data, outfile)
    with open(f'{file_name}/label.txt', 'w') as outfile:
        json.dump(label, outfile)
    return data, label

def read_data(file_name):
    with open(f'{file_name}/data.txt') as jsonfile:
        data = json.load(jsonfile)
    with open(f'{file_name}/label.txt') as jsonfile:
        label = json.load(jsonfile)
    return data, label

def prepare_data(file_name):
    if not os.path.isdir(f'{file_name}/data.txt'):
        if not os.path.isdir(file_name):
            os.makedirs(file_name)
        data, label = write_data(file_name)
    else:
        data, label = read_data(file_name)
    
    return data, label