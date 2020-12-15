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

import torch.utils.data as Data
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

import torchvision.models as models
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module
from torch.optim import Adam
from tqdm.notebook import tqdm as tqdm
from ipywidgets import IntProgress

import cut
import json

class myDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        if img is None:
            print('Not found img : ', self.x[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, self.y[idx]
    def target(self, slice_arr):
        tar = []
        for idx in slice_arr:
            tar.append(self.y[idx])
        return tar
    
    
def images_aug_balanced(cut_img , label_index ,label,transform,batchsize):
    
    data_x = []
    data_y = []
    minority_num = 0
    majority_num = 0
    for img_path , label_indices in zip(cut_img , label_index):
        if label_indices[label] == 1:
            data_y.append(1) 
            minority_num += 1
        else: 
            data_y.append(0)
            majority_num += 1
        data_x.append(img_path)
    
    dataset = myDataset(data_x, data_y, transform)
    
    #切分80%當作訓練集、30%當作驗證集
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    trainset = {}
    trainset['train'],trainset['valid'] = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    minority_weight = 1/minority_num
    majority_weight = 1/majority_num  if minority_num*4 > majority_num else 3/majority_num
    print(f'1:{minority_weight} , 0:{majority_weight}')
    sample_weights = np.array([majority_weight, minority_weight])
    
    trainloader = {}
    labels = trainset['train'].dataset.target(trainset['train'].indices)
    labels = list(map(int, labels))
    weights = sample_weights[labels]
    print(f'train stage balanced')
    print(f'weights:{weights[:50]}')
    print(f'len:{len(weights)}\n\n')
    num_samples = len(trainset['train']) if minority_num*4 > majority_num else minority_num*10
    sampler = Data.WeightedRandomSampler(weights=weights,num_samples = num_samples)
    trainloader['train'] = DataLoader(trainset['train'], batch_size = batchsize, sampler=sampler)
    
    trainloader['valid'] = DataLoader(trainset['valid'], batch_size = batchsize)
    
    return trainloader
    
def train(model,name,n_epochs,train_loader,valid_loader,optimizer,criterion,batch_size,patience):
    """
    intro:
        每次epoch都 train the model , validate the model
        並計算Early Stopping
        印出 train_loss , train_acc , val_loss , val_acc
        回傳 model
    aug:
        model,n_epochs,train_loader,valid_loader,optimizer,criterion,batch_size
    output:
        model
    """
    print(f'Start to run {name}')
    if os.path.isfile(f'./result/{name}/result.json'):
        with open(f'./result/{name}/result.json', 'r') as read_file:
            history = json.load(read_file)
        best_train_loss = history['loss']
        best_train_acc = history['accuracy']
        best_val_loss = history['val_loss']
        best_val_acc = history['val_accuracy']
        best_F1 = history['f1']
    else:
        history = {
            'accuracy':[0],
            'val_accuracy':[0],
            'loss':[100],
            'val_loss':[100],
            'precision':[0],
            'recall':[0],
            'f1':[0]
        }
        best_train_loss = 100
        best_train_acc = 0
        best_val_loss = 100
        best_val_acc = 0
        best_F1 = 0
    last_epoch = 0

    
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('no gpu use')
    for epoch in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss,valid_loss = 0.0,0.0
        train_losses,valid_losses=[],[]
        train_correct,val_correct,train_total,val_total=0,0,0,0
        confusion_stacks = []

        print('running epoch: {}'.format(epoch))
        #############################################################################################################
        #                                              train the model                                              #
        #############################################################################################################
        model.train()
        for data, target in tqdm(train_loader):
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():#train_on_gpu
                data, target = data.cuda(), target.cuda()
            else:
                print('1')
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            #calculate accuracy
            pred = output.data.max(dim = 1, keepdim = True)[1]
            train_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            train_total += data.size(0)
            
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_losses.append(loss.item()*data.size(0))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
        
        #############################################################################################################
        #                                            validate the model                                             #
        #############################################################################################################
        model.eval()
        for data, target in tqdm(valid_loader):
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():#train_on_gpu
                data, target = data.cuda(), target.cuda()
                
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss =criterion(output, target)
            #calculate accuracy
            pred = output.data.max(dim = 1, keepdim = True)[1]
            val_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            val_total += data.size(0)
            # update validation loss
            valid_losses.append(loss.item()*data.size(0))
            
            #confusion matrix
            stacked = torch.stack((target,pred.t()[0]),dim=1)
            confusion_stacks += stacked.cpu().detach().tolist()
        
        #############################################################################################################
        #                                     print train/val/cmt epoch result                                      #
        #############################################################################################################
        # calculate average losses
        train_loss=np.average(train_losses)
        valid_loss=np.average(valid_losses)
        # calculate average accuracy
        train_acc=train_correct/train_total
        valid_acc=val_correct/val_total
        print(f'\tTraining Loss: {train_loss:.3f} \tValidation Loss: {valid_loss:.3f}')
        print(f'\tTraining Accuracy: {train_acc:.3f} \tValidation Accuracy: {valid_acc:.3f}')
        
        cmt = torch.zeros(2,2, dtype=torch.int64)
        for p in confusion_stacks:
            tl, pl = p
            cmt[tl, pl] = cmt[tl, pl] + 1
        print(f'cmt\tpred:0\tpred:1\nlabel:0\t{cmt[0,0]}\t{cmt[0,1]}\nlabel:1\t{cmt[1,0]}\t{cmt[1,1]}\n')
        
        TP = cmt[1,1].item()
        FP = cmt[1,0].item()
        FN = cmt[0,1].item()
        TN = cmt[1,0].item()
        
        p = np.float64(TP / (TP + FP))
        r = np.float64(TP / (TP + FN))
        F1 = np.float64(2 * r * p / (r + p))
        
        print(f'precision = {p}\trecall = {r}\tF1 = {F1}\n\n')
        
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(valid_acc)
        history['loss'].append(train_loss)
        history['val_loss'].append(valid_loss)
        history['precision'].append(p)
        history['recall'].append(r)
        history['f1'].append(F1)
        
        #############################################################################################################
        #                                              Early Stopping                                               #
        #############################################################################################################
        if best_F1 >= F1:
            trigger_times += 1
            print(f'trigger times: {trigger_times}\n')
            if trigger_times > patience:
                print(f'Early stopping at trigger times: {trigger_times}')
                print(f'Least Training Loss: {best_train_loss:.4f} \nLeast Validation Loss: {best_val_loss:.4f}')
                print(f'Best Training Accuracy: {best_train_acc:.4f} \nBest Validation Accuracy: {best_val_acc:.4f}')
                print(f'Best f1-score: {best_F1:.4f}')
                last_epoch = epoch
                break
        else:
            save_json = {
                'accuracy':[train_acc],
                'val_accuracy':[valid_acc],
                'loss':[train_loss],
                'val_loss':[valid_loss],
                'precision':[p],
                'recall':[r],
                'f1':[F1]
            }
            with open(f'./result/{name}/result.json', 'w') as json_file:
                json.dump(save_json, json_file)
                
            trigger_times = 0
            torch.save(model, f'./result/{name}/model.pt')
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_val_loss = valid_loss
            best_val_acc = valid_acc
            best_F1 = F1
    
        #############################################################################################################
        #                                                Draw picture                                               #
        #############################################################################################################
        
    x = np.arange(1,last_epoch+1,1)
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    train_loss = history['loss']
    val_loss = history['val_loss']
    f1 = history['f1']
    
    fig = plt.figure(figsize=(12,4))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    plt.subplot(2,2,1)
    plt.title(f"{name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(x, train_acc[1:], label='training')
    plt.plot(x, val_acc[1:], label='validation')
    plt.legend(loc='lower right')

    plt.subplot(2,2,2)
    plt.title(f"{name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, train_loss[1:], label='training')
    plt.plot(x, val_loss[1:], label='validation')
    plt.legend(loc='upper right')
    
    plt.subplot(2,2,3)
    plt.title(f"{name} F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("f1-score")
    plt.plot(x, f1[1:], label='f1')
    plt.legend(loc='upper right')
    
    #save result
    plt.savefig(f'./result/{name}/plot.png')
        
    
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)
    # 1. load and cut：讀檔並切芒果
    ## output : cut_img, label_index

    dir_path = "./../C2_TrainDev"
    dest = "./../C2_TrainDev_After_Cut"

    subdir = ['Dev', 'Train']
    for sub in subdir:
        new_dir = os.path.join(dest, sub)
    if not os.path.isdir(new_dir):
      try:
          os.makedirs(new_dir)
      except OSError as e:
          print(e)
      else:
          print(f"Successfully created the directory {new_dir}")
    path, box, label = cut.load_mango_csv(csv_path=f'{dir_path}/train.csv',dir_path = dir_path)
    print('Start Cutting')
    cut_img = cut.cut_mango(path, dest=dest, isCut=True, box=box)
    print('Finish Cutting')

    print(path[:10])
    print(box[:10])
    print(label[:10])
    print(cut_img[:3])
    print(len(cut_img))

    label2idx = {
        '不良-乳汁吸附': 0,
        '不良-機械傷害': 1,
        '不良-炭疽病': 2,
        '不良-著色不佳': 3,
        '不良-黑斑病': 4
    }
    label_index = []
    for check in label:
        new = [0,0,0,0,0]
        for check_ in check:
            new[label2idx[check_]] = 1
        label_index.append(new)
    print(label_index[:10])

    # 2. binary classification and balanced：把五類分開來，變成（1,0）問題
    ## 使multi-label 成為 five multi-class problem
    ## output : 分成五類的 dataloader

    transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    batchsize = 32
#     dataloader_0 = images_aug_balanced(cut_img , label_index, 0, transform, batchsize)
#     dataloader_1 = images_aug_balanced(cut_img , label_index, 1, transform, batchsize)
#     dataloader_2 = images_aug_balanced(cut_img , label_index, 2, transform, batchsize)
#     dataloader_3 = images_aug_balanced(cut_img , label_index, 3, transform, batchsize)
#     dataloader_4 = images_aug_balanced(cut_img , label_index, 4, transform, batchsize)

#     for stage in ['train' , 'valid']:
#         print(f'len of 0 {stage}:{len(dataloader_0[stage])} from {len(dataloader_0[stage])*batchsize}')
#         print(f'len of 1 {stage}:{len(dataloader_1[stage])} from {len(dataloader_1[stage])*batchsize}')
#         print(f'len of 2 {stage}:{len(dataloader_2[stage])} from {len(dataloader_2[stage])*batchsize}')
#         print(f'len of 3 {stage}:{len(dataloader_3[stage])} from {len(dataloader_3[stage])*batchsize}')
#         print(f'len of 4 {stage}:{len(dataloader_4[stage])} from {len(dataloader_4[stage])*batchsize}\n')



    # 2. Train fine-tune model
    ## 使用wide resnet50_2來進行pretrained
    ## output : 五個ft wr

    # Class iter ：wide_resnet50_2 finetune
#     

#     name = 'model_ft_wide_resnet50_class_0'
#     name = 'model_ft_wide_resnet50_class_1'
#     name = 'model_ft_wide_resnet50_class_2'
#     name = 'model_ft_wide_resnet50_class_3'
    name = 'model_ft_wide_resnet50_class_4'
    name_iter = ['model_ft_wide_resnet50_class_0', 'model_ft_wide_resnet50_class_1','model_ft_wide_resnet50_class_2','model_ft_wide_resnet50_class_3','model_ft_wide_resnet50_class_4',]
    
    for class_index, name in enumerate(name_iter):
        batchsize = 32
        wide_resnet_dataloader = images_aug_balanced(cut_img , label_index, class_index, transform, batchsize)
        for stage in ['train','valid']:
            print(f'len of {stage}:{len(wide_resnet_dataloader[stage])} from {len(wide_resnet_dataloader[stage])*batchsize}')


        if not os.path.isdir(f'./result/{name}/'):
            os.makedirs(f'./result/{name}/')
        model_ft = models.wide_resnet50_2(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,2)
        model_ft = model_ft.to(device)# 放入裝置

        n_epochs = 100
        optimizer = torch.optim.Adam([
            {'params':model_ft.parameters()}
        ], lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        patience = 3
        train(model_ft,
               name,
               n_epochs,
               wide_resnet_dataloader['train'],
               wide_resnet_dataloader['valid'],
               optimizer,
               criterion,
               batchsize,
               patience)
    
