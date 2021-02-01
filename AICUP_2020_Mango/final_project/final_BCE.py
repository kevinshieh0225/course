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
from torch.nn import Linear, ReLU, Sigmoid, CrossEntropyLoss, BCELoss, Conv2d, MaxPool2d, Module ,Softmax 
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
    
# 1.2 不切芒果

def load_mango_csv(csv_path='../C2_TrainDev/train.csv'):
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
    folder = '../C2_TrainDev/'
    # subdir : Train , Dev
    # 此時我需要的輸出格式為：
    # path: 照片路徑 : ./Train/img.jpg
    # label: 標籤  : [1,0,0,0,0]
    resize_folder = f'{folder}resize/'
    if not os.path.isdir(resize_folder):
        os.makedirs(resize_folder)
    with open(csv_path, 'r', encoding='utf8') as f:        
        for line in tqdm(f):
            clean_line = re.sub(',+\n', '', line).replace('\n', '').replace('\ufeff', '').split(',')
            curr_img_path = f'{folder}{subdir}/{clean_line[0]}'
            new_img_path = f'{resize_folder}{subdir}{clean_line[0]}'
            column = 5
            curr_label = [0,0,0,0,0]
            while column <= len(clean_line):
              symptom = clean_line[column]
              if symptom in label2idx:
                curr_label[label2idx[symptom]] = 1
              column += 5
            if not os.path.isfile(curr_img_path):
                print(f'No file for path : {curr_img_path}')
                continue
            if not os.path.isfile(new_img_path):
                img = cv2.imread(curr_img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(new_img_path, img)
            path.append(new_img_path)
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


# 1.2 不切芒果
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
    print(f'len:{len(weights)}\n\n')
    num_samples = len(trainset['train']) if minority_num*10 > len(trainset['train']) else minority_num*10
    sampler = Data.WeightedRandomSampler(weights=weights,num_samples = num_samples,replacement = True)
    trainloader['train'] = DataLoader(trainset['train'], batch_size = batchsize, sampler=sampler)
    
    trainloader['valid'] = DataLoader(trainset['valid'], batch_size = batchsize)
    
    return trainloader

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

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
#     best_train_loss = 100
#     best_train_acc = 0
#     best_val_loss = 100
#     best_val_acc = 0
#     best_F1 = 0
#     last_epoch = 0

    history = {
        'accuracy':[],
        'val_accuracy':[],
        'loss':[],
        'val_loss':[],
        'precision':[],
        'recall':[],
        'f1':[]
    }
    
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('no gpu use')
    for epoch in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss,valid_loss = 0.0,0.0
        train_losses,valid_losses=[],[]
        y_pred_all = []
        y_true_all = []
        train_correct,val_correct,train_total,val_total=0,0,0,0
        confusion_stacks = []

        print(f'running epoch: {epoch}')
        #############################################################################################################
        #                                              train the model                                              #
        #############################################################################################################
        model.train()
        print('start train')
        for num, (data, target) in enumerate(train_loader):
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():#train_on_gpu
                data, target = data.cuda(), target.cuda()
            else:
                print('1')
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            target = target.unsqueeze(1).type_as(output)
            # calculate the batch loss
            loss = criterion(output, target)
            #calculate accuracy
            y_pred_tag = torch.round(torch.sigmoid(output))
            correct_results_sum = (y_pred_tag == target).sum().int().item()
            train_correct += correct_results_sum
            train_total += data.size(0)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_losses.append(loss.item()*data.size(0))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            
            if num%10 == 0 :
                print(f'{num}/{len(train_loader)}', end='\r')
        #############################################################################################################
        #                                            validate the model                                             #
        #############################################################################################################
        model.eval()
        print('start valid')
        for num, (data, target) in enumerate(valid_loader):
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():#train_on_gpu
                data, target = data.cuda(), target.cuda()
                
            # forward pass: compute predicted outputs by passing inputs to the model
            
            output = model(data)
            target = target.unsqueeze(1).type_as(output)
            # calculate the batch loss
            loss =criterion(output, target)
            #calculate accuracy
            y_pred_tag = torch.round(torch.sigmoid(output))
            correct_results_sum = (y_pred_tag == target).sum().int().item()
            val_correct += correct_results_sum
            val_total += data.size(0)
            # update validation loss
            valid_losses.append(loss.item()*data.size(0))
            
            #confusion matrix
            target = target.squeeze(1)
            stacked = torch.stack((target,y_pred_tag.t()[0]),dim=1)
            confusion_stacks += stacked.cpu().detach().tolist()
            if num%10 == 0 :
                print(f'{num}/{len(valid_loader)}', end='\r')
        #############################################################################################################
        #                                     print train/val/cmt epoch result                                      #
        #############################################################################################################
        # calculate average losses
        train_loss=np.average(train_losses)
        valid_loss=np.average(valid_losses)
        # calculate average accuracy
        train_acc=train_correct/train_total
        valid_acc=val_correct/val_total
        print(f'Training Loss: {train_loss:.3f} \tValidation Loss: {valid_loss:.3f}')
        print(f'Training Accuracy: {train_acc:.3f} \tValidation Accuracy: {valid_acc:.3f}')
        
        cmt = torch.zeros(2,2, dtype=torch.int64)
        for p in confusion_stacks:
            tl, pl = int(p[0]) , int(p[1])
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
#         if best_F1 >= F1:
#             trigger_times += 1
#             print(f'trigger times: {trigger_times}\n')
#             if trigger_times > patience:
#                 print(f'Early stopping at trigger times: {trigger_times}')
#                 print(f'Least Training Loss: {best_train_loss:.4f} \nLeast Validation Loss: {best_val_loss:.4f}')
#                 print(f'Best Training Accuracy: {best_train_acc:.4f} \nBest Validation Accuracy: {best_val_acc:.4f}')
#                 print(f'Best f1-score: {best_F1:.4f}')
#                 last_epoch = epoch
#                 break
#         else:
#             save_json = {
#                 'accuracy':[train_acc],
#                 'val_accuracy':[valid_acc],
#                 'loss':[train_loss],
#                 'val_loss':[valid_loss],
#                 'precision':[p],
#                 'recall':[r],
#                 'f1':[F1]
#             }
#             with open(f'./result/{name}/result.json', 'w') as json_file:
#                 json.dump(save_json, json_file)
                
#             trigger_times = 0
#             torch.save(model, f'./result/{name}/model.pt')
#             best_train_loss = train_loss
#             best_train_acc = train_acc
#             best_val_loss = valid_loss
#             best_val_acc = valid_acc
#             best_F1 = F1
    
        #############################################################################################################
        #                                                Draw picture                                               #
        #############################################################################################################
    
    with open(f'./result/{name}/result.json', 'w') as json_file:
        json.dump(history, json_file)
    
    x = np.arange(1,n_epochs+1,1)
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
    plt.plot(x, train_acc, label='training')
    plt.plot(x, val_acc, label='validation')
    plt.legend(loc='lower right')

    plt.subplot(2,2,2)
    plt.title(f"{name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, train_loss, label='training')
    plt.plot(x, val_loss, label='validation')
    plt.legend(loc='upper right')
    
    plt.subplot(2,2,3)
    plt.title(f"{name} F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("f1-score")
    plt.plot(x, f1, label='f1')
    plt.legend(loc='upper right')
    
    #save result
    plt.savefig(f'./result/{name}/plot.png')
        
    
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)
    # 1.1 load and cut：讀檔並切芒果
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
    path_train, box_train, label_train = cut.load_mango_csv(csv_path=f'{dir_path}/train.csv',dir_path = dir_path)
    path_dev, box_dev, label_dev = cut.load_mango_csv(csv_path=f'{dir_path}/dev.csv',dir_path = dir_path)
    print('Start Cutting')
    cut_img_train = cut.cut_mango(path_train, dest=dest, isCut=True, box=box_train)
    cut_img_dev = cut.cut_mango(path_dev, dest=dest, isCut=True, box=box_dev)
    print('Finish Cutting')

    label2idx = {
        '不良-乳汁吸附': 0,
        '不良-機械傷害': 1,
        '不良-炭疽病': 2,
        '不良-著色不佳': 3,
        '不良-黑斑病': 4
    }
    label_index = label_train + label_dev
    label = []
    for check in label_index:
        new = [0,0,0,0,0]
        for check_ in check:
            new[label2idx[check_]] = 1
        label.append(new)
    
    path = cut_img_train + cut_img_dev
    
    # 1.2 不切芒果
#     path_train , label_train = load_mango_csv()
#     path_dev , label_dev = load_mango_csv(csv_path='../C2_TrainDev/dev.csv')

#     path = path_train + path_dev
#     label = label_train + label_dev
#     print(len(path))
#     print(len(label))
#     print(path[:5])
#     print(label[:5])
    
    # 2. binary classification and balanced：把五類分開來，變成（1,0）問題
    ## 使multi-label 成為 five multi-class problem
    ## output : 分成五類的 dataloader

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    batchsize = 32

    # 2. Train fine-tune model
    ## 使用wide resnet50_2來進行pretrained
    ## output : 五個ft wr

    # Class iter ：wide_resnet50_2 finetune
    
    
    for class_index in range(5):
        ##############################################自己取名########################################################
        name = f'model_ft_wide_resnet50_class_{class_index}_opt_ALL_BCElogit'
        ##############################################自己取名########################################################
        batchsize = 32
        wide_resnet_dataloader = images_aug_balanced(path , label, class_index, transform, batchsize)
        for stage in ['train','valid']:
            print(f'len of {stage}:{len(wide_resnet_dataloader[stage])} from {len(wide_resnet_dataloader[stage])*batchsize}')


        if not os.path.isdir(f'./result/{name}/'):
            os.makedirs(f'./result/{name}/')
            
        #############################################################################################################
        #                                               ft model setting                                            #
        #############################################################################################################
        model_ft = models.wide_resnet50_2(pretrained=True)
#         for param in model_ft.parameters():
#             param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc  = nn.Sequential(
            nn.Linear(num_ftrs, 1),
        )
        model_ft = model_ft.to(device)# 放入裝置

        n_epochs = 10
        optimizer = torch.optim.Adam([
            {'params':model_ft.parameters()}
        ], lr=0.001)
        
        #############################################################################################################
        #                                               ft model setting                                            #
        #############################################################################################################
        
        
        criterion = nn.BCEWithLogitsLoss()
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
    
