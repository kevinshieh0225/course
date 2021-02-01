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

from matplotlib.font_manager import FontProperties
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

def load_mango_csv(csv_path):

  path, box, label = [], [], []
  subdir = csv_path.split('/')[-1].split('.')[0].capitalize() # get subdir = Train or Dev
  with open(csv_path, 'r', encoding='utf8') as f:
    for line in f:
      clean_line = re.sub(',+\n', '', line).replace('\n', '').replace('\ufeff', '').split(',')
      curr_img_path = f'./../C2_TrainDev/{subdir}/{clean_line[0]}'
      curr_info = np.array(clean_line[1:]).reshape(-1, 5)
      curr_box = curr_info[:, :-1].astype('float16').tolist()
      curr_label = curr_info[:, -1].tolist()
      path.append(curr_img_path)
      box.append(curr_box)
      label.append(curr_label)

  return path, box, label

def cut_image_label(path, box, label, isTrain):

    label2idx = {
    '不良-乳汁吸附': 0,
    '不良-機械傷害': 1,
    '不良-炭疽病': 2,
    '不良-著色不佳': 3,
    '不良-黑斑病': 4
    }

    box_path, box_label = [], []
    for choose_idx in tqdm(range(len(path))):
        try:
            curr_path, curr_box, curr_label = path[choose_idx], box[choose_idx], label[choose_idx]
            for i in range(len(curr_box)):
                if isTrain == True:
                    file_path = f'./../C2_TrainDev_Label/Train/{curr_path[-9:-4]}_{i+1}.jpg'
                else:
                    file_path = f'./../C2_TrainDev_Label/Dev/{curr_path[-9:-4]}_{i+1}.jpg'

                if not os.path.isfile(file_path):
                    x, y, w, h = int(curr_box[i][0]), int(curr_box[i][1]), int(curr_box[i][2]), int(curr_box[i][3])
                    # label_img = cv2.rectangle(curr_img.copy(), (x,y), (x+w,y+h), (0,0,255), 3)
                    curr_img = cv2.imread(f'{curr_path}')
#                     cut_img = curr_img[y-10:y+h+10, x-10:x+w+10]
                    cut_img = curr_img[y:y+h, x:x+w]
                    img = cv2.resize(cut_img.copy(), (224, 224), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f'{file_path}', img) # save image
                    print('happen')

                box_path.append(f'{file_path}') # save path of image
                # box_label.append(curr_label[i]) 
                binary_label = [0, 0, 0, 0]
                binary_label.insert(label2idx[curr_label[i]], 1)
                box_label.append(binary_label) # save label of image

        except:
            print(f'error pic: {curr_path}')
            continue

    print(f'path len:{len(box_path)}\t{box_path[:5]}')    
    return box_path, box_label

# curr_img = cv2.imread('./../C2_TrainDev/Train/32391.jpg')
# plt.imshow(curr_img)
# print(curr_img.shape)

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
        train_correct,val_correct,train_total,val_total=0,0,0,0
        confusion_stacks = []

        print('running epoch: {}'.format(epoch))
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
            tl, pl = p
            cmt[tl, pl] = cmt[tl, pl] + 1
        print(f'cmt\tpred:0\tpred:1\nlabel:0\t{cmt[0,0]}\t{cmt[0,1]}\nlabel:1\t{cmt[1,0]}\t{cmt[1,1]}\n')
        
        TP = cmt[1,1].item()
        FN = cmt[1,0].item()
        FP = cmt[0,1].item()
        TN = cmt[0,0].item()
        
        p = np.float64(TP / (TP + FP)) if TP + FP != 0 else 0
        r = np.float64(TP / (TP + FN)) if TP + FN != 0 else 0
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
    
    with open(f'./result_disease/{name}/result.json', 'w') as json_file:
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
    plt.savefig(f'./result_disease/{name}/plot.png') 
    torch.save(model, f'./result_disease/{name}/model.pt')

def images_aug_balanced(cut_img, label_index, label, batchsize, istrain = True):
    
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
    
    
    if istrain == True:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = myDataset(data_x, data_y, transform)
        minority_weight = 1/minority_num
        majority_weight = 1/majority_num  if minority_num * 4 > majority_num else 3 / majority_num
        print(f'1:{minority_weight} , 0:{majority_weight}')
        sample_weights = np.array([majority_weight, minority_weight])
        weights = sample_weights[data_y]
        num_samples = len(dataset) if minority_num * 10 > len(dataset) else minority_num * 10
        sampler = Data.WeightedRandomSampler(weights=weights,num_samples = num_samples,replacement = True)
        trainloader = DataLoader(dataset, batch_size = batchsize, sampler=sampler)   
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = myDataset(data_x, data_y, transform)
        trainloader = DataLoader(dataset, batch_size = batchsize)
    
    return trainloader

def predict_real_f1(model, model_ft_revise , class_idx, path_dev, label_dev, box_path_dev):
    """
    intro:
    output:
        model
    """
    pred_label = np.zeros(len(path_dev))
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    model.eval()
    model_ft_revise.eval()
    if class_idx == 2 or class_idx == 3:
        for path in box_path_dev:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = transform(img).cuda()
            img = img.unsqueeze(0)
            with torch.no_grad(): 
                output=model(img)
            pred = output.data.max(dim = 1, keepdim = True)[1]
            #######尋找對應的path
            if int(pred) == 1:
                img_index = path.split('/')[-1].split('_')[0]
                img_path = f'./../C2_TrainDev/Dev/{img_index}.jpg'
                print(f'try {path_dev.index(img_path)}', end='\r')
                pred_label[path_dev.index(img_path)] = 1
    else:
        for path in box_path_dev:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = transform(img).cuda()
            img = img.unsqueeze(0)
            with torch.no_grad(): 
                output=model_ft_revise(img)
            pred_revise = output.data.max(dim = 1, keepdim = True)[1]
            if int(pred_revise) == 1:
                continue
            
            with torch.no_grad(): 
                output=model(img)
            pred = output.data.max(dim = 1, keepdim = True)[1]
            #######尋找對應的path
            if int(pred) == 1:
                img_index = path.split('/')[-1].split('_')[0]
                img_path = f'./../C2_TrainDev/Dev/{img_index}.jpg'
                print(f'try {path_dev.index(img_path)}', end='\r')
                pred_label[path_dev.index(img_path)] = 1
    
    
    ############# preprocess label_dev  ################
    label_dev_by_mango = []
    label2idx = {
        '不良-乳汁吸附': 0,
        '不良-機械傷害': 1,
        '不良-炭疽病': 2,
        '不良-著色不佳': 3,
        '不良-黑斑病': 4
    }
    for label_dev_ in label_dev:
        curr_label = [0,0,0,0,0]
        for symptom in label_dev_:
            if symptom in label2idx:
                curr_label[label2idx[symptom]] = 1
        label_dev_by_mango.append(curr_label)
        
        
    ############# evaluate cmt ################
    print(f'\n\n第{class_idx}類別結果\n')
    cmt = torch.zeros(2,2)
    for t , p in zip(label_dev_by_mango , pred_label):
        tl, pl = t[class_idx] , int(p)
        cmt[tl, pl] = cmt[tl, pl] + 1
    print(f'cmt\tpred:0\tpred:1\nlabel:0\t{cmt[0,0]}\t{cmt[0,1]}\nlabel:1\t{cmt[1,0]}\t{cmt[1,1]}\n')

    TP = cmt[1,1]
    FN = cmt[1,0]
    FP = cmt[0,1]
    TN = cmt[0,0]

    p = np.float64(TP / (TP + FP)) if TP + FP != 0 else 0
    r = np.float64(TP / (TP + FN)) if TP + FN != 0 else 0
    F1 = np.float64(2 * r * p / (r + p)) if r+p != 0 else 0

    print(f'precision = {p}\trecall = {r}\tF1 = {F1}\n\n')
    return p,r,F1

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)

    path_train, box_train, label_train = load_mango_csv('./../C2_TrainDev/train.csv')
    box_path_train, box_label_train = cut_image_label(path_train, box_train, label_train, isTrain=True)
    print(f'path_train len:{len(path_train)}')
    print(f'box_path_train len:{len(box_path_train)}')
    
    path_dev, box_dev, label_dev = load_mango_csv('./../C2_TrainDev/dev.csv')
    box_path_dev, box_label_dev = cut_image_label(path_dev, box_dev, label_dev, isTrain=False)

    print(f'path_dev len:{len(path_dev)}')
    print(f'box_path_dev len:{len(box_path_dev)}')

    # 2. binary classification and balanced：把五類分開來，變成（1,0）問題
    ## 使multi-label 成為 five multi-class problem
    ## output : 分成五類的 dataloader

    # 2. Train fine-tune model
    ## 使用wide resnet50_2來進行pretrained
    ## output : 五個ft wr

    # Class iter ：wide_resnet50_2 finetune


#     for class_index in range(5):
#         ##############################################自己取名########################################################
#         name = f'vgg16_ft_diseasepart_{class_index}'
#         ##############################################自己取名########################################################
#         batchsize = 48
#         train_dataloader = images_aug_balanced(box_path_train, box_label_train, class_index, batchsize, istrain = True)
#         dev_dataloader = images_aug_balanced(box_path_dev, box_label_dev, class_index, batchsize, istrain = False)
#         print(f'len of train:{len(train_dataloader)} from {len(train_dataloader)*batchsize}')
#         print(f'len of dev:{len(dev_dataloader)} from {len(dev_dataloader)*batchsize}')

#         if not os.path.isdir(f'./result_disease/{name}/'):
#             os.makedirs(f'./result_disease/{name}/')

#         #############################################################################################################
#         #    ft model setting
#         #############################################################################################################
#         model_ft = models.vgg16(pretrained=True)
#         for param in list(model_ft.parameters())[:-24]:
#             param.requires_grad = False
#         num_ftrs = model_ft.classifier[6].in_features
#         model_ft.classifier[6]  = nn.Sequential(
#             nn.Linear(num_ftrs, 2),
#         )
#         model_ft = model_ft.to(device) # 放入裝置

#         n_epochs = 10
#         optimizer = torch.optim.Adam([
#             {'params':model_ft.parameters()}
#         ], lr=0.0001)

#         #############################################################################################################
#         #    ft model setting
#         #############################################################################################################

#         criterion = nn.CrossEntropyLoss()
#         patience = 3
#         train(model_ft,
#             name,
#             n_epochs,
#             train_dataloader,
#             dev_dataloader,
#             optimizer,
#             criterion,
#             batchsize,
#             patience)
    
    # 3. evaluate：做最終 f1,ma測試
    ## 
    f1_history = {}
    for class_index in range(5):
        name = f'vgg16_ft_diseasepart_{class_index}'
        model_ft = torch.load(f'./result_disease/{name}/model.pt')
        model_ft_revise = torch.load(f'./result_disease/vgg16_ft_diseasepart_3/model.pt')
        p,r,f1 = predict_real_f1(model_ft , model_ft_revise , class_index, path_dev, label_dev, box_path_dev)
        f1_history[f'class_{class_index}'] = {'precision':p , 'recall':r , 'F1':f1}
    with open(f'./result_disease/result.json', 'w') as json_file:
        json.dump(f1_history, json_file)
    