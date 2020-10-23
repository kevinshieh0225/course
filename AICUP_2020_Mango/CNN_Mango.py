import os
import numpy as np
import re
import cv2
from glob import glob
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
import gc

label2idx = {
    '不良-乳汁吸附': 0,
    '不良-機械傷害': 1,
    '不良-炭疽病': 2,
    '不良-著色不佳': 3,
    '不良-黑斑病': 4
}

def load_mango_csv(csv_path='./C2_TrainDev/train.csv'):
    path = []
    box = []
    label = []
    subdir = csv_path.split('/')[-1].split('.')[0].capitalize() #[-1]意思是倒數最後一col，.capitalize()將首英文字母大寫，其他小寫
    # subdir : Train , Dev
    # 此時我需要的輸出格式為：
    # path: 照片路徑 : ./C2_TrainDev/Train/img.jpg
    # label: 標籤  : [1,0,0,0,0]
    with open(csv_path, 'r', encoding='utf8') as f:        
        for line in f:
            clean_line = re.sub(',+\n', '', line).replace('\n', '').replace('\ufeff', '').split(',')
            curr_img_path = f'./C2_TrainDev/{subdir}/{clean_line[0]}'
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

def mkdir(pixel,file_path='file_data'):
  file_name = file_path+'_'+str(pixel)
  if not os.path.isdir(file_name):
    os.makedirs(file_name)

def data_preprocessing(path, label, resize_shape = (150,150)):
    '''
    Args:
        path: List of image paths.
        label: List of labels.
        resize_shape: union the size of the picture
    '''
    x = []
    print('Start ready data')
    dh = display('Start',display_id=True)
    #dh2 = display('Start',display_id=True)
    for i in range(len(path)):
        # use TRY because some images are missing in the image folder
        try:
            img = cv2.cvtColor(cv2.imread(path[i]), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, resize_shape)
            x.append(img)
        except:
            print('badluck')
            print(i)
            continue
        if i%50 == 0:
          dh.update(i)
          #dh2.update(x.shape)
    return x, label
def load_prepare_data(file_prefix, pixel, datatype):
  """
  Arg:
    file_prefix: Any file name you want.
    pixel: the resize shape of picture.
    datatype: string of train, dev
  """
  print('build '+datatype+' data...')
  file_name = f'./file_data_{pixel}/{file_prefix}_'+datatype
  path, label = load_mango_csv(csv_path='./C2_TrainDev/'+datatype+'.csv')
  if os.path.isfile(f'{file_name}_x.npy') and os.path.isfile(f'{file_name}_y.npy'):
      x = np.load(f'{file_name}_x.npy')
      y = np.load(f'{file_name}_y.npy')
  else:
      x, y = data_preprocessing(path, label, resize_shape=(pixel,pixel))
      np.save(f'{file_name}_x', np.array(x))
      np.save(f'{file_name}_y', np.array(y))
      x = np.array(x)
      y = np.array(y)

  print(datatype+"_x shape : "+str(x.shape))
  print(datatype+"_y shape : "+str(y.shape))
  
  return x,y


def CNNforClassify(file_prefix,pixel):
    '''
    Args:
        file_prefix: Any file name you want.
        pixel: the resize shape of picture.
    '''
    print('創建目錄')
    mkdir(pixel)
    train_x,train_y = load_prepare_data(file_prefix,pixel,'train')
    print('train CNN model...')
    #像素是0~255，將像素收斂在0~1之間
    
    print('alive')
    # Model Structure
    model = Sequential()
    
    model.add(Conv2D(filters=64, kernel_size=5, input_shape=(pixel, pixel, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
    """
    model.add(Conv2D(filters=128, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
    """
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='softmax'))
    model.add(Dense(5, activation='softmax'))
    print(model.summary())

    # Train
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=1)
    # Test
    del train_x,train_y
    gc.collect()
    dev_x,dev_y = load_prepare_data(file_prefix,pixel,'dev')

    loss, accuracy = model.evaluate(dev_x, dev_y)
    print('Test:')
    print('Loss: %s\nAccuracy: %s' % (loss, accuracy))
"""
    # Save model
    model.save('./CNN_Mango_model.h5')

    # Load Model
    model = load_model('./CNN_Mango_model.h5')

# Display
def plot_img(n):
    plt.imshow(X_test[n], cmap='gray')
    plt.show()


def all_img_predict(model):
    print(model.summary())
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Loss:', loss)
    print('Accuracy:', accuracy)
    predict = model.predict_classes(x_test)
    print(pd.crosstab(Y_test.reshape(-1), predict, rownames=['Label'], colnames=['predict']))


def one_img_predict(model, n):
    predict = model.predict_classes(x_test)
    print('Prediction:', predict[n])
    print('Answer:', Y_test[n])
    plot_img(n)
"""
def VGGnet(file_prefix,pixel):
  '''
  Args:
      file_prefix: Any file name you want.
      pixel: the resize shape of picture.
  '''
  print('創建目錄')
  mkdir(pixel)
  train_x,train_y = load_prepare_data(file_prefix,pixel,'train')
  print('train CNN model...')
  #像素是0~255，將像素收斂在0~1之間
  
  print('alive')
  # Model Structure
  model = Sequential() 
  # Block 1 
  model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(pixel, pixel, 3), activation='relu', padding='same'))    
  model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))    
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  # Block 2
  model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))    
  model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))    
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  # Block 3
  model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'))    
  model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'))       
  model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'))    
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  # Block 4
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))   
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))    
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) 
  # Block 5
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))    
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) 
  # FC layers
  model.add(Flatten())    
  model.add(Dense(4096, activation='relu'))    
  model.add(Dropout(0.5))    
  model.add(Dense(4096, activation='relu'))    
  model.add(Dropout(0.5))    
  model.add(Dense(5, activation='softmax'))

  print(model.summary())

  # Train
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=1)
  # Test
  del train_x,train_y
  gc.collect()
  dev_x,dev_y = load_prepare_data(file_prefix,pixel,'dev')

  loss, accuracy = model.evaluate(dev_x, dev_y)
  print('Test:')
  print('Loss: %s\nAccuracy: %s' % (loss, accuracy))

if __name__ == '__main__':
    # file_prefix = f'{defective_type.split('-')[-1]}'
    # HOG_ANOVA_SVM(file_prefix, defective_type, anova_percentile=5, slice_img=True, linear_svm=False)
    exit()