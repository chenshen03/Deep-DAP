#!/usr/bin/python
# -*- coding: utf-8 -*-

import keras
from keras.preprocessing import image
import scipy.io as sio
import numpy as np
import h5py, os, sys


# 读取属性信息
def load_attrs(DATASET, fileName):
    attrsFilePath = DATASET + fileName
    print('loading attributes...')
    attrs = np.loadtxt(attrsFilePath).astype(float)
    print('Attributes have been loaded successful!')
    return attrs


# 读取类别信息
def load_classes(DATASET, fileName):
    classesFilePath = DATASET + fileName
    print('loading classes...')
    classes = {}
    with open(classesFilePath) as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            classes[line[1]] = int(line[0])
    print('Classes have been loaded successful!')
    return classes


# 读取图片、标签信息
def load_data(DATASET, fileName):
    filePath = DATASET + fileName
    imgPath = DATASET + '/images/'
    print('loading ' + filePath + '...')
    imgsName = []
    for line in open(filePath):
        line = line.strip()
        if DATASET == 'AwA':
            line = list(line)
            line.insert(-8, '1')
            line = ''.join(line)
        if os.path.exists(imgPath + line):
            imgsName.append(line)
    imgNum = len(imgsName)
    print('the numbers of ' + filePath + ": " + str(imgNum))
    data = np.empty((imgNum, 224, 224, 3), dtype="float32")
    label = np.empty((imgNum, 1), dtype=int)
    attr = np.empty((imgNum, 85), dtype='float32')
    for i in range(imgNum):
        # print('loading image: ' + imgsName[i])
        img = image.load_img(imgPath + imgsName[i], target_size=(224, 224))
        data[i, :, :, :] = image.img_to_array(img)
        label[i] = classes.get(imgsName[i].split('/')[0])
    print('Images and labels have been loaded successful!')
    return data, label


# 保存训练集、验证集和测试集的信息
def save_data(DATASET):
    print('Saving dataset ' + DATASET + '...')
    # 创建HDF5文件
    file = h5py.File(DATASET + '/H5py/' + DATASET + '_data_standard.h5', 'w')
    # 写入
    file.create_dataset('X_train', data = X_train)
    file.create_dataset('y_train', data = y_train)
    file.create_dataset('X_val', data = X_val)
    file.create_dataset('y_val', data = y_val)
    file.create_dataset('X_test', data = X_test)
    file.create_dataset('y_test', data = y_test)
    file.create_dataset('attrs', data = attrs)
    file.close()
    print(DATASET + ' have been saved successful!')


if __name__ == '__main__':
    try:
        DATASET = sys.argv[1]
    except IndexError:
        print("Must specify dataset name!")
        raise SystemExit

    attrs = load_attrs(DATASET, '/Codes_Attributes.txt')
    classes = load_classes(DATASET, '/classes.txt')
    X_test, y_test = load_data(DATASET, '/testZS.txt')
    X_val, y_val = load_data(DATASET, '/testRecg.txt')
    X_train, y_train = load_data(DATASET, '/train.txt')
    save_data(DATASET)