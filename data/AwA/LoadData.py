#!/usr/bin/python
# -*- coding: utf-8 -*-


import keras
from keras.preprocessing import image
import scipy.io as sio
import numpy as np
import h5py, os


data_path = "./AwA_dataset/classes.mat"
data = sio.loadmat(data_path)
classes = data['classes']
classes = [c.rstrip() for c in classes]
trainclasses_id = data['trainclasses_id'][0]
testclasses_id = data['testclasses_id'][0]

print(classes)
print(trainclasses_id)
print(testclasses_id)


'''
Load the image files form the folder
input:
    imgDir: the direction of the folder
    imgName:the name of the folder
output:
    data:the data of the dataset
    label:the label of the datset
'''
def load_Img(imgDir, imgFoldName):
    print("loading " + imgFoldName.rstrip("/") + "...")
    imgs = os.listdir(imgDir + imgFoldName)
    imgNum = len(imgs)
    data = np.empty((imgNum, 224, 224, 3), dtype="float32")
    label = np.empty((imgNum, 1), dtype=int)
    for i in range (imgNum):
        # print("loading " + imgs[i] + "...")
        img = image.load_img(imgDir + imgFoldName + imgs[i], target_size=(224, 224))
        x = image.img_to_array(img)
        data[i, :, :, :] = x
        labelName = imgs[i].split('_')[0]
        label[i] = classes.index(labelName)
    print("the number of " + imgFoldName.rstrip("/") + " = " + str(data.shape[0]))
    return data, label


def load_AwA_dataset(classes, trainclasses_id, testclasses_id, data_path="./Animals_with_Attributes/"):
    X_test = []
    y_test = []
    for test_id in testclasses_id:
        test_foldname = classes[test_id - 1]
        test_foldname = test_foldname + "/"
        data, label = load_Img(data_path, test_foldname)
        X_test.extend(data)
        y_test.extend(label)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_test shape: " + str(X_test.shape))
    print("y_test shape: " + str(y_test.shape))

    X_train = []
    y_train = []
    for train_id in trainclasses_id:
        train_foldname = classes[train_id - 1]
        train_foldname = train_foldname + "/"
        data, label = load_Img(data_path, train_foldname)
        X_train.extend(data)
        y_train.extend(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print("number of train examples = " + str(X_train.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))
    return X_train, y_train, X_test, y_test


def save_data(X_train, y_train, X_test, y_test):
    # 创建HDF5文件
    file = h5py.File('./AwA_dataset/AwA_data.h5', 'w')
    # 写入
    file.create_dataset('X_train', data = X_train)
    file.create_dataset('y_train', data = y_train)
    file.create_dataset('X_test', data = X_test)
    file.create_dataset('y_test', data = y_test)
    file.close()


if __name__ == '__main__':
    # classes, trainclasses_id, testclasses_id = load_classes()
    X_train, y_train, X_test, y_test = load_AwA_dataset(classes, trainclasses_id, testclasses_id)
    save_data(X_train, y_train, X_test, y_test)