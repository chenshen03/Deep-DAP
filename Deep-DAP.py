#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import os, sys
import h5py
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.applications.xception import preprocess_input
from datatool import set_dataset, load_classes, load_attrs, load_allAttrs, load_data
from modeltool import load_ResNet50, load_Xception, load_InceptionV3, set_numclasses, load_model


DATASET = ''
MODELNAME = ''


def save_history(hist):
	loss = hist.history['loss']
	val_loss = hist.history['val_loss']
	acc = hist.history['acc']
	val_acc = hist.history['val_acc']

	loss = np.array(loss)
	val_loss = np.array(val_loss)
	acc = np.array(acc)
	val_acc = np.array(val_acc)

	file = h5py.File('./his/[dp5]history_' + DATASET + '_' + MODELNAME + '.h5', 'w')
	file.create_dataset('loss', data = loss)
	file.create_dataset('val_loss', data = val_loss)
	file.create_dataset('acc', data = acc)
	file.create_dataset('val_acc', data = val_acc)
	file.close()


def train_model(model, X_train, y_train, X_val, y_val, attrs):
    # X_train_norm = preprocess_input(X_train)
    # X_val_norm = preprocess_input(X_val)
    X_train_norm = X_train / 255
    X_val_norm = X_val / 255
    attr_train = attrs[y_train - 1][:,0,:]
    attr_val = attrs[y_val - 1][:,0,:]

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    fileName = "./model/[dp5]Deep-RIS_" + DATASET + '_' + MODELNAME
    tbCallBack = TensorBoard(log_dir='./logs')
    checkpointer = ModelCheckpoint(filepath=fileName+'.h5', verbose=1, save_best_only=True)
    # train the model on the new data for a few epochs
    hist = model.fit(X_train_norm, attr_train, epochs=40, batch_size=32, validation_data=(X_val_norm, attr_val),
              callbacks=[checkpointer, tbCallBack])
    save_history(hist)
    model.save(fileName + '-final.h5')


def evalute_model(model, X, y, attrs):
    # X_eval = preprocess_input(X)
    X_eval = X / 255
    attr = attrs[y - 1][:, 0, :]
    res = model.evaluate(X_eval, attr)
    print(res)


def parseArgv():
    global DATASET, MODELNAME
    
    try:
        DATASET = sys.argv[1]
    except IndexError:
        print("Must specify dataset name!")
        raise SystemExit
    if DATASET not in ['AwA', 'CUB']:
        print('dataset ' + DATASET + ' not exist!')
        raise SystemExit
    set_dataset(DATASET)
    set_numclasses(DATASET)

    try:
        MODELNAME = sys.argv[2].strip()
    except IndexError:
        print("Must specify model name!")
        raise SystemExit
    if MODELNAME == 'ResNet50':
        model = load_ResNet50()
    elif MODELNAME == 'InceptionV3':
        model = load_InceptionV3()
    elif MODELNAME == 'Xception':
        model = load_Xception()
    else:
        print('model ' + MODELNAME + ' not exist!')
        raise SystemExit
    print('load model ' + MODELNAME + ' successful!')

    return DATASET, model


if __name__ == '__main__':
    DATASET, model = parseArgv()
    classes = load_classes()
    attrs = load_attrs(src_idx=classes['source_idx'], binary=True)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    train_model(model, X_train, y_train, X_val, y_val, attrs)
    evalute_model(model, X_test, y_test, attrs)