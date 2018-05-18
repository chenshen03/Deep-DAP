#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os
import h5py
import sys


DATAPATH = './data/'


def set_dataset(dataset):
    global DATAPATH
    DATAPATH = DATAPATH + dataset + '/'


def load_classes():
    classes = [line.strip().split()[1] for line in open(DATAPATH + 'classes.txt')]
    source_classes = [line.strip().split()[1] for line in open(DATAPATH + 'train_classes.txt')]
    target_classes = [line.strip().split()[1] for line in open(DATAPATH + 'test_classes.txt')]

    source_idx = [classes.index(cls) for cls in source_classes]
    target_idx = [classes.index(cls) for cls in target_classes]
    classes = {'all': classes,
               'source': source_classes,
               'source_idx': source_idx,
               'target': target_classes,
               'target_idx': target_idx}
    print('load classes successful!')
    return classes


def load_attrs(src_idx=None, trg_idx=None, mean_correction=False, binary=False):
    if binary:
        codes_fn = DATAPATH + 'predicate-matrix-binary.txt'
    else:
        codes_fn = DATAPATH + 'Codes_Attributes.txt'
    semantics_fn = DATAPATH + 'attributes.txt'

    semantics = [line.strip().split()[1] for line in open(semantics_fn)]
    codes = np.loadtxt(codes_fn).astype(float)
    if codes.max() > 1:
        codes /= 100.

    # Set undefined codes (typically marked with -1) to the mean
    code_mean = codes[src_idx, :].mean(axis=0)
    for s in range(len(semantics)):
        codes[codes[:, s] < 0, s] = code_mean[s] if mean_correction else 0.5

    # 在上面对未定义的属性赋值为均值，一定程度上影响了数据分布，因此需要对其进行修正
    # Mean correction
    if mean_correction:
        for s in range(len(semantics)):
            codes[:, s] = codes[:, s] - code_mean[s] + 0.5
    print('load attributes successful!')
    print("attrs shape: " + str(codes.shape))
    return codes


def load_allAttrs():
    allAttrs = []
    with open(DATAPATH + 'attributes.txt') as f:
        allAttrs = [line.strip().split(' ')[1] for line in f.readlines()]
    print('load all attributes successful!')
    print('attributes number: ' + len(allAttrs))
    return allAttrs


def load_data(data_path='H5py/data_standard.h5'):
    data_path = DATAPATH + data_path
    if os.path.exists(data_path):
        print("loading data_standard.h5...")
        file = h5py.File(data_path, 'r')
        X_train = file['X_train'][:]
        y_train = file['y_train'][:]
        X_val = file['X_val'][:]
        y_val = file['y_val'][:]
        X_test = file['X_test'][:]
        y_test = file['y_test'][:]
        attrs = file['attrs'][:]
        file.close()

        # print("number of train examples = " + str(X_train.shape[0]))
        # print("number of val examples = " + str(X_val.shape[0]))
        # print("number of test examples = " + str(X_test.shape[0]))
        # print("number of attributes = " + str(attrs.shape[0]))
        print("X_train shape: " + str(X_train.shape))
        print("y_train shape: " + str(y_train.shape))
        print("X_val shape: " + str(X_val.shape))
        print("y_val shape: " + str(y_val.shape))
        print("X_test shape: " + str(X_test.shape))
        print("y_test shape: " + str(y_test.shape))
        # print("attrs shape: " + str(attrs.shape))
    else:
        print("data_standard.h5 not exist.")

    return X_train, y_train, X_val, y_val, X_test, y_test