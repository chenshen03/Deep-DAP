#!/usr/bin/python
# -*- coding: utf-8 -*-

import keras
from keras.models import Model, load_model, Input
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3


NUMCLASSES = 85


def set_numclasses(dataset):
    global NUMCLASSES
    if dataset == 'AwA':
        NUMCLASSES = 85
    elif dataset == 'CUB':
        NUMCLASSES = 312


def load_InceptionV3():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 10 classes
    predictions = Dense(NUMCLASSES, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def load_Xception():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # x = Dropout(rate=0.5)(x)
    # and a logistic layer -- let's say we have 10 classes
    predictions = Dense(NUMCLASSES, activation='sigmoid')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def load_ResNet50():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    x = Flatten()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 10 classes
    predictions = Dense(NUMCLASSES, activation='sigmoid')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model