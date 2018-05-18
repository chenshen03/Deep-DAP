#!/usr/bin/python
# -*- coding: utf-8 -*-

import keras
from keras import backend as K
from keras.models import Model, load_model, Input
from keras.layers import Dense, Flatten
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint

import numpy as np
import os
import h5py
from tool import load_attrs, load_classes, load_data


def load_ResNet50(output_weight):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = True

    # add a global spatial average pooling layer
    x = base_model.output
    x = Flatten()(x)
    # let's add a fully-connected layer
    x = Dense(85, activation='relu')(x)
    # and a logistic layer -- let's say we have 10 classes
    predictions = Dense(50, activation='softmax', use_bias=False)(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    print("model ouput shape: " + str(model.output_shape))

    # 将最后一层的权重固定为semantic code of class
    output_weight = np.expand_dims(output_weight.T, axis=0)
    print(output_weight.shape)
    output_layer = model.get_layer('dense_2')
    output_layer.set_weights(output_weight)
    output_layer.trainable = False

    return model


def train_model(model, X_train, y_train, X_val, y_val, attrs):
    onehot_train = keras.utils.to_categorical(y_train - 1, num_classes=50)
    onehot_val = keras.utils.to_categorical(y_val - 1, num_classes=50)

    # X_train_norm = preprocess_input(X_train)
    # X_val_norm = preprocess_input(X_val)
    # attr_train = attrs[y_train - 1][:,0,:]
    # attr_val = attrs[y_val - 1][:,0,:]
    # print(attr_train.shape)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="./model/Deep-RULE_AwA_ResNet50.h5", verbose=1, save_best_only=True)
    # train the model on the new data for a few epochs
    model.fit(X_train, onehot_train, epochs=20, batch_size=32, validation_data=(X_val, onehot_val),
              callbacks=[checkpointer])
    model.save('./model/Deep-RULE_AwA_ResNet50-final.h5')
    model.evaluate()


def evalute_model(model, X, y, attrs):
    # X = preprocess_input(X)
    attr = attrs[y - 1][:, 0, :]
    res = model.evaluate(X, attr)
    print(res)


def test_model(model, X_test, attrs, classes):
    X_test = np.expand_dims(X_test, axis=0)
    print("test shape:" + str(X_test.shape))
    X_test_norm = preprocess_input(X_test)
    preds = model.predict(X_test)
    print(preds)
    dist = np.sqrt(np.sum(np.square(attrs - preds), axis=1))
    pred_index = np.argmax(dist)
    print('prediction: No.' + str(pred_index + 1) + ' ' + classes['all'][pred_index])


if __name__ == '__main__':
    classes = load_classes()
    attrs = load_attrs(src_idx=classes['source_idx'])
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    model = load_ResNet50(attrs)
    # model = load_model('./model/Deep-RIS-final.h5')
    train_model(model, X_train, y_train, X_val, y_val, attrs)
    # test_model(model, X_test[2], attrs, classes)
    evalute_model(model, X_train[0:1000,:,:,:], y_train[0:1000], attrs)
    evalute_model(model, X_val[1:2000,:,:,:], y_val[1:2000], attrs)
    evalute_model(model, X_test[0:1000,:,:,:], y_test[0:1000], attrs)