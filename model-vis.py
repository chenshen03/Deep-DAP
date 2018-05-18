#!/usr/bin/python
# -*- coding: utf-8 -*-


from keras.applications.inception_v3 import InceptionV3
from keras.applications import VGG16
from keras import activations
from keras.models import load_model
from vis.utils import utils
from vis.visualization import visualize_activation
from vis.input_modifiers import Jitter
import PIL.Image as Image
import numpy as np


def save_image(data, file_name):
	img_path='./imgs/Deep-RIS_AwA_Xception_top/'
	img = Image.fromarray(data, 'RGB')
	img.save(img_path + file_name)


# model = InceptionV3(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
model = load_model('./model/Deep-RIS_AwA_Xception.h5')
# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'dense_1')
# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)


def vis_1():
	# 20 is the imagenet category for 'ouzel'
	img = visualize_activation(model, layer_idx, filter_indices=6)
	save_image(img, '1.png')


def vis_2():
	# 20 is the imagenet category for 'ouzel'
	img = visualize_activation(model, layer_idx, filter_indices=6, max_iter=500, verbose=True)
	save_image(img, '2.png')


def vis_3():
	# 20 is the imagenet category for 'ouzel'
	# Jitter 16 pixels along all dimensions to during the optimization process.
	img = visualize_activation(model, layer_idx, filter_indices=36, max_iter=2000, input_modifiers=[Jitter(16)])
	save_image(img, 'filter_single_6_' + str(37) + '.png')


def vis_4():
	import numpy as np
	# categories = np.random.permutation(1000)[:15]

	vis_images = []
	image_modifiers = [Jitter(16)]
	for idx in range(10):
		print('filter_indices_: ' + str(idx + 1))
		img = visualize_activation(model, layer_idx, filter_indices=idx, max_iter=4000, input_modifiers=image_modifiers)
		save_image(img, '[4000]filter_' + str(idx + 1) + '.png')


def vis_5():
	from vis.visualization import get_num_filters
	# The name of the layer we want to visualize
	# You can see this in the model definition.
	layer_name = 'conv2d_1'
	layer_idx = utils.find_layer_idx(model, layer_name)

	# Visualize all filters in this layer.
	filters = np.arange(get_num_filters(model.layers[layer_idx]))

	# Generate input image for each filter.
	for idx in filters:
            print('filter_indices: ' + str(idx))
	    img = visualize_activation(model, layer_idx, filter_indices=idx)
	    save_image(img, layer_name + '_filter_' + str(idx) + '.png')


if __name__ == '__main__':
	vis_1()
	vis_2()
	vis_3()
	vis_4()
	vis_5()