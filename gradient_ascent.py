#gradient ascent

import time
import os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import cv2

import matplotlib.pyplot as plt
from scipy.misc import toimage
from scipy.misc import imsave

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Model
from keras import backend as K
#K.set_image_dim_ordering('tf')

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import load_model

output_index = 3
step = 10
layer_name = 'conv2d_92'
filter_index = 0



print('Defining model...')
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299,299,3))
print('Base Model defined')
top_model = load_model('top_model_inception500.h5')
print('Top Model defined')
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
print('Model Defined')
#layer_dict = dict([(layer.name, layer) for layer in model.layers])
#print(layer_dict)

#print(model.output)
#print(model.output[:, output_index])
#layer_output = layer_dict[layer_name].output
#loss = K.mean(layer_output[:, :, :, filter_index])
input_img = model.input
loss = K.mean(model.output[:, output_index])
grads = K.gradients(loss, input_img)[0]
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
iterate = K.function([input_img], [loss, grads])
input_img_data = np.random.random((1, 299, 299, 3)) * 20 + 128.
#print(input_img_data.shape)

for i in range(100):
	print(i)
	loss_value, grads_value = iterate([input_img_data])
	input_img_data += grads_value * step
	
def deprocess_image(x):
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	x += 0.5
	x = np.clip(x, 0, 1)

	x *= 255
	x = x.transpose((0, 1, 2))
	x = np.clip(x, 0, 255).astype('uint8')
	return x

img = input_img_data[0]
img = deprocess_image(img)
print(type(img))
print(img.shape)
imsave('class'+str(output_index)+'.png', img)