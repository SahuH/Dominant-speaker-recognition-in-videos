#plotting activations

import time
import os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.preprocessing import image as image_utils
import numpy as np
import scipy
import argparse
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import backend as K
from keras.models import load_model

#layer_name = 'conv2d_7'
layer_nos = [5,100, 200, 300]
filter_index = 0

imageName = str('./Train/jk/frame135.jpg')
image = image_utils.load_img(imageName, target_size=(299, 299))
image = image_utils.img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

print('Defining model...')
#base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299,299,3))
model = InceptionV3(include_top=True, weights='imagenet', input_shape=(299,299,3))
#top_model = load_model('top_model_inception2000.h5')
#model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
print('Model Defined')

#print(len(model.layers))
#layer_dict = dict([(layer.name, layer) for layer in model.layers])
#layer_output = layer_dict[layer_name].output
for layer_no in layer_nos:
	print(layer_no)
	get_activations = K.function([model.layers[0].input],[model.layers[layer_no].output])
	activations = get_activations([image,0])
	activations = np.array(activations)
	print(activations.shape)
	plt.imshow(activations[0,0,:,:,filter_index],cmap='gray')
	plt.savefig('jk_layer'+str(layer_no)+'.png')