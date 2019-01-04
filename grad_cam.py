#Code to plot Grad-CAM heatmap plot to visualize rained model

'''
Resources:
https://github.com/raghakot/keras-vis/blob/master/examples/vggnet/attention.ipynb
https://raghakot.github.io/keras-vis/
https://raghakot.github.io/keras-vis/vis.visualization/#visualize_saliency
'''


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
#from keras.models import load_weights
from keras.models import Model
from keras import backend as K
#K.set_image_dim_ordering('tf')
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import load_model
import vis
from vis import visualization
import matplotlib.cm as cm
from vis.visualization import overlay
from vis.utils import utils
from keras import activations

print('Defining model...')
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299,299,3))
print('Base Model defined')
top_model = load_model('top_model_inception500.h5')
print('Top Model defined')
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
print('Model Defined')

layer_idx = utils.find_layer_idx(model, 'sequential_1')
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

image_array = cv2.imread('frame3054.jpg')
image_array = cv2.resize(image_array, (299, 299))
heatmap = visualization.visualize_cam(model, layer_idx, 1, seed_input=image_array)
#jet_heatmap = np.uint8(cm.jet(heatmap)[..., :3] * 299)
plt.imshow(overlay(heatmap, image_array))
plt.show()
