import time
import os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import cv2

import matplotlib.pyplot as plt
from scipy.misc import toimage

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
from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers

class Object_identification(object):
	def __init__(self):
		self.images_path = os.path.join(os.path.dirname(__file__), 'images')
		self.videos_path = os.path.join(os.path.dirname(__file__), 'VIDEOS_360P')
		self.results_path = os.path.join(os.path.dirname(__file__), 'results')
		self.num_example = None
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.y_predict = None
		self.y_train_vector = None
		self.y_test_vector = None
		self.y_predict_vector = None
		self.num_labels = 7
		self.output_dim = 7
		self.class_var = [0,1,2,3,4,5,6]
		self.model = None
		self.history = None
		self.epochs = 2
		self.seed = 7
		self.w = 1
		self.top_model = None
		self.base_model = None
		self.accuracy = None
		self.model=None
		self.features_test=None
		self.top_model_weights_path = './results/top_model_weights'+str(self.w)+'.h5'
		self.model_path = './results/model_without_finetuned'+str(self.w)+'.h5'

	def plot_images(X, y):
		y = np.argmax(y,axis=1)
		step = int(round(X.shape[0]/5))
		for i in range(0, 5*step, step):
			print(i)
			for j in range(0, 20):
				plt.subplot(4,5,1 + j)	
				plt.imshow(toimage(X[i+j]))
				plt.title('label: '+str(y[i+j]))
			plt.show()
			
	def load_numpy_arrays(self):
		self.X_train = np.load('./X_train.npy')
		self.y_train_vector = np.load('./y_train.npy')
		self.X_test = np.load('./numpy_test/X_test.npy')
		self.y_test_vector = np.load('./numpy_test/y_test.npy')
		print(self.X_train.shape, self.y_train_vector.shape)
		print(self.X_test.shape, self.y_test_vector.shape)	
		
	def using_pretrained_model(self, X_train, y_train):
		print('using pretrained model..')
		print(X_train.shape, y_train.shape)		
		features_train = self.base_model.predict(X_train, verbose=0)
		
		top_model = Sequential()
		top_model.add(Flatten(input_shape=features_train.shape[1:]))
		top_model.add(Dense(256, activation='relu'))
		top_model.add(Dropout(0.5))
		top_model.add(Dense(self.num_labels, activation='softmax'))
		top_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
		print(top_model.summary())
		history = top_model.fit(features_train, y_train, validation_data=(self.features_test,self.y_test_vector), epochs=self.epochs, batch_size=32)
		self.accuracy = history.history['val_acc'][-1]
		top_model.save('top_model_inception'+str(X_train.shape[0])+'.h5')

if __name__ == '__main__':
	obj = Object_identification()
	obj.load_numpy_arrays()
	train_size = [50,100,500,1000,1500,2000,2500,3000,3500]
	accuracy = []
	obj.base_model = InceptionV3(include_top=False, weights='imagenet')
	obj.features_test = obj.base_model.predict(self.X_test, verbose=0)
	for i in train_size:
		X_train = obj.X_train[:i]
		y_train = obj.y_train_vector[:i]
		obj.using_pretrained_model(X_train, y_train)
		accuracy.append(obj.accuracy)
		np.save('accuracy.npy',accuracy)
	plt.plot(train_size, accuracy)
	plt.savefig('accu_vs_size.png')