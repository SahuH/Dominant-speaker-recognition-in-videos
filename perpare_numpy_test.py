#preparing numpy arrays for test data

import time
import os, os.path
import numpy as np
import cv2
from keras.utils import np_utils
from scipy import misc

images_path = 'drive/dl_assn2/Test'

def prepare_input():
	start = time.time()
	X = []
	y = []
	folder_no = 0
	dict ={'AK':0,'SB':1,'FR':2,'SG':3,'SM':4,'SK':5,'jk':6}
	label_file = open('label_1.csv','r')
	i=0
	images = os.listdir(images_path)
	for label in label_file:
		image = images[i]
		print('Creating array for '+images_path+'/'+image+'...')
		image_array = cv2.imread(class_images_path+'/'+image)
		image_array = cv2.resize(image_array, (299, 299))
		X.append(image_array)
		y.append(dict[strip('\n').replace(',','')])
		i+=1

	X = np.array(X)
	y = np.array(y)
	p = np.random.permutation(X.shape[0])
	X = X[p]
	y = y[p]

def plot_images(X, y):
	y = np.argmax(y, axis=1)
	step = round(X.shape[0]/5)
	for i in range(0, 5*step, step):
		print(i)
		for j in range(0, 20):
			plt.subplot(4,5,1 + j)	
			plt.imshow(toimage(X[i+j]))
			plt.title('label: '+str(y[i+j]))
		plt.show()
	
def write(X,y):
	print('normalizing...')
	X = X/ 255.0	
	y = np_utils.to_categorical(y)
	print("Time taken for creating numpy arrays: ", time.time() - start)
	
	np.save('drive/dl_assn2/numpy_arrays/X_test.npy', X)
	np.save('drive/dl_assn2/numpy_arrays/y_test.npy', y)
	print('Done.')
	
if __name__ == '__main__':
	X,y = prepare_input()
	plot_images(X,y)
	write(X,y)