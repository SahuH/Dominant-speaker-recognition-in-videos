#preparing numpy arrays for train data

import time
import os, os.path
import numpy as np
import cv2
from keras.utils import np_utils
from scipy import misc

images_path = 'drive/dl_assn2/Train'

def prepare_input():
		start = time.time()
		X = []
		y = []
		folder_no = 0
		dict ={'AK':0,'SB':1,'FR':2,'SG':3,'SM':4,'SK':5,'jk':6}
		for folder in os.listdir(images_path):
			class_images_path = images_path+'/'+folder
			print('Moving to: '+ class_images_path)
			i=0
			for image in os.listdir(class_images_path):
				print('Creating array for '+class_images_path+'/'+image+'...')
				image_array = cv2.imread(class_images_path+'/'+image)
				image_array = cv2.resize(image_array, (299, 299))
				X.append(image_array)
				y.append(dict[folder])
				
				if i==10:
					break
				i+=1
				
		folder_no+=1
		X = np.array(X)
		y = np.array(y)
		
		p = np.random.permutation(X.shape[0])
		X = X[p]
		y = y[p]
		#X = X.astype('float32')
		print('normalizing...')
		X = X/ 255.0	
		y = np_utils.to_categorical(y)
		print("Time taken for creating numpy arrays: ", time.time() - start)
		
		np.save('drive/dl_assn2/numpy_arrays/X_train.npy', X)
		np.save('drive/dl_assn2/numpy_arrays/y_train.npy', y)
		print('Done.')
if __name__ == '__main__':
	prepare_input()
