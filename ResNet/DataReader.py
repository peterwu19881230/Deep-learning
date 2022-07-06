import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
	""" Load the CIFAR-10 dataset.

	Args:
		data_dir: A string. The directory where data batches are stored.

	Returns:
		x_train: An numpy array of shape [50000, 3072]. 
		(dtype=np.float32)
		y_train: An numpy array of shape [50000,]. 
		(dtype=np.int32)
		x_test: An numpy array of shape [10000, 3072]. 
		(dtype=np.float32)
		y_test: An numpy array of shape [10000,]. 
		(dtype=np.int32)
	"""
	### YOUR CODE HERE
	dicts=[]
	for file in [data_dir+file_batch for file_batch in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']]:
		with open(file, 'rb') as fo:	    
			dicts.append(pickle.load(fo, encoding='bytes')) 
	train_data=np.concatenate((dicts[0][b'data'],dicts[1][b'data'],dicts[2][b'data'],dicts[3][b'data'],dicts[4][b'data']))
	train_label=np.concatenate((dicts[0][b'labels'],dicts[1][b'labels'],dicts[2][b'labels'],dicts[3][b'labels'],dicts[4][b'labels']))
	test_data=dicts[5][b'data']
	test_label=dicts[5][b'labels']      

	x_train=train_data
	y_train=train_label
	x_test=test_data
	y_test=test_label

	### YOUR CODE HERE

	return x_train, y_train, x_test, y_test    

def train_vaild_split(x_train, y_train, split_index=45000):
	""" Split the original training data into a new training dataset
		and a validation dataset.

	Args:
		x_train: An array of shape [50000, 3072].
		y_train: An array of shape [50000,].
		split_index: An integer.

	Returns:
		x_train_new: An array of shape [split_index, 3072].
		y_train_new: An array of shape [split_index,].
		x_valid: An array of shape [50000-split_index, 3072].
		y_valid: An array of shape [50000-split_index,].
	"""
	x_train_new = x_train[:split_index]
	y_train_new = y_train[:split_index]
	x_valid = x_train[split_index:]
	y_valid = y_train[split_index:]

	return x_train_new, y_train_new, x_valid, y_valid
