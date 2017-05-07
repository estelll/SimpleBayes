# -*-coding:utf-8 -*-
"""
Created on April 1st 2017

@author: ML Hou
"""
import os
import sys
import pdb

import data_utils
import model

import pickle

def init_model(model_name,config_dir):
	"""Initialize the model according to model_name and config.
	Args:
		model_name: the name of model, such as Bayes, SVM and so on.
		config_dir: a file containing different paramter settings.
	Return:
		a training model.
	"""
	if model_name == 'Bayes':
		train_model = model.Bayes(config_dir)
		train_model.get_config()
	return train_model

def training(model, train):
	"""Training the model."""
	model.fit(train)

def main():
	# Prepare dataset.
	raw_file = './data/y_n_all' # raw data file.
	prepared_dir = './data/' # dir of prepared data(sentences cut down and labels are number)
	# cut_mode = 'character'
	cut_mode = 'jieba'
	# cut_mode = '2-gram'
	vocab_dir = './data/Bayes_vocabulary'
	prepared_data,prepared_label = data_utils.prepare_data(raw_file,prepared_dir,cut_method = cut_mode, vocab_dir = vocab_dir)
	print('Get prepared dataset.')
	# pdb.set_trace()
	# print(prepared_data)

	# Get training and test dataset.
	traning_dir = './data/train/'
	test_dir = './data/test/'
	ratio = 0
	train,test = data_utils.get_data(list(zip(prepared_data,prepared_label)),traning_dir,test_dir,ratio = ratio, cut_method = cut_mode)
	print('Get training and test dataset.')
	# pdb.set_trace()
	# print(train.data)
	# train
	model_name = 'Bayes'
	config_dir = './model/'
	train_model = init_model(model_name,config_dir)
	print('Initialize the model.')
	training(train_model,train)
	print('Training finished.')
	# store the variable.
	v = {'model':train_model}
	model_file = train_model.model_path + 'model.pickle'
	with open(model_file,'wb') as f:
		pickle.dump(v, f)
	# evaluate
	# acc = model.evaluate(train_model,test)

	# test
	# acc = train_model.test(test)

	# test
	# test_str = ['对']
	# print(train_model.predict(test_str))

	# print('Accuracy:')
if __name__ == '__main__':
	main()