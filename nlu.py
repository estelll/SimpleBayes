# -*-coding:utf-8 -*-
"""
Created on April 1st 2017

@author: ML Hou
"""
import os
import sys
import pdb
import pickle

import data_utils
import model

def download_model(model_name):
	"""Download the model according to the model name.
	Args:
		model_name: the name of model to be downloaded.
	Return:
		model downloaded.
	"""
	model_file = './model/' + model_name + '/model.pickle'
	with open(model_file,'rb') as f:
		v = pickle.load(f)
	model = v['model']
	# pdb.set_trace()
	return model

def interface(user_input):
	"""An interface to other mudual"""
	model = download_model('Bayes')
	vocab_dir = './data/Bayes_vocabulary'
	vocabulary = read_in_vocabulary(vocab_dir)

	_input = data_utils.cut_sentence(user_input,'jieba')

	pred = model.predict([_input])
	result = vocabulary[pred[0]]
	return result

def read_in_vocabulary(vocab_dir):
	# vocab_list = []
	if os.path.isfile(vocab_dir):
		with open(vocab_dir,'rt',encoding = 'utf-8') as f:
			vocab_list = [line.strip() for line in f.readlines()]
		return vocab_list
	else:
		raise ValueError('Vocabulary file %s does not exists.' % vocab_dir)
if __name__ == '__main__':
	# model = download_model('Bayes')
	while  True:
		# user_input = []
		# user_input.append(input('Please input a sentence:'))
		user_input = input('Please input a sentence:')
		result = interface(user_input)
		print(result)

