# -*-coding:utf-8 -*-
"""
Created on April 1st 2017

@author: ML Hou
"""
import sys
import os
import pdb

import random

import jieba

class Dataset(object):
	"""Format for dataset, including data and target."""
	def __init__(self, data, label):
		self.data = data
		self.target = label
		# super(Dataset, self).__init__()
		# self.arg = arg

def n_gram(sentence,n):
	"""cut the sentence by n-gram.
	Args:
		sentence: the sentence to be cut.
		n: n-gram.
	Return:
		words separated by space.
	"""
	if len(sentence) <= n:
		return sentence

	word_list = []
	for i in range(len(sentence) - (n - 1)):
		word_list.append(sentence[i:i+n])
	return ' '.join(word_list)

def cut_sentence(sentence,method):
	"""cut sentences into words.
	Args:
		sentence: the sentence to be cut.
		method: the method of cutting, e.g. jieba | n-gram | character.
		n: when the method is n-gram, n refers to the 'n'
	Return:
		words separated by space.
	Raises:
		ValueError: if the method is Invalid
	"""
	if '-gram' in method:
		n = (int)(method.split('-')[0])
		return n_gram(sentence,n)
	elif method == 'jieba':
		return ' '.join(list(jieba.cut(sentence)))
	elif method == 'character':
		return ' '.join(list(sentence))
	else:
		raise ValueError("Invalid method name %s." % method)
		# return

def get_vocabulary(word_list, vocab_file):
	"""Transform the word_list to a vocabulary.
	Args:
		word_list: the word list to be transformed.
	Return:
		the vocabulary.
	"""
	vocabulary = {}
	if os.path.isfile(vocab_file):
		print('Get vocabulary from file %s.' % vocab_file)
		with open(vocab_file,'rt',encoding = 'utf-8') as f:
			for line in f.readlines():
				vocabulary[line.strip()] = len(vocabulary)
		return vocabulary
	else:
		print('Get vocabulary from word_list and write the vocabulary into file %s.' % vocab_file)
		for word in word_list:
			if word not in vocabulary:
				# pdb.set_trace()
				vocabulary[word] = len(vocabulary)
		vocab_list = sorted(vocabulary, key = vocabulary.get, reverse = False)
		# pdb.set_trace()
		with open(vocab_file,'wt', encoding = 'utf-8') as f:
			for w in vocab_list:
				f.write(w + '\n')
		return vocabulary

def get_Q_vocab(question_list,vocab_file):
	"""Transform the question_list to a vocabulary. Each single character is a word.
	Args:
		question_list: the question list to be transformed.
	Return:
		the vocabulary.
	"""
	if os.path.isfile(vocab_file):
		print('Get vocabulary from file %s.' % vocab_file)
		vocabulary = {}
		with open(vocab_file,'rt',encoding = 'utf-8') as f:
			for line in f.readlines():
				vocabulary[line.strip()] = len(vocabulary)
		return vocabulary
	else:
		print('Get vocabulary from raw data and write the vocabulary into file %s.' % vocab_file)
		word_list = []
		for question in question_list:
			word_list.extend(list(question))
		return get_vocabulary(word_list, vocab_file)

def sentence2ids(sentence, vocabulary):
	"""Transform a sentence into a list of ids according to the vocabulary.
	Args:
		sentence: the sentence to be handled.
		vocabulary: vocabulary.
	Return:
		a list of ids.
	"""
	words = list(sentence)
	word_list = []
	for word in words:
		word_list.append(vocabulary.get(word))
	return word_list

def get_data(data,training_dir,test_dir,cut_method,ratio):
	"""Prepare the training dataset and test dataset by the ratio (ratio = test/(test+training)).
	Args:
		data: dataset in format of tuple (data,label)
		training_dir: directory of training dataset. Store the training dataset.
		test_dir: directory of test dataset. Store the test dataset.
		ratio: the ratio of test dataset to whole data.
	Return:
		tuple: train and test. Each is of class Dataset.
	"""
	# separate training and test data.
	random.shuffle(data)
	test_size = (int)(len(data) * ratio)
	_test = data[:test_size]
	_train = data[test_size:]

	# write into file.
	if not os.path.exists(training_dir):
		os.mkdir(training_dir)
	if not os.path.exists(test_dir):
		os.mkdir(test_dir)
	training_file = training_dir + cut_method + '_train_' + str(ratio)
	test_file = test_dir + cut_method + '_test_' + str(ratio)

	if os.path.isfile(training_file) and os.path.isfile(test_file):
		print('Get data from file %s and %s.' % (training_file, test_file))
		_data = []
		_label = []
		with open(training_file,'rt',encoding = 'utf-8') as f:
			# tmp = f.read(1)
			# pdb.set_trace()
			for line in f.readlines():
				tmp = line.strip().split('\t')
				_data.append(tmp[0])
				if len(tmp) < 2:
					pdb.set_trace()
				_label.append((int)(tmp[1]))
		train = Dataset(_data,_label)
		_data = []
		_label = []
		with open(test_file,'rt',encoding = 'utf-8') as f:
			for line in f.readlines():
				tmp = line.strip().split('\t')
				_data.append(tmp[0])
				_label.append((int)(tmp[1]))
		test = Dataset(_data,_label)
	else:
		with open(test_file,'wt', encoding = 'utf-8') as f:
			for line in _test:
				f.write(line[0] + '\t' + (str)(line[1]) + '\n')
		with open(training_file,'wt', encoding = 'utf-8') as f:
			for line in _train:
				f.write(line[0] + '\t' + (str)(line[1]) + '\n')

		# transform into dataset form.
		_data = []
		_label = []
		for line in _test:
			_data.append(line[0])
			_label.append(line[1])
		test = Dataset(_data,_label)
		_data = []
		_label = []
		for line in _train:
			_data.append(line[0])
			_label.append(line[1])
		train = Dataset(_data,_label)
	return train, test

def get_LSTM_data(data,training_dir,test_dir,ratio):
	"""Prepare the training dataset and test dataset for LSTM by the ratio (ratio = test/(test+training)).
	Args:
		data: dataset in format of tuple (data,label)
		training_dir: directory of training dataset. Store the training dataset.
		test_dir: directory of test dataset. Store the test dataset.
		ratio: the ratio of test dataset to whole data.
	Return:
		tuple: train and test. Each is of class Dataset.
	"""
	# separate training and test data.
	random.shuffle(data)
	test_size = (int)(len(data) * ratio)
	_test = data[:test_size]
	_train = data[test_size:]

	# write into file.
	if not os.path.exists(training_dir):
		os.mkdir(training_dir)
	if not os.path.exists(test_dir):
		os.mkdir(test_dir)
	training_file = training_dir + 'LSTM_train_' + str(ratio)
	test_file = test_dir + 'LSTM_test_' + str(ratio)

	if os.path.isfile(training_file) and os.path.isfile(test_file):
		print('Get data from file %s and %s.' % (training_file, test_file))
		_data = []
		_label = []
		with open(training_file,'rt',encoding = 'utf-8') as f:
			# tmp = f.read(1)
			# pdb.set_trace()
			for line in f.readlines():
				tmp = line.strip().split('\t')
				_data.append((int)(x) for x in tmp[0].strip().split())
				if len(tmp) < 2:
					pdb.set_trace()
				_label.append((int)(tmp[1]))
		train = Dataset(_data,_label)
		_data = []
		_label = []
		with open(test_file,'rt',encoding = 'utf-8') as f:
			for line in f.readlines():
				tmp = line.strip().split('\t')
				_data.append((int)(x) for x in tmp[0].strip().split())
				_label.append((int)(tmp[1]))
		test = Dataset(_data,_label)
	else:
		with open(test_file,'wt', encoding = 'utf-8') as f:
			for line in _test:
				f.write(' '.join(line[0]) + '\t' + (str)(line[1]) + '\n')
		with open(training_file,'wt', encoding = 'utf-8') as f:
			for line in _train:
				f.write(' '.join(line[0]) + '\t' + (str)(line[1]) + '\n')

		# transform into dataset form.
		_data = []
		_label = []
		for line in _test:
			_data.append(line[0])
			_label.append(line[1])
		test = Dataset(_data,_label)
		_data = []
		_label = []
		for line in _train:
			_data.append(line[0])
			_label.append(line[1])
		train = Dataset(_data,_label)
	return train, test

def prepare_data(raw_file, tar_dir, cut_method, vocab_dir):
	"""Handle with the raw dataset, (cut sentences and) transform the label into ids.
	Args:
		raw_file: file name of the raw data.
		tar_dir: directory of target data.
		cut_method: how to cut sentences into words, e.g. by jieba | by n-gram | by character
	Return:
		dataset: (data(cut),label(id))
	"""
	tar_data_file = tar_dir + 'prepared_data_' + cut_method
	tar_label_file = tar_dir + 'prepared_label'
	prepared_data = []
	prepared_label = []
	if os.path.isfile(tar_data_file) and os.path.isfile(tar_label_file):
		print('Prepare data from file %s and %s.' % (tar_data_file, tar_label_file))
		with open(tar_data_file, encoding = 'utf-8') as f:
			# tmp = f.read(1) # first byte of the data ----'\ufeff',this character is the BOM or "Byte Order Mark".      
			# pdb.set_trace()
			# prepared_data.extend(f.readlines())
			prepared_data = [line.strip() for line in f.readlines()]
		with open(tar_label_file, encoding = 'utf-8') as f:
			for line in f.readlines():
				prepared_label.append((int)(line.strip()))
				# prepared_label = [(int)]
		return prepared_data,prepared_label
	else:
		print('Prepare data from raw file %s.' % raw_file)
		raw_data,raw_label = read_in_raw(raw_file)
		for sentence in raw_data:
			prepared_data.append(cut_sentence(sentence,cut_method))

		label_vocabulary = get_vocabulary(raw_label,vocab_dir)
		for label in raw_label:
			prepared_label.append(label_vocabulary.get(label))
		# write into file.

		with open(tar_data_file,'wt',encoding = 'utf-8') as f:
			f.writelines('\n'.join(prepared_data))
		with open(tar_label_file,'wt',encoding = 'utf-8') as f:
			for label in prepared_label:
				f.write((str)(label) + '\n')
		return prepared_data,prepared_label

def prepare_LSTM_data(raw_file,tar_dir,vocab_dir):
	"""Prepare data for LSTM model.
	Args:
		raw_file: file name of raw data.
		tar_dir: dir of prepared data, including question data and label file.
		vocab_dir: dir of vocabulary, including vocabulary for question and label.
	Return:
		prepared_data, prepared_label
	"""
	tar_data_file = tar_dir + 'prepared_data_LSTM'
	tar_label_file = tar_dir + 'prepared_label_LSTM'
	prepared_data = []
	prepared_label = []
	if os.path.isfile(tar_data_file) and os.path.isfile(tar_label_file):
		print('Prepare data from file %s and %s.' % (tar_data_file, tar_label_file))
		with open(tar_data_file, encoding = 'utf-8') as f:
			for line in f.readlines():
				prepared_data.append((int)(x) for x in line.strip().split())
		with open(tar_label_file, encoding = 'utf-8') as f:
			for line in f.readlines():
				prepared_label.append((int)(line.strip()))
				# prepared_label = [(int)]
		return prepared_data,prepared_label
	else:
		print('Prepare data from raw file %s.' % raw_file)
		raw_data,raw_label = read_in_raw(raw_file)
		q_vocabulary = get_Q_vocab(raw_data, vocab_dir + '_question')
		for sentence in raw_data:
			prepared_data.append(sentence2ids(sentence, q_vocabulary))

		label_vocabulary = get_vocabulary(raw_label,vocab_dir + '_label')
		for label in raw_label:
			prepared_label.append(label_vocabulary[label])

		# write into file.
		with open(tar_data_file,'wt',encoding = 'utf-8') as f:
			for piece in prepared_data:
				f.writelines(' '.join(piece) + '\n')
		with open(tar_label_file,'wt',encoding = 'utf-8') as f:
			for label in prepared_label:
				f.write((str)(label) + '\n')
		return prepared_data,prepared_label

def read_in_raw(raw_file):
	"""Read in raw dataset.
	Args:
		raw_file: file name of the raw data.
	Return:
		dataset: raw data in format (data, label)
	Raises:
		ValueError: data file does not exist.
	"""
	if os.path.isfile(raw_file):
		data = []
		label = []
		with open(raw_file,'rt',encoding = 'utf-8') as f:
			tmp = f.read(1) # first byte of the data ----'\ufeff',this character is the BOM or "Byte Order Mark".      
			# pdb.set_trace()
			for line in f.readlines():
				tmp = line.strip().split('\t')
				# pdb.set_trace()
				data.append(tmp[0])
				label.append(tmp[1])
		return data, label
	else:
		raise ValueError("File %s does not exists." % raw_file)