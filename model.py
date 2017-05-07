# -*-coding:utf-8 -*-
"""
Created on April 1st 2017

@author: ML Hou
"""
import sys
import os
import pdb

import data_utils

from sklearn import *
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer
from sklearn import decomposition 
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.pipeline import Pipeline 
from sklearn.cross_validation import cross_val_score, KFold

from scipy.stats import sem
import numpy as np



def evaluate(target,preds):
	"""Evaluate the result.
	Args:
		target: the target label.
		preds: predicted label.
	Return:
		accuracy:
	"""
	acc_count = 0
	for i in range(len(preds)):
		if preds[i] == target[i]:
			acc_count += 1
	acc = (float)(acc_count) / len(preds)
	print('Accuracy:%.3f --- %d / %d' % (acc, acc_count, len(preds)))
	return acc

def calculate_result(model,test):  
	pred = model.predict(test.data)
	m_precision = metrics.precision_score(test.label,pred)
	m_recall = metrics.recall_score(test.label,pred) 
	print('predict info:\n') 
	print('precision:{0:%.3f}' % m_precision)  
	print('recall:{0:%.3f}' % m_recall)
	print('f1-score:{0:%.3f}'%metrics.f1_score(test.label,pred))
	# confusion_matrix = metrics.confusion_matrix(actual,pred)
	# print ("Confusion matrix:\n%s" % confusion_matrix)
	# pre = np.sum(confusion_matrix,axis = 1)
	# re = np.sum(confusion_matrix,axis = 0)
	# class_pre = []
	# class_re = []
	# for i in range(len(confusion_matrix)):
	# 	class_pre.append((float)(confusion_matrix[i][i]) / pre[i])
	# 	class_re.append((float)(confusion_matrix[i][i]) / re[i])
	# print ("class precision: \n%s" % class_pre)
	# print ("class recall: \n%s" % class_re)

class Bayes(object):
	def __init__(self, config_dir):
		"""Initialize a bayes model.
		Args:
			alpha: parameter for bayes model.
		"""
		self.name = 'Bayes'
		self.config_dir = config_dir
		self.config = dict()
		self.Vec = None
		self.clf = None

		self.output_path = './result/Bayes/'
		if not os.path.exists(self.output_path):
			os.mkdir(self.output_path)
		self.model_path = './model/Bayes/'
		if not os.path.exists(self.model_path):
			os.mkdir(self.model_path)

	def get_config(self):
		"""Get the config from the config file. Put them into the config dictionary.
		Raises:
			ValueError: the config file does not exists.
		"""
		if os.path.exists(self.config_dir):
			print('Read in configuration from dir %s.\n' % self.config_dir)
			with open(self.config_dir + 'Bayes_config','rt') as f:
				for line in f.readlines():
					tmp = line.strip().split(':')
					self.config[tmp[0]] = tmp[1]
			for key,value in self.config.items():
				print('%s:%s' % (key, value))

			print('Build up the model according to configuration.')
			if self.config.get('alpha') == None:
				raise ValueError('Parameter Alpha has not been set. Please re-edit the configuration file.')
			# Build up the feature vector.
			if self.config.get('feature') == None:
				raise ValueError('Feature parameter has not been set. Please re-edit the configuration file.')
			else:
				if self.config['feature'] == 'CountVectorizer':
					self.Vec = CountVectorizer()
				elif self.config['feature'] == 'TfidfVectorizer':
					self.Vec = TfidfVectorizer()
				elif self.config['feature'] == 'HashingVectorizer':
					self.Vec = HashingVectorizer()
				else:
					raise ValueError('Can not use %s as a feature, please re-edit your configuration file.' % self.config['feature'])

			# Build up model.
			if self.config.get('model') == None:
				raise ValueError('Model parameter has not been set. Please re-edit the configuration file.')
			else:
				if self.config['model']  == 'GaussianNB':
					self.clf = GaussianNB()
				elif self.config['model'] == 'MultinomialNB':
					self.clf = MultinomialNB(alpha = (float)(self.config['alpha']))
				elif self.config['model'] == 'BernoulliNB':
					self.clf = BernoulliNB()
				else:
					raise ValueError('No model named %s, please re-edit your configuration file.' % self.config['model'])

		else:
			print('Configuration file %s does not exists.' % self.config_dir)

	def fit(self,train):
		"""Fit the data into the model and train.
		Args:
			train: training data in format (data,label)
		Return:
			None
		Raises:
			ValueError: invalid config value.
		"""
		fea = self.Vec.fit_transform(train.data)
		fea = fea.todense()
		self.clf.fit(fea,train.target)


	def predict(self,test_data):
		"""Run the model for a single step to get the predicted result.
		Args:
			data_piece: a piece of data fit into the model.
		Return:
			predicted result.
		"""
		fea  = self.Vec.transform(test_data)
		fea = fea.todense()
		return self.clf.predict(fea)

	def test(self, test):
		"""Test the model.
		Args:
			model: the model to be evaluated.
			test: test dataset in format (data,label)
		Output:
			result: predicted label. A label each line.
		Return:
			accuracy:
		"""
		preds = self.predict(test.data)
		# Write into files.
		filename = self.output_path + self.config['model'] + '_' + self.config['feature']
		with open(filename, 'wt', encoding = 'utf-8') as f:
			for pred in preds:
				f.write(str(pred) + '\n')
		return evaluate(test.target, preds)
