# -*- coding:utf-8 -*-
"""
Created on April 5th 2017

@author: ML Hou
"""

"""Utilities for handling with data: tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np

import data_utils

import sys
import os
import pdb

# Special symbol
_PAD = '_PAD'
_UNK = '_UNK'
_GO = '_GO'
_EOS = '_EOS'
START_VOCAB = [_PAD, _UNK, _GO, _EOS]

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d") 

