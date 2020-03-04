# -*- encoding: utf-8 -*-
'''
@Research: ECG classification using CNN
@developer: Seul-Ki Yeom (from TUB)
'''

import numpy as np
from scipy import io
from numpy.random import RandomState
import sys
import re
import os
import glob
import argparse
import torch.nn.functional as F
from torch import nn
import torch
from random import shuffle

from braindecode.models.shallow_fbcsp import ShallowFBCSPNet #Shallow CNN inspired by FBCSP
from braindecode.models.deep4 import Deep4Net #Deep CNN
from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.signalproc import lowpass_cnt, highpass_cnt, exponential_running_standardize
from braindecode.datautil.iterators import CropsFromTrialsIterator, get_balanced_batches
from braindecode.datautil.splitters import split_into_two_sets, split_into_train_test
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds, np_to_var, var_to_np
from braindecode.torch_ext.schedulers import ScheduledOptimizer, CosineAnnealing
from braindecode.experiments.monitors import compute_preds_per_trial_from_crops

import modules.data as dataset
import sklearn.metrics as metric

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='ECG classification')
# parser.add_argument('--subj-idx', type=int, default=9, metavar='N', help='Subject Number')
parser.add_argument('--deep', action='store_true', help='DeepNet vs. ShallowNet inspired by FBCSP')
# parser.add_argument('--deep', type=bool, default=False, metavar='N',
#                     help='DeepNet vs. ShallowNet inspired by FBCSP')
parser.add_argument('--lr-wd', type=int, default=1, metavar='N', help='Learning rate and Weight decay type')
args = parser.parse_args()

path = 'data/ECG_deep_jenny_ver2.tar' #lr = 0.0625 * 0.01, weight decay = 0.5 * 0.001
data = torch.load(path)
# path_all = []
# path_all.append(torch.load(path))

Accuracies = data['Acc']#
Train_Losses = data['Train_Losses']
Test_Losses = data['Test_Losses']
Test_Length = data['Test_Length']#
Test_Labels = data['Test_label']#
Pred_Labels = data['Pred_label']#
n_epochs = data['n_epochs']
in_chans = data['in_chans']
num_folds = data['num_folds']
n_classes = data['n_classes']
label_name = data['class_label']
input_time_length = data['input_time_length']
lr = data['lr']
wd = data['wd']
train_data_path = data['train_group']
test_data_path = data['test_group']

hit = 0
length = 0
pred_label = []
true_label = []
for i in range(num_folds):
    max_idx = np.argmax(Accuracies[i,:])
    max_acc = Accuracies[i, max_idx]
    max_length = Test_Length[i, max_idx]
    max_pred_label = Pred_Labels[i, max_idx, :int(Test_Length[i, max_idx])]
    max_true_label = Test_Labels[i, max_idx, :int(Test_Length[i, max_idx])]
    hit += max_acc
    length += max_length
    pred_label = np.append(pred_label, max_pred_label)
    true_label = np.append(true_label, max_true_label)

    print('Fold No.: {} \nBest Epoch No.: {} \nAccuracy: {}\n'.format(
        str(i), str(max_idx), max_acc / max_length * 100))

target_name = ['Healthy', 'Patient']

# print(metric.classification_report(true_label, pred_label, target_names=target_name))

print('Accuracy: {} \n'.format(hit / length * 100)) #Accuracy
print('Specificity: {}\n'.format(metric.recall_score(true_label, pred_label))) #specificity
print('Precision: {}\n'.format(metric.precision_score(true_label, pred_label)))
print('Recall: {}\n'.format(metric.recall_score(true_label, pred_label)))
print('F1 score: {}\n'.format(metric.f1_score(true_label, pred_label)))
print('AUC: {}\n'.format(metric.roc_auc_score(true_label, pred_label)))