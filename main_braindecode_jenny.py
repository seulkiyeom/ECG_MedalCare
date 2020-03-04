# -*- encoding: utf-8 -*-
'''
@Research: ECG classification using CNN
@developer: Seul-Ki Yeom (from TUB)
@dataset: PTB diagnostic 50 healthy and 50 MI from Jenny (NPL)
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='ECG classification')
# parser.add_argument('--subj-idx', type=int, default=9, metavar='N', help='Subject Number')
parser.add_argument('--deep', action='store_true', help='DeepNet vs. ShallowNet inspired by FBCSP')
# parser.add_argument('--deep', type=bool, default=False, metavar='N',
#                     help='DeepNet vs. ShallowNet inspired by FBCSP')
parser.add_argument('--lr-wd', type=int, default=1, metavar='N', help='Learning rate and Weight decay type')
args = parser.parse_args()

############################################### Parameters
n_epochs = 50
batch_size = 30
num_folds = 5
n_classes = 2
in_chans = 1  # train_set.X.shape[1]  # number of channels = 12
input_time_length = 3000  # length of time of each epoch/trial = 4000
fs = 1000 #sampling frequency
band = [0.5, 150]
###############################################

lr = {
    0: 0.0625 * 0.01,
    1: 0.0625 * 0.01,
    2: 0.0625 * 0.01,
}[args.lr_wd]

wd = {
    0: 0.5 * 0.001,
    1: 0.5 * 0.001,
    2: 0.5 * 0.001,
}[args.lr_wd]

#Measurements
Train_Losses = np.zeros((num_folds, n_epochs))
Accuracies = np.zeros((num_folds, n_epochs))
Test_Losses = np.zeros((num_folds, n_epochs))
Test_Length = np.zeros((num_folds, n_epochs))
Pred_Labels = np.zeros((num_folds, n_epochs, 100))
Test_Labels = np.zeros((num_folds, n_epochs, 100))

# Set if you want to use GPU
cuda = torch.cuda.is_available()
set_random_seeds(seed=20170631, cuda=cuda)

################### Load data
idx_healthy, idx_patient, CV_healthy_train, CV_healthy_test, CV_patient_train, CV_patient_test = dataset.get_PTB_new(n_splits=num_folds)

for CV in range(num_folds): #Cross-validation
    if args.deep:
        save_path = 'ECG_model_deep_' + str(CV) + '.pt'
        # save_path = 'ECG_model_deep_' + str(args.lr_wd) + '.pt'
        var_save_path = 'ECG_deep_' + str(CV) + '.tar'  # type: str
    else:
        save_path = 'ECG_model_shallow_' + str(CV) + '.pt'
        var_save_path = 'ECG_shallow_' + str(CV) + '.tar'  # type: str

    if args.deep:
        model = Deep4Net(in_chans=in_chans, n_classes=n_classes, input_time_length=input_time_length, final_conv_length='auto').create_network()
    else:
        model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes, input_time_length=input_time_length, final_conv_length='auto').create_network() #Cropped version

    to_dense_prediction_model(model) #for cropped verision (integrate all the predicted output into one)

    if cuda:
        model.cuda()

    ##################Create cropped iterator (only for croppted training strategy !!!!)
    # determine output size
    test_input = np_to_var(np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    print("{:d} predictions per input/trial".format(n_preds_per_input))

    iterator = CropsFromTrialsIterator(batch_size=batch_size, input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    train_data_path = np.concatenate((CV_healthy_train[CV], CV_patient_train[CV]))
    test_data_path = np.concatenate((CV_healthy_test[CV], CV_patient_test[CV]))

    #Class label information - Healthy: 0, Patient: 1
    train_label = np.concatenate((np.repeat(0, len(CV_healthy_train[CV])), np.repeat(1, len(CV_patient_train[CV]))))
    test_label = np.concatenate((np.repeat(0, len(CV_healthy_test[CV])), np.repeat(1, len(CV_patient_test[CV]))))

    combined_1 = list(zip(train_data_path, train_label))
    shuffle(combined_1) #shuffle
    train_data_path[:], train_label[:] = zip(*combined_1)

    # FeatVect, y_labels = dataset.Epochs(train_data_path, train_label, time_segment = 3000)
    FeatVect, y_labels = dataset.Epochs_mid(train_data_path, train_label, time_segment = 3000)
    # FeatVect, y_labels = dataset.Epochs(train_data_path[31:52], train_label[31:52], time_segment=3000)

    # preprocessing
    for ii in range(0, FeatVect.shape[0]):
        # 1. Data reconstruction
        temp_data = FeatVect[ii, :, :]
        temp_data = temp_data.transpose()
        # 2. Lowpass filtering
        lowpassed_data = lowpass_cnt(temp_data, band[1], fs, filt_order=3)
        # 3. Highpass filtering
        bandpassed_data = highpass_cnt(lowpassed_data, band[0], fs, filt_order=3)
        # 4. Exponential running standardization
        ExpRunStand_data = exponential_running_standardize(bandpassed_data, factor_new=0.001, init_block_size=None,
                                                           eps=0.0001)
        # 5. Renewal preprocessed data
        ExpRunStand_data = ExpRunStand_data.transpose()
        FeatVect[ii, :, :] = ExpRunStand_data
        del temp_data, lowpassed_data, bandpassed_data, ExpRunStand_data

    X = FeatVect.astype(np.float32)
    X = X[:, :, :]
    y = (y_labels).astype(np.int64)

    train_set = SignalAndTarget(X, y=y)
    del X, y, FeatVect, y_labels

    ### Training and Test
    for i_epoch in range(n_epochs):  # Need to change
        print("Epoch: {}".format(i_epoch))
        ###################Training
        if i_epoch != 0:
            model.load_state_dict(torch.load(save_path))

        ###################Define optimizer
        rng = RandomState((2018, 8, 7))
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)  # these are good values for the deep model
        # Need to determine number of batch passes per epoch for cosine annealing
        # n_updates_per_epoch = len([None for b in iterator.get_batches(train_set, True)])

        n_updates_per_epoch = sum(
            [1 for _ in iterator.get_batches(train_set, shuffle=True)])
        scheduler = CosineAnnealing(n_epochs * n_updates_per_epoch)
        # schedule_weight_decay must be True for AdamW
        optimizer = ScheduledOptimizer(scheduler, optimizer, schedule_weight_decay=True)

        train_loss = 0
        test_loss = 0
        accuracy = 0

        ###################Training
        model.train()
        all_preds = []
        all_losses = []
        batch_sizes = []
        for batch_X, batch_y in iterator.get_batches(train_set, shuffle=True):
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            # Remove gradients of last backward pass from all parameters
            optimizer.zero_grad()
            outputs = model(net_in)
            # Mean predictions across trial
            # Note that this will give identical gradients to computing
            # a per-prediction loss (at least for the combination of log softmax activation
            # and negative log likelihood loss which we are using here)
            outputs = torch.mean(outputs, dim=2, keepdim=False)

            loss = F.nll_loss(outputs, net_target)
            loss.backward()
            optimizer.step()
            # print('Training loss: {}'.format(loss.item()))
            loss = float(var_to_np(loss))
            all_losses.append(loss)
            batch_sizes.append(len(batch_X))
        # Compute mean per-input loss
        train_loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                              np.mean(batch_sizes))
        print('Training loss: {}'.format(train_loss))

        torch.save(model.state_dict(), save_path)  # save pre-trained model

        ###################Test
        pred_labels = []
        test_labels = []
        model.eval()
        info = io.loadmat('PTBdb_30s_window_indices_new.mat')['info']
        for i in range(len(test_data_path)):  # Test for each test subject
            #### Step1: Load test data
            dat_path = os.listdir(test_data_path[i])

            # if len(dat_path) is not 1:  # use the first session only !!
                # print("Multiple session No. {}".format(test_data_path[i][-3:]))

            print('Load test data: Subject ID. {}'.format(test_data_path[i][-10:]))
            # print('subject: {}, data: {}'.format(list(info[:, 0]).index(data_path[i][-10:]), list(info[:, 1]).index(dat_path[0][:-4])))
            start = info[list(info[:, 1]).index(dat_path[0][:-4]), 2]
            end = info[list(info[:, 1]).index(dat_path[0][:-4]), 3]

            data = io.loadmat(test_data_path[i] + '/' + dat_path[0])
            ECG_unfiltered = data['ECG_unfiltered']  # Raw ECG
            # ECG_unfiltered = data['ECG_filtered']  # Processed ECG
            # FPT_x = data['FPT_x']
            # FPT_y = data['FPT_y']

            ECG_unfiltered = ECG_unfiltered.transpose()
            FeatVect = np.expand_dims(np.expand_dims(ECG_unfiltered[1, int(start):int(end)], 0), 0)
            y_labels = np.array([test_label[i]])

            # preprocessing
            for ii in range(0, FeatVect.shape[0]):
                # 1. Data reconstruction
                temp_data = FeatVect[ii, :, :]
                temp_data = temp_data.transpose()
                # 2. Lowpass filtering
                lowpassed_data = lowpass_cnt(temp_data, band[1], fs, filt_order=3)
                # 3. Highpass filtering
                bandpassed_data = highpass_cnt(lowpassed_data, band[0], fs, filt_order=3)
                # 4. Exponential running standardization
                ExpRunStand_data = exponential_running_standardize(bandpassed_data, factor_new=0.001,
                                                                   init_block_size=None,
                                                                   eps=0.0001)
                # 5. Renewal preprocessed data
                ExpRunStand_data = ExpRunStand_data.transpose()
                FeatVect[ii, :, :] = ExpRunStand_data
                del temp_data, lowpassed_data, bandpassed_data, ExpRunStand_data

            # Convert data to Braindecode format
            X = FeatVect.astype(np.float32)
            X = X[:, :, :]
            y = (y_labels).astype(np.int64)
            test_set = SignalAndTarget(X, y=y)
            del X, y, ECG_unfiltered, FeatVect, y_labels, data

            # Collect all predictions and losses
            all_preds = []
            all_losses = []
            batch_sizes = []
            #### Step2: Test
            for batch_X, batch_y in iterator.get_batches(test_set, shuffle=False):
                net_in = np_to_var(batch_X)
                if cuda:
                    net_in = net_in.cuda()
                net_target = np_to_var(batch_y)
                if cuda:
                    net_target = net_target.cuda()
                outputs = model(net_in)
                all_preds.append(var_to_np(outputs))
                outputs = torch.mean(outputs, dim=2, keepdim=False)
                loss = F.nll_loss(outputs, net_target)
                loss = float(var_to_np(loss))
                all_losses.append(loss)
                batch_sizes.append(len(batch_X))

            # Compute mean per-input loss
            test_loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                                np.mean(batch_sizes))
            # print("Subject ID: {}, Test Loss: {:.5f}".format(test_data_path[i][40:], test_loss))
            # Assign the predictions to the trials
            preds_per_trial = compute_preds_per_trial_from_crops(all_preds,
                                                                 input_time_length,
                                                                 test_set.X)

            # preds per trial are now trials x classes x timesteps/predictions
            # Now mean across timesteps for each trial to get per-trial predictions
            meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
            predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
            # print('Predict Label: {}, True Label: {}'.format(predicted_labels, test_set.y))
            accuracy += np.mean(predicted_labels == test_set.y)
            pred_labels = np.append(pred_labels, predicted_labels)
            test_labels = np.append(test_labels, test_set.y)

            del test_set

        print("Test Accuracy: {:.1f}%".format((accuracy / len(test_data_path)) * 100))


        Pred_Labels[CV, i_epoch, :len(pred_labels)] = pred_labels
        Test_Labels[CV, i_epoch, :len(test_labels)] = test_labels
        Accuracies[CV, i_epoch] = accuracy
        Test_Length[CV, i_epoch] = len(test_data_path)
        Train_Losses[CV, i_epoch] = train_loss
        Test_Losses[CV, i_epoch] = test_loss

    torch.save({'Acc': Accuracies, 'Train_Losses': Train_Losses, 'Test_Losses': Test_Losses, 'Test_Length': Test_Length,
                'Test_label': Test_Labels, 'Pred_label': Pred_Labels,
                'n_epochs': n_epochs, 'in_chans': in_chans, 'num_folds': num_folds, 'n_classes': n_classes,
                'class_label': {'Healthy_0', 'Patient_1'},
                'input_time_length': input_time_length, 'lr': lr, 'wd': wd, 'train_group': train_data_path, 'test_group': test_data_path}, var_save_path)

    del model

print('Best epoch: {}, Best Accuracy: {}'.format(np.argmax(np.mean(Accuracies, 0)), np.mean(Accuracies, 0).max()))


# num_folds = 5
# i = 0
# # for i in range(num_folds):
# path = 'ECG_deep_' + str(i) + '.tar' #lr = 0.0625 * 0.01, weight decay = 0.5 * 0.001
# path_all = []
# path_all.append(torch.load(path))
# # path_all.append(torch.load(subjs_path1))
# for i in range(len(path_all)):
#     print('Accuracy: {} \n Test Length: {} \n Training loss: {} \n Fine-tuning loss: {} \n lr: {} \n wd: {}'.format(
#         path_all[i]['Acc'][0, :].max(), path_all[i]['Test_Length'][0, :].max(), path_all[i]['Train_Losses'][i, :].min(),
#         path_all[i]['Test_Losses'][0, :].min(), path_all[i]['lr'], path_all[i]['wd']))