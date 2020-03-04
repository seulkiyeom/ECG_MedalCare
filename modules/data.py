'''
Codes for loading the MNIST data
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import os, fnmatch
import numpy as np
import torch
from scipy.io import loadmat
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import imageio
import pandas as pd
from functools import lru_cache
import wfdb
import matplotlib.pyplot as plt

def get_mit_bih(datapath='../data/ECG/'):
    '''
    Load MIT-BIH dataset (from https://github.com/mikkelhartmann/ecg-analysis-mit-bih)
    class: "Normal" or "Abnormal"
    '''
    #Downloading the database
    if os.path.isdir(datapath + 'mitdb'):
        print('You already have the data.')
    else:
        wfdb.dl_database(datapath + 'mitdb', datapath + 'mitdb')

    #Reading a record
    record = wfdb.rdsamp(datapath + 'mitdb/100', sampto=3000)
    annotation = wfdb.rdann(datapath + 'mitdb/100', 'atr', sampto=3000)

    record[1]

    #Plotting a record
    fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
    I = record[0][:, 0]
    II = record[0][:, 1]

    ax[0].plot(I)
    ax[1].plot(II)
    ax[0].set_ylabel('Lead I')
    ax[1].set_xlabel('Datapoints')
    ax[1].set_ylabel('Lead II')
    plt.show()



def get_cifar10(datapath='../data/', download=True):
    '''
    Get CIFAR10 dataset
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Cifar-10 Dataset
    train_dataset = datasets.CIFAR10(root=datapath,
                                     train=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize
                                     ]),
                                     download=download)

    test_dataset = datasets.CIFAR10(root=datapath,
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    return train_dataset, test_dataset

def get_CinC(datapath='../data/ECG/', n_splits=5):
    if not os.path.isdir(datapath + 'Training_WFDB'):
        print("No pathway")
    else:
        print("Loading CinC Challenge Diagnostic ECG dataset")

    count = 0
    data_list = []
    label = np.zeros((6877, 9)) #total number of samples x total number of classes
    label_idx = np.zeros((6877))  # total number of samples x total number of classes
    # label = np.zeros((47, 9))  # total number of samples x total number of classes
    # label_idx = np.zeros((47))  # total number of samples x total number of classes

    # for filename in sorted(os.listdir(datapath + 'Training_WFDB')[:100]): #in case of 47, 9
    for filename in sorted(os.listdir(datapath + 'Training_WFDB')):
        ext = os.path.splitext(filename)
        if ext[-1] == '.mat':
            record = wfdb.rdsamp(datapath + 'Training_WFDB/' + ext[0])
            data_list.append(filename)
            # print('subject No.: {}, data size: {}\n'.format(ext[0], record[0].shape))
            # print('class: {}'.format(record[1]['comments'][2]))

            if 'Normal' in record[1]['comments'][2]: #label 0
                label[count, 0] = 1.0
                label_idx[count] = 0
            if 'AF' in record[1]['comments'][2]: #label 1
                label[count, 1] = 1.0
                label_idx[count] = 1
            if 'I-AVB' in record[1]['comments'][2]: #label 2
                label[count, 2] = 1.0
                label_idx[count] = 2
            if 'LBBB' in record[1]['comments'][2]: #label 3
                label[count, 3] = 1.0
                label_idx[count] = 3
            if 'RBBB' in record[1]['comments'][2]: #label 4
                label[count, 4] = 1.0
                label_idx[count] = 4
            if 'PAC' in record[1]['comments'][2]: #label 5
                label[count, 5] = 1.0
                label_idx[count] = 5
            if 'PVC' in record[1]['comments'][2]: #label 6
                label[count, 6] = 1.0
                label_idx[count] = 6
            if 'STD' in record[1]['comments'][2]: #label 7
                label[count, 7] = 1.0
                label_idx[count] = 7
            if 'STE' in record[1]['comments'][2]: #label 8
                label[count, 8] = 1.0
                label_idx[count] = 8

            count += 1

    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(range(len(label)))

    CV_train = []
    CV_test = []
    CV_train_label = []
    CV_test_label = []
    CV_train_label_idx = []
    CV_test_label_idx = []
    for train_idx, test_idx in kf.split(range(len(label))):
        CV_train.append(np.array(data_list)[train_idx])
        CV_train_label.append(label[train_idx,:])
        CV_train_label_idx.append(label_idx[train_idx])
        CV_test.append(np.array(data_list)[test_idx])
        CV_test_label.append(label[test_idx, :])
        CV_test_label_idx.append(label_idx[test_idx])

    return CV_train, CV_test, CV_train_label, CV_test_label, CV_train_label_idx, CV_test_label_idx


def get_PTB(datapath='../data/ECG/', n_splits=5):
    '''
    Load PTB Diagnostic ECG dataset (available on physionet.org)


    There are two folders,
    1: the healthy control group
    2: the myocardial infarction patients group, respectively.

    A subfolder exists for each patient belonging to either the healthy control group or to the MI patients.
    Each subfolder contains a *.mat file with:
    - ECG_unfiltered (Nx12 matrix): Raw ECG, time sample point * channel
    - ECG_filtered (Nx12 matrix): Processed ECG (Baseline removal, Frequency filtering, and Isoline correction)
    - FPT_x: 12x1 cell array. Each cell entry belongs to the FPT table of one ECG channel. The FPT table is a Mx9 matrix, whereas M represents the number of heartbeats present in the lead. The 9 columns represent the timestamp (in ms) of the following fiducial points: 1: P-wave onset, 2: P-wave peak, 3: P-wave offset, 4: QRS onset, 5: QRS peak, 6: QRS offset, 7: T-wave onset, 8: T-wave peak, 9: T-wave offset
    - FPT_y: 12x1 cell array. Each cell entry belongs to the FPT table of one ECG channel. See above. The entries of the FPT_y table represent the amplitudes of the respective ECG trace at the fiducial points in FPT_x. Therefore: FPT_y = ECG_filtered(FPT_x) holds.
    '''
    if not os.path.isdir(datapath + 'PTB_Diagnostic_ECG_Database'):
        print("No pathway")
    else:
        print("Loading PTB Diagnostic ECG dataset")

    category = ['Processed_ECGs_healthy', 'Processed_ECGs_MI']

    idx_healthy = [x[0] for x in os.walk(datapath + 'PTB_Diagnostic_ECG_Database/' + category[0]) if x[0]][1:]
    idx_healthy = [c for c in idx_healthy]

    idx_patient = [x[0] for x in os.walk(datapath + 'PTB_Diagnostic_ECG_Database/' + category[1]) if x[0]][1:]
    idx_patient = [c for c in idx_patient]

    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(range(len(idx_healthy)))

    CV_healthy_train = []
    CV_healthy_test = []
    for train_idx, test_idx in kf.split(range(len(idx_healthy))):
        CV_healthy_train.append(np.array(idx_healthy)[train_idx])
        CV_healthy_test.append(np.array(idx_healthy)[test_idx])

    CV_patient_train = []
    CV_patient_test = []
    for train_idx, test_idx in kf.split(range(len(idx_patient))):
        CV_patient_train.append(np.array(idx_patient)[train_idx])
        CV_patient_test.append(np.array(idx_patient)[test_idx])
    return idx_healthy, idx_patient, CV_healthy_train, CV_healthy_test, CV_patient_train, CV_patient_test

def get_PTB_new(datapath='../data/ECG/', n_splits=5):
    '''
    Load PTB Diagnostic ECG dataset (available on physionet.org)


    There are two folders,
    1: the healthy control group
    2: the myocardial infarction patients group, respectively.

    A subfolder exists for each patient belonging to either the healthy control group or to the MI patients.
    Each subfolder contains a *.mat file with:
    - ECG_unfiltered (Nx12 matrix): Raw ECG, time sample point * channel
    - ECG_filtered (Nx12 matrix): Processed ECG (Baseline removal, Frequency filtering, and Isoline correction)
    - FPT_x: 12x1 cell array. Each cell entry belongs to the FPT table of one ECG channel. The FPT table is a Mx9 matrix, whereas M represents the number of heartbeats present in the lead. The 9 columns represent the timestamp (in ms) of the following fiducial points: 1: P-wave onset, 2: P-wave peak, 3: P-wave offset, 4: QRS onset, 5: QRS peak, 6: QRS offset, 7: T-wave onset, 8: T-wave peak, 9: T-wave offset
    - FPT_y: 12x1 cell array. Each cell entry belongs to the FPT table of one ECG channel. See above. The entries of the FPT_y table represent the amplitudes of the respective ECG trace at the fiducial points in FPT_x. Therefore: FPT_y = ECG_filtered(FPT_x) holds.
    '''
    if not os.path.isdir(datapath + 'PTB_Diagnostic_ECG_Database'):
        print("No pathway")
    else:
        print("Loading PTB Diagnostic ECG dataset")

    category = ['Processed_ECGs_healthy', 'Processed_ECGs_MI']

    idx_healthy = [x[0] for x in os.walk(datapath + 'PTB_Diagnostic_ECG_Database_from_Jenny/' + category[0]) if x[0]][1:]
    idx_healthy = [c for c in idx_healthy]

    idx_patient = [x[0] for x in os.walk(datapath + 'PTB_Diagnostic_ECG_Database_from_Jenny/' + category[1]) if x[0]][1:]
    idx_patient = [c for c in idx_patient]

    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(range(len(idx_healthy)))

    CV_healthy_train = []
    CV_healthy_test = []
    for train_idx, test_idx in kf.split(range(len(idx_healthy))):
        CV_healthy_train.append(np.array(idx_healthy)[train_idx])
        CV_healthy_test.append(np.array(idx_healthy)[test_idx])

    CV_patient_train = []
    CV_patient_test = []
    for train_idx, test_idx in kf.split(range(len(idx_patient))):
        CV_patient_train.append(np.array(idx_patient)[train_idx])
        CV_patient_test.append(np.array(idx_patient)[test_idx])
    return idx_healthy, idx_patient, CV_healthy_train, CV_healthy_test, CV_patient_train, CV_patient_test

def Epochs(data_path, label, time_segment = 1000, time_step = 250):
    FeatVect = np.array([])
    y_labels = np.array([])

    count = 0

    for i in range(len(data_path)): #iterative data loading
    # for i in range(len(data_path)-155):  # iterative data loading
        dat_path = os.listdir(data_path[i])  # Load training data

        if len(dat_path) is not 1: #use the first session only !!
            print("Multiple session No. {}".format(data_path[i][-3:]))

        print('Load training data: Subject ID. {}'.format(data_path[i][-10:]))
        data = loadmat(data_path[i] + '/' + dat_path[0])
        ECG_unfiltered = data['ECG_unfiltered']  # Raw ECG
        # ECG_unfiltered = data['ECG_filtered']  # Processed ECG
        # FPT_x = data['FPT_x']
        # FPT_y = data['FPT_y']

        ECG_unfiltered = ECG_unfiltered.transpose()

        for idx, t in enumerate(range(0, ECG_unfiltered.shape[1]-time_segment, time_segment)):
            if count == 0:
                FeatVect = np.expand_dims(ECG_unfiltered[:,t:t+time_segment], 0)
                y_labels = np.expand_dims(label[i], 0)
                count += 1
            else:
                FeatVect = np.concatenate((FeatVect, np.expand_dims(ECG_unfiltered[:,t:t+time_segment], 0)))
                y_labels = np.concatenate((y_labels, np.expand_dims(label[i], 0)))

    FeatVect, y_labels = shuffle_in_unison(FeatVect, y_labels)

    return FeatVect, y_labels

def Epochs_mid(data_path, label, time_segment = 1000, time_step = 250):
    FeatVect = np.array([])
    y_labels = np.array([])

    count = 0
    info = loadmat('PTBdb_30s_window_indices_new.mat')

    for i in range(len(data_path)): #iterative data loading
    # for i in range(len(data_path)-155):  # iterative data loading
        dat_path = os.listdir(data_path[i])  # Load training data

        if len(dat_path) is not 1: #use the first session only !!
            print("Multiple session No. {}".format(data_path[i][-3:]))

        print('Load training data: Subject ID. {}'.format(data_path[i][-10:]))
        data = loadmat(data_path[i] + '/' + dat_path[0])
        ECG_unfiltered = data['ECG_unfiltered']  # Raw ECG
        # ECG_unfiltered = data['ECG_filtered']  # Processed ECG
        # FPT_x = data['FPT_x']
        # FPT_y = data['FPT_y']

        ECG_unfiltered = ECG_unfiltered.transpose()

        for idx, t in enumerate(range(0, ECG_unfiltered.shape[1]-time_segment, time_segment)):
            if count == 0:
                FeatVect = np.expand_dims(ECG_unfiltered[:,t:t+time_segment], 0)
                y_labels = np.expand_dims(label[i], 0)
                count += 1
            else:
                FeatVect = np.concatenate((FeatVect, np.expand_dims(ECG_unfiltered[:,t:t+time_segment], 0)))
                y_labels = np.concatenate((y_labels, np.expand_dims(label[i], 0)))

    FeatVect, y_labels = shuffle_in_unison(FeatVect, y_labels)

    return FeatVect, y_labels


def Epochs_filt(data_path, label, time_segment = 1000, time_step = 250):
    FeatVect = np.array([])
    y_labels = np.array([])

    count = 0
    for i in range(len(data_path)): #iterative data loading
    # for i in range(len(data_path)-155):  # iterative data loading
        dat_path = os.listdir(data_path[i])  # Load training data

        if len(dat_path) is not 1: #use the first session only !!
            print("Multiple session No. {}".format(data_path[i][-3:]))

        print('Load training data: Subject ID. {}'.format(data_path[i][-10:]))
        data = loadmat(data_path[i] + '/' + dat_path[0])
        # ECG_unfiltered = data['ECG_unfiltered']  # Raw ECG
        ECG_unfiltered = data['ECG_filtered']  # Processed ECG
        # FPT_x = data['FPT_x']
        # FPT_y = data['FPT_y']

        ECG_unfiltered = ECG_unfiltered.transpose()

        for idx, t in enumerate(range(0, ECG_unfiltered.shape[1]-time_segment, time_step)):
            if count == 0:
                FeatVect = np.expand_dims(ECG_unfiltered[:,t:t+time_segment], 0)
                y_labels = np.expand_dims(label[i], 0)
                count += 1
            else:
                FeatVect = np.concatenate((FeatVect, np.expand_dims(ECG_unfiltered[:,t:t+time_segment], 0)))
                y_labels = np.concatenate((y_labels, np.expand_dims(label[i], 0)))

    FeatVect, y_labels = shuffle_in_unison(FeatVect, y_labels)

    return FeatVect, y_labels


def Epochs_CinC(data_path, label, label_idx, time_segment = 1000, time_step = 250):
    FeatVect = np.array([])
    y_labels = np.array([])

    count = 0
    for i in range(len(data_path)): #iterative data loading
    # for i in range(len(data_path)-155):  # iterative data loading
        print('Load training data: Subject ID. {}'.format(data_path[i][:5]))
        data = loadmat('../data/ECG/Training_WFDB/' + data_path[i])  # Load training data
        ECG_unfiltered = data['val']  # Processed ECG

        for idx, t in enumerate(range(0, ECG_unfiltered.shape[1]-time_segment, time_step)):
            if count == 0:
                FeatVect = np.expand_dims(ECG_unfiltered[:,t:t+time_segment], 0)
                y_labels = np.expand_dims(label[i,:], 0)
                y_labels_idx = np.expand_dims(label_idx[i], 0)
                count += 1
            else:
                FeatVect = np.concatenate((FeatVect, np.expand_dims(ECG_unfiltered[:,t:t+time_segment], 0)))
                y_labels = np.concatenate((y_labels, np.expand_dims(label[i,:], 0)))
                y_labels_idx = np.concatenate((y_labels_idx, np.expand_dims(label_idx[i], 0)))

    FeatVect, y_labels, y_labels_idx = shuffle_in_unison_ver2(FeatVect, y_labels, y_labels_idx)

    return FeatVect, y_labels, y_labels_idx

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def shuffle_in_unison_ver2(a, b, c):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    shuffled_c = np.empty(c.shape, dtype=c.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        shuffled_c[new_index] = c[old_index]
    return shuffled_a, shuffled_b, shuffled_c

if __name__ == "__main__":
    get_PTB(n_splits=10)