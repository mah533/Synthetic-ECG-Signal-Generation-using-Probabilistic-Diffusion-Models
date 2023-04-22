"""
this is a classifier
input is the normalized ECG vector
output is the class of the ECG
"""


import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import csv
from torchvision import datasets
import torchvision.transforms as transforms
from collections import Counter
import gc
import copy

from tqdm import tqdm

from ekg_class import dicts
import torch.nn as nn
from models_classifier import EcgResNet34
from sklearn.metrics import classification_report as report
from sklearn.metrics import confusion_matrix as cf_matrix
from utils import print_confusion_matrix

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

num2descr, letter2num, letter2descr, num2letter = dicts()
start_time = datetime.now()

print(("\n" + "*" * 61 + "\n\t\t\t\t\tstart time  {0:02d}:{1:02d}:{2:02.0f}\n" + "*" * 61).format(
    start_time.hour, start_time.minute, start_time.second))

drive = "E:\\"
myPath_base = os.path.join(drive, "UTSA")
path_aux = 'paper3_DM\\paper3_data'
myPath_base = os.path.join(myPath_base, 'paper3_DM\\paper3_data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters etc.
dry_run = False
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_CLASSES = 2

if dry_run:
    NUM_EPOCHS = 1
else:
    NUM_EPOCHS = 10

classes2keep = ['N', 'L']
classes2keep_folder = ['N', 'L']

key_aug = "aug"
# key_aug = "notaug"

key_bal = 'balanced'
# key_bal = 'imbalanced'

# key_case = 'rl'
# key_case = "02"
key_case = "wgan"

# len_ratio = .01                   # shorter train sets
len_ratio = 1                  # shorter train sets
num_samples = 8000
tst_len = 1000
print('\ncase: {}, {}, {}\n'.format(key_case, key_bal, key_aug))
# num_N_samples = int(len_ratio*num_samples)

if '.' in str(len_ratio):
    len_ratio_str = str(len_ratio).replace('.', '')
else:
    len_ratio_str = str(len_ratio)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path = "gb_dm_case_{}".format(key_case)
myPath_save = os.path.join(myPath_base, path)
os.makedirs(myPath_save, exist_ok=True)

brk = 'here'

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  begin: generate and save MIT-BIH dataset   %%%%%%%%%%%%%%%%%%
"""
with open(os.path.join(myPath_base, 'mitbih_64_allN.json'.format(key_case)), "r") as f:
    mitbih_64_N = json.load(f)[:num_samples]

with open(os.path.join(myPath_base, 'mitbih_64_allL.json'.format(key_case)), "r") as f:
    mitbih_64_L = json.load(f)[:num_samples]

rand_idx = random.sample(range(0, num_samples), num_samples)
idx_trn = rand_idx[:7000]
idx_tst = rand_idx[7000:]
X_N_trn = [mitbih_64_N[idx] for idx in idx_trn]
X_N_tst = [mitbih_64_N[idx] for idx in idx_tst]

rand_idx = random.sample(range(0, num_samples), num_samples)
idx_trn = rand_idx[:7000]
idx_tst = rand_idx[7000:]
X_L_trn = [mitbih_64_L[idx] for idx in idx_trn]
X_L_tst = [mitbih_64_L[idx] for idx in idx_tst]


path = os.path.join(myPath_base, "X_N_trn.json")
with open(path, 'w') as f:
    json.dump(X_N_trn, f)

path = os.path.join(myPath_base, "X_N_tst.json")
with open(path, 'w') as f:
    json.dump(X_N_tst, f)

path = os.path.join(myPath_base, "X_L_trn.json")
with open(path, 'w') as f:
    json.dump(X_L_trn, f)

path = os.path.join(myPath_base, "X_L_tst.json")
with open(path, 'w') as f:
    json.dump(X_L_tst, f)
"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  end: generate and save MIT-BIH dataset   %%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  begin: load X and y   %%%%%%%%%%%%%%%%
path = os.path.join(myPath_base, "X_N_trn.json")
with open(path, 'r') as f:
    X_N_trn = json.load(f)

path = os.path.join(myPath_base, "X_N_tst.json")
with open(path, 'r') as f:
    X_N_tst = json.load(f)

path = os.path.join(myPath_base, "X_L_trn.json")
with open(path, 'r') as f:
    X_L_trn = json.load(f)

path = os.path.join(myPath_base, "X_L_tst.json")
with open(path, 'r') as f:
    X_L_tst = json.load(f)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  end: load X and y   %%%%%%%%%%%%%%%%%%
X_tst = X_N_tst + X_L_tst
y_N_tst = [0 for _ in range(len(X_N_tst))]
y_L_tst = [1 for _ in range(len(X_L_tst))]
y_tst = y_N_tst + y_L_tst

if key_aug == 'aug':
    with open(os.path.join(myPath_base, 'gb_dm_case_{}\\gb_dm_1d_case_{}.json'.format(key_case, key_case)), "r") as f:
        gb_for_aug = json.load(f)

    rand_idx = random.sample(range(0, len(gb_for_aug)), len(gb_for_aug))
    temp = [gb_for_aug[idx] for idx in rand_idx]
    X_aug = temp[:7000-350]
else:
    X_aug = []

# # imbalanced case N: 350, select 350 elements in X_N_trn if 'imbalanced' or 'augmented'
if key_bal == 'imbalanced' or key_aug == 'aug':
    rand_idx = random.sample(range(0, len(X_N_trn)), len(X_N_trn))[:350]
    X_N_trn = [X_N_trn[idx] for idx in rand_idx]

X_N_trn = X_N_trn + X_aug
X_trn = X_N_trn + X_L_trn

y_N_trn = [0 for _ in range(len(X_N_trn))]
y_L_trn = [1 for _ in range(len(X_L_trn))]
y_trn = y_N_trn + y_L_trn

trn_set = TensorDataset(torch.tensor(X_trn), torch.tensor(y_trn))
tst_set = TensorDataset(torch.tensor(X_tst), torch.tensor(y_tst))
trn_loader = DataLoader(trn_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
tst_loader = DataLoader(tst_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

brk = 'here'
# %%%%%%%%%%%%%%%%%     begin save sample plots of classes in classes2keep    %%%%%%%%%%%%%%%%%%%%
'''
for cl in classes2keep:
    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.suptitle("Class {} ({}: {}), count: {}".
                 format(classes2keep.index(cl), cl, letter2descr[cl], len(data2keep_dict[cl])))

    count = 0
    for i in range(3):
        for j in range(3):
            count += 1
            if count >= len(data2keep_dict[cl]):
                continue
            axes[i][j].plot(data2keep_dict[cl][count])
            axes[i][j].grid()
    plt.savefig(os.path.join(myPath_save, "00_sample_cl_{}.png".format(classes2keep.index(cl))))

plt.close("all")
'''
# %%%%%%%%%%%%%%%%%     end save sample plots of classes in classes2keep      %%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%% begin:     Select and Initialize Network       %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# net = net_cnn(num_classes=NUM_CLASSES).to(device)
# net = net_fc(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
net = EcgResNet34(num_classes=NUM_CLASSES).to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
# %%%%%%%%%%%%%%%%%%%% end:     Select and Initialize Network       %%%%%%%%%%%%%%%%%%%%%%%%%%%%

'''
# NUM_EPOCHS = 1
# %%%%%%%%%%%%%%%%%%%%    begin:  Train Classifier       %%%%%%%%%%%%%%%%%%%%%%%%%%
for epoch in range(NUM_EPOCHS):
    for batch_idx, (inputs, labels) in enumerate(trn_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.reshape(inputs.shape[0], 1, -1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        scores = net(inputs)
        loss = criterion(scores.squeeze(), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        if batch_idx % 200 == 0:  # print every 200 mini-batches
            now = datetime.now()
            print('{:02d}:{:02d}:{:02d}\t\tepoch={:4d} / {:4d}\t\titer={:5d} / {:5d}\t\t\tloss: {:7.5f}'.
                  format(now.hour, now.minute, now.second, epoch, NUM_EPOCHS, batch_idx, len(trn_loader), loss))

print('\n\tFinished Training\n')
# %%%%%%%%%%%%%%%%%%%%    end  Train Network       %%%%%%%%%%%%%%%%%%%%%%%%%%%%


print('\tSaving model ...\n')

# %%%%%%%%%%%%%%%%%%%%    begin save model   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f_name = "classifier_{}_{}_{}.pth".format(key_case, key_bal, key_aug)
PATH = os.path.join(myPath_save, f_name)
torch.save(net.state_dict(), PATH)
# %%%%%%%%%%%%%%%%%%%%    end save model     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''


# %%%%%%%%%%%%%%%%%%%%    load trained classifier     %%%%%%%%%%%%%%%%%%%%%
path = "gb_dm_case_{}".format(key_case)
myPath = os.path.join(myPath_base, path)
classifier_name = "classifier_{}_{}_{}.pth".format(key_case, key_bal, key_aug)
net.load_state_dict(torch.load(os.path.join(myPath, classifier_name)))

brk = 'here'

# %%%%%%%%%%%%%%%%%%    print Classification Report to file      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trntst = ['trn', 'tst']
for key_trntst in trntst:
    print('Preparing Classification Report: {}'.format(key_trntst))
    path = os.path.join(myPath_save, 'classification_report_{}_{}_{}_{}.txt'.
                        format(key_trntst, key_case, key_bal, key_aug))
    with open(path, 'w') as sys.stdout:
        y_scores = []
        y_true = []
        y_pred = []
        exec('loader = {}_loader'.format(key_trntst))
        for (inputs, labels) in tqdm(loader):
            inputs = inputs.reshape(inputs.shape[0], 1, -1).to(device)
            labels = labels.to(device)
            temp = net(inputs)
            output = temp.max(dim=1)

            y_true.extend(labels.data.tolist())
            y_pred.extend(output.indices.tolist())
            y_scores.extend((F.softmax(temp, dim=1).tolist()))

        print("\n")
        print("%" * 20 + "\tClassification Report ({} Set, {}, {}, {})\t".
              format(key_trntst, key_case, key_bal, key_aug) + "%" * 20)
        print("Classes:                             {}".format(classes2keep_folder))
        print("Classifier Model:                    {}".format(net._get_name()))
        print(f"number of epochs:                    {NUM_EPOCHS}\n")
        print("Train set length:                    {}".format(len(trn_set)))
        print("Test set length:                     {}".format(len(tst_set)))
        print("Train set reduction length ratio:    {}\t".format(len_ratio_str))
        print('\nClassification Report:')
        print(report(y_true, y_pred, target_names=classes2keep_folder))
        print("\nConfusion Matrix:\n {}".format(cf_matrix(y_true, y_pred)))
        precision, recall, thresholds = precision_recall_curve(y_true, np.asarray(y_scores)[:, 1])
        pr_recall_auc_score = auc(recall, precision)
        print('\nPrecision-Recall AUC Score:        {:6.4f}'.format(pr_recall_auc_score))
        rocauc_score = roc_auc_score(y_true, np.asarray(y_scores)[:, 1])
        print('\nROC AUC Score:                     {:6.4f}'.format(rocauc_score))
        print_confusion_matrix(cf_matrix(y_true, y_pred), class_names=classes2keep_folder,
                               fig_name="Conf. Matrix_{}_{}_{}_{}".format(key_trntst, key_case, key_bal, key_aug))
        plt.savefig(os.path.join(myPath_save, "cfmx_{}_{}_{}_{}.png".format(key_trntst, key_case, key_bal, key_aug)))
        plt.close("all")

        plt.plot(precision, recall)
        plt.title("Precision - Recall Curve ({}, {}, {}, {})".format(key_trntst, key_case, key_bal, key_aug))
        plt.grid()
        plt.savefig(os.path.join(myPath_save, "prec_recall_curve_{}_{}_{}_{}.png".
                                 format(key_trntst, key_case, key_bal, key_aug)))
        plt.close("all")
        sys.stdout = sys.__stdout__

finish_time = datetime.now()
print(("\n\n\n" + "finish time = {0:02d}:{1:02d}:{2:02.0f}").format(
    finish_time.hour, finish_time.minute, finish_time.second))

laps = finish_time - start_time
tot_sec = laps.total_seconds()
h = int(tot_sec // 3600)
m = int((tot_sec % 3600) // 60)
s = int(tot_sec - (h * 3600 + m * 60))

print("total elapsed time = {:02d}:{:2d}:{:2d}".format(h, m, s))

"""
#load model
model = model_name(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""


