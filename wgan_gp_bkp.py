"""
from GitHub: https://github.com/mah533/Augmentation-of-ECG-Training-Dataset-with-CGAN/blob/main/main_wgan_gp_ecg.py
"""
import copy
import datetime
import json
import os
from collections import Counter
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model_wgan_gp import Disc_dcgan_gp_1d, Gen_dcgan_gp_1d, initialize_weights
from utils_wgan_gp import gradient_penalty, grid_plot_save, normalize

from ekg_class import dicts
import csv
import torch.nn as nn
import torchvision.datasets as datasets

num2descr, letter2num, letter2descr, num2letter = dicts()

start_time = datetime.datetime.now()
print(("\n" + "*" * 50 + "\n\t\tstart time:      {0:02d}:{1:02d}:{2:02.0f}\n" + "*" * 50).format(
    start_time.hour, start_time.minute, start_time.second))

drive = "E:\\"
myPath_base = os.path.join(drive, "UTSA")
# myPath_dataset = os.path.join(myPath_base, "Datasets\\mitbih_datasets_Dictionaries\\")
myPath_dataset = os.path.join(drive, "UTSA\\PycharmProjects_F\\DM_post_processing")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters etc.
dry_run = False
if dry_run:
    NUM_EPOCHS = 1
else:
    NUM_EPOCHS = 60
BATCH_SIZE = 16
CRITIC_ITERATIONS = 5
CHANNELS_IMG = 1
IMAGE_SIZE = 64
FEATURES_DISC = 64
FEATURES_GEN = 64
LEARNING_RATE = 1e-4
LAMBDA_GP = 10
Z_DIM = 100
ratio = 0.8  # train/test split ratio
tr_ts_ratio = 1  # "training set / test set" split ratio
len_ratio = 0.5  # to study the effect of support set number of samples (shorter train sets)

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)],
        )
    ]
)

all_classes = ['/', 'A', 'E', 'F', 'J', 'L', 'N', 'Q', 'R', 'S', 'V', 'a', 'e', 'f', 'j']
all_classes.remove('Q')
all_classes.remove('S')
# all_classes.remove('V')      # why remove this? plots show no-uniform pattern

'''
stats_all_classes = 
        {
        'N': 75052,   'L': 8075,  'R': 7259,  'V': 7130,
        '/': 7028,    'A': 2546,  'f': 982,   'F': 803,
        'j': 229,     'a': 150,   'E': 106,   'J': 83,
        'Q': 33,      'e': 16,    'S': 2
        }
'''
#  %%%%%%%%%%%%%%  '/' and 'P' are the same       %%%%%%%%%%%%%%%%%%

# classes2keep = ['/', 'A', 'L', 'N', 'R', 'f', 'j']    # selected for the paper, Jan 13
# classes2keep_folder = ['P', 'A', 'L', 'N', 'R', 'f', 'j']    # selected for the paper, Jan 13
classes2keep_folder = ['N']
classes2keep = ['N']

myPath_save = os.path.join(myPath_base, "paper3_DM\\paper3_data\\gb_wgan")
os.makedirs(myPath_save, exist_ok=True)

# %%%%%%%%%%%%%%%%       begin MIT-BIH Dataset      %%%%%%%%%%%%%%%%%%%%%%%%%%%%

with open(os.path.join(myPath_dataset, "record_X_y_adapt_win_bef075_aft075_Normalized.json"), "r") as f:
    data = json.load(f)

stats_all_classes = Counter(np.asarray(data, dtype=object)[:, 2])

# create dictionary of data to be kept
vals = []
for idx in range(len(classes2keep)):
    vals.append([])

data2keep_dict = dict(zip(classes2keep, vals))
data_train_dict = copy.deepcopy(data2keep_dict)
data_test_dict = copy.deepcopy(data2keep_dict)

for item in data:
    if item[2] in classes2keep:
        data2keep_dict[item[2]].append(item[1])
del data

# randomly splitting the dataset into train and test sets
for key in data2keep_dict.keys():
    val_len = len(data2keep_dict[key])
    idx_train = torch.randperm(val_len)[:int(tr_ts_ratio * val_len)]
    idx_test = torch.randperm(val_len)[int(tr_ts_ratio * val_len) + 1:]
    data_train_dict[key] = [data2keep_dict[key][idx] for idx in idx_train]
    data_test_dict[key] = [data2keep_dict[key][idx] for idx in idx_test]

if '/' in classes2keep:
    temp = data2keep_dict['/']
    data2keep_dict.pop('/')
    data2keep_dict['P'] = temp

    temp = data_train_dict['/']
    data_train_dict.pop('/')
    data_train_dict['P'] = temp

    temp = data_test_dict['/']
    data_test_dict.pop('/')
    data_test_dict['P'] = temp

# plot samples of beats in classes2keep
"""
for cl in classes2keep:
    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.suptitle("Class {} ({}: {}), count:{}".format(classes2keep.index(cl), cl, letter2descr[cl],
                                                      len(data2keep_dict[cl])))

    count = 0
    for i in range(3):
        for j in range(3):
            count += 1
            if count >= len(data2keep_dict[cl]):
                continue
            axes[i][j].plot(data2keep_dict[cl][count])
            axes[i][j].grid()
    plt.savefig(os.path.join(myPath_save, "00_sample_cl_{}".format(classes2keep)))

plt.close("all")
"""

'''
X = []
y = []
for item in data:
    if item[2] in classes2keep:
        X.append(item[1])
        idx = classes2keep.index(item[2])
        y.append(idx)
stats_y = Counter(y)
'''

# create X, y for train and test sets
X_train = []
y_train = []
for key in data_train_dict.keys():
    X_train.extend(data_train_dict[key])
    idx = [classes2keep_folder.index(key)] * len(data_train_dict[key])
    y_train.extend(idx)

X_train = X_train[:10000]
y_train = y_train[:10000]

X_test = []
y_test = []
for key in data_test_dict.keys():
    X_test.extend(data_test_dict[key])
    idx = [classes2keep_folder.index(key)] * len(data_test_dict[key])
    y_test.extend(idx)
# %%%%%%%%%%%%%%%%       end MIT-BIH Dataset        %%%%%%%%%%%%%%%%%%%%%%%%%%

a = 0
'''
with open(os.path.join(myPath_save, "X_train.json"), "w") as f:
    json.dump(X_train, f)
with open(os.path.join(myPath_save, "y_train.json"), "w") as f:
    json.dump(y_train, f)
with open(os.path.join(myPath_save, "X_test.json"), "w") as f:
    json.dump(X_test, f)
with open(os.path.join(myPath_save, "y_test.json"), "w") as f:
    json.dump(y_test, f)
'''

dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

gen = Gen_dcgan_gp_1d(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Disc_dcgan_gp_1d(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1).to(device)

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.unsqueeze(1).to(device)

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # Train Generator: min -E[critic(gen_fake)]
            output = critic(fake).reshape(-1)
            loss_gen = -torch.mean(output)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            t_now = datetime.datetime.now()
            print(
                f"{t_now.hour:02d}:{t_now.minute:02d}:{t_now.second:02d}     Epoch [{epoch: 3d} / {NUM_EPOCHS: 3d}]    Batch {batch_idx: 4d}/{len(dataloader): 5d} \
                     Loss D: {loss_critic: .4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise)
                path = os.path.join(myPath_save, 'images')
                plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.99, wspace=0.99)
                grid_plot_save(n_row=4, n_col=4, signal=fake.squeeze().to("cpu"), path=path,
                               f_name='wgangp_gb_ep{}_{}.png'.format(epoch, batch_idx))

                # benchmarking Generator at end of each epoch
                '''
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': gen.state_dict(),
                    'optimizer_state_dict': opt_gen.state_dict(),
                    'loss': loss_gen
                }, myPath_save + "generator_bench.tar")

                # benchmarking Discriminator end of each epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': critic.state_dict(),
                    'optimizer_state_dict': opt_critic.state_dict(),
                    'loss': loss_critic
                }, myPath_save + "discriminator_bench.tar")
                '''

# save model
torch.save(gen.state_dict(), os.path.join(myPath_save, "generator_trained_cl_{}.pt".format(classes2keep[0])))
torch.save(critic.state_dict(), os.path.join(myPath_save, "discriminator_trained_cl_{}.pt".format(classes2keep[0])))

now = datetime.datetime.now()
print("\ntotal elapsed time: {}".format(now - start_time))

"""
Codes for:
YT, "WGAN implementation from scratch (with gradient penalty)" by Aladdin Persson
Part II - WGAN GP
https://www.youtube.com/watch?v=pG0QZ7OddX4

in WGAN:
   1) better stability
   2) loss is a metric: it is a Termination Criteria
"""
