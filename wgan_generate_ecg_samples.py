"""
loads a trained generator and generates samples
"""

from __future__ import print_function
import argparse
import datetime
import random
import torch.optim as optim
import winsound
import os
import json
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

import models.wg2_dcgan as dcgan
import models.wg2_mlp as mlp
import torch
from ekg_class import dicts
from matplotlib import pyplot as plt

from model_wgan_gp import Gen_dcgan_gp_1d

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("start time:      in month {0:} day {1:} at {2:02d}:{3:02d}:{4:02.0f}\n****".format(
        start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))

    drive = "F:"
    myPath_base = os.path.join(drive, "\\UTSA")
    myPath_read = os.path.join(myPath_base, "PycharmProjects_F\\DM_post_processing")
    myPath_write = os.path.join(myPath_base, "paper3_DM\\paper3_data\\dm_and_case_wgan")
    os.makedirs(myPath_write, exist_ok=True)

    resampled_to = 64
    Z_DIM = 100
    CHANNELS_IMG = 1
    FEATURES_GEN = 64
    BATCH_SIZE = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wrong one: netG = dcgan.DCGAN_G(opt.beat_length, nz, nc, ngf, ngpu, n_extra_layers)
    netG = Gen_dcgan_gp_1d(Z_DIM, CHANNELS_IMG, FEATURES_GEN)

    temp_path = os.path.join(myPath_write, 'generator_trained_cl_N.pt')
    netG.load_state_dict(torch.load(temp_path))
    netG.eval()
    print(netG)

    # Generate fake data
    genbeats = []
    num_samples = 2000
    for idx in tqdm(range(num_samples)):
        # noise = torch.randn((BATCH_SIZE, Z_DIM, 1)).to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1))
        fake = netG(noise)
        genbeats.extend(fake.tolist())

    genbeats = np.asarray(genbeats).squeeze().tolist()
    temp_path = os.path.join(myPath_base, myPath_write, 'wgan_genbeats_{}.json'.format(num_samples))
    # temp_path = os.path.join("C:\\Users\\clearlab-admin\\OneDrive\\Documents\\Flores", 'wgan_genbeats_{}.json'.format(num_samples))
    with open(temp_path, 'a') as f:
        json.dump(genbeats, f)

    '''
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    '''

    # calculation of "elapsed time"
    elapsed_time = datetime.datetime.now() - start_time
    print("total elapsed time: {}\n".format(elapsed_time))
    hours, remainder = divmod(elapsed_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("elapsed time: \t  {0:02d}:{1:02.0f}:{2:02.0f}\n (hh:mm:ss)".format(hours, minutes, seconds))

    # *******************************************************************************
    #                               completion alarm
    # *******************************************************************************

    frequency = 440  # Set Frequency To 2500 Hz
    duration = 1200  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

    frequency = 262  # Set Frequency To 2500 Hz
    duration = 1200  # Set Duration To 1000 ms == 1 second
    # winsound.Beep(frequency, duration)
