"""
Model 05
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
import models.wg2_dcgan as dcgan
import models.wg2_mlp as mlp
import torch
from ekg_class import dicts
from matplotlib import pyplot as plt

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("start time:      in month {0:} day {1:} at {2:02d}:{3:02d}:{4:02.0f}\n****".format(
        start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))

    sim = "05"
    drive = "E:"
    myPath_base = os.path.join(drive, "\\UTSA")
    myPath_read = os.path.join(myPath_base, "PycharmProjects_F\\DM_post_processing")
    myPath_write = os.path.join(myPath_base, "paper3_DM\\paper3_data\\gb_wgan")
    os.makedirs(myPath_write, exist_ok=True)

    resampled_to = 64

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=False, default='cifar10', help='cifar10 | lsun | beatnet | folder | lfw ')
    # parser.add_argument('--dataroot', required=False, default='.', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batchSize', type=int, default=9, help='input batch size')
    parser.add_argument('--beat_length', type=int, default=resampled_to, help='the length of the input beat')
    parser.add_argument('--nc', type=int, default=1, help='input beat channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=210, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use BatchNorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    opt = parser.parse_args()
    seed = 0

    opt.niter = 1

    opt_keys = list(opt.__dict__.keys())
    opt_values = list(opt.__dict__.values())
    opt_tuple = list(zip(opt_keys, opt_values))
    print("\n\nSettings:")
    [print("{}\n".format(opt_tuple[i * 10:(i + 1) * 10])) if (i + 1) * 10 < len(opt_keys)
     else print("{}\n".format(opt_tuple[i * 10:])) for i in range(len(opt_keys) // 10 + 1)]
    # print("\n\nopt:{}".format(opt))

    """
    if opt.experiment is None:
        opt.experiment = '05_wg2_classic_main_samples_ecg_{}_{}_{}{}'.format(start_time.month, start_time.day,
                                                                              start_time.hour, start_time.minute)

        opt.experiment = "/home/jprevost/pycharm_proj_edmond/01/05_gen_beats_run5"

    # os.system('mkdir {}'.format(opt.experiment))
    """
    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Dataset and Configure data loader

    num2descr, letter2num, letter2descr, num2letter = dicts()
    temp_path = os.path.join(myPath_read, "mitbih_64_allN.json")
    with open(temp_path, 'r') as f:
        mitbih_64_allN = json.load(f)

    X = []
    y = []
    mitbih_64_allN_array = np.asarray(mitbih_64_allN)
    X = mitbih_64_allN_array[:10000, :]
    y = [0 for _ in X]
    # *********************  End of ECG Dataset **************************

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, drop_last=True)
    del dataset

    ngpu = int(opt.ngpu)  # ngpu=1
    nz = int(opt.nz)  # nz=100
    ngf = int(opt.ngf)  # ngf=64
    ndf = int(opt.ndf)  # ndf=64
    nc = int(opt.nc)  # nc=1
    n_extra_layers = int(opt.n_extra_layers)  # n_extra_layers=0

    # write out generator config to generate beats together wth training checkpoints (.pth)
    generator_config = dict(beat_length=opt.beat_length, nz=nz, nc=nc, ngf=ngf, ngpu=ngpu,
                            n_extra_layers=n_extra_layers, noBN=opt.noBN, mlp_G=opt.mlp_G)


    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    # use batchnorm or mlp_G ?
    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.beat_length, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.mlp_G:
        netG = mlp.MLP_G(opt.beat_length, nz, nc, ngf, ngpu)
    else:
        netG = dcgan.DCGAN_G(opt.beat_length, nz, nc, ngf, ngpu, n_extra_layers)

    with open(os.path.join(myPath_base, myPath_write, "05_generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config) + "\n")

    netG.apply(weights_init)
    if opt.netG != '':  # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    if opt.mlp_D:
        netD = mlp.MLP_D(opt.beat_length, nz, nc, ndf, ngpu)
    else:
        netD = dcgan.DCGAN_D(opt.beat_length, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)
    print("\n")

    input = torch.FloatTensor(opt.batchSize, 1, opt.beat_length)
    noise = torch.FloatTensor(opt.batchSize, nz, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    # *************************************  epoch iterations  *************************************

    gen_iterations = 0

    errD_total = []
    errG_total = []

    with open(os.path.join(myPath_base, myPath_write, '_loss_D.json'), 'w', encoding='utf-8') as f:
        json.dump("["[0], f, ensure_ascii=False, indent=4)
    with open(os.path.join(myPath_base, myPath_write, '_loss_G.json'), 'w', encoding='utf-8') as f:
        json.dump("["[0], f, ensure_ascii=False, indent=4)

    for epoch in range(opt.niter):
        gen_beats_epoch = []
        errD_epoch = []
        errG_epoch = []
        # %%%%%%%%%%%%%%%%%%%%%   look down ↓  %%%%%%%%%%%%%%%%%%%%%%%%%%%
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ############################

            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters  # opt.Diters = 5

            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = data_iter.next()  # one batch (9) from dataloader_train
                i += 1

                # train with real
                real_data_cpu, _ = data

                # real_data_cpu reshaped by ea
                real_data_cpu = real_data_cpu.reshape(real_data_cpu.size(0), 1, real_data_cpu.size(-1))

                netD.zero_grad()
                batch_size = real_data_cpu.size(0)

                if opt.cuda:
                    real_data_cpu = real_data_cpu.cuda()
                input.resize_as_(real_data_cpu).copy_(real_data_cpu)  # copy real_data_cpu onto input

                errD_real = netD(input)
                errD_real.backward(one)

                # train with fake
                noise.resize_(opt.batchSize, nz, 1).normal_(0, 1)
                fake = netG(noise).data

                errD_fake = netD(fake)
                errD_fake.backward(mone)

                errD = errD_real - errD_fake
                optimizerD.step()

            ############################
            # (2) Update G network
            ############################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            noise.resize_(opt.batchSize, nz, 1).normal_(0, 1)
            fake = netG(noise)
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1

            errD_epoch.extend([errD.item()])
            errG_epoch.extend([errG.item()])
            gen_beats_epoch.extend([fake.tolist()])

            print('[{:3d}/{:3d}], [{:6d}/{:6d}], gen it:{:5d},  Loss_D:{:8.6f},  Loss_G:{:8.6f},'
                  '  Loss_D_r:{:8.6f},  Loss_D_f:{:8.6f}'.
                  format(epoch, opt.niter, i, len(dataloader), gen_iterations,
                         errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            fake = netG(fixed_noise)
            fake = fake.cpu().detach().numpy().squeeze(axis=1)
            # i += 1

        # %%%%%%%%%%%%%%%%    checkpointing    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        # torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))

        '''
        temp_path = os.path.join(myPath_base, myPath_write, '_loss_D.json')
        with open(temp_path, 'a', encoding='utf-8') as f:
            json.dump(",", f, ensure_ascii=False, indent=4)

        with open(temp_path, 'a', encoding='utf-8') as f:
            json.dump(errD_epoch, f, ensure_ascii=False, indent=4)
        '''
        '''
        temp_path = os.path.join(myPath_base, myPath_write, '_loss_G.json')
        with open(temp_path, 'a', encoding='utf-8') as f:
            json.dump(",", f, ensure_ascii=False, indent=4)
        with open(temp_path, 'a', encoding='utf-8') as f:
            json.dump(errG_epoch, f, ensure_ascii=False, indent=4)
        '''

        temp_path = os.path.join(myPath_base, myPath_write, '_gen_beats_ep{}.json'.format(epoch))
        with open(temp_path, 'a', encoding='utf-8') as f:
            json.dump(",", f, ensure_ascii=False, indent=4)

        gbe = np.array(gen_beats_epoch).squeeze(axis=2).tolist()

        with open(temp_path, 'a', encoding='utf-8') as f:
            json.dump(gbe, f, ensure_ascii=False, indent=4)

    temp_path = os.path.join(myPath_base, myPath_write, '_loss_D.json')
    with open(temp_path, 'a', encoding='utf-8') as f:
        json.dump("]", f, ensure_ascii=False, indent=4)

    temp_path = os.path.join(myPath_base, myPath_write, '_loss_G.json')
    with open(temp_path, 'a', encoding='utf-8') as f:
        json.dump("]", f, ensure_ascii=False, indent=4)

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
