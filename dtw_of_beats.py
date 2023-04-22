"""
calculates average DTW distance of beats (generated or real)
from a visually selected template
"""
import json
import os
import random
import sys
from datetime import datetime
from similaritymeasures import frechet_dist as frechet
from dtaidistance.dtw import distance as dtw
from tqdm import tqdm
from matplotlib import pyplot as plt
from statistics import mean
import numpy as np

start_time = datetime.now()
print(("\n" + "*" * 61 + "\n\t\t\t\t\tstart time  {0:02d}:{1:02d}:{2:02.0f}\n" + "*" * 61).format(
    start_time.hour, start_time.minute, start_time.second))

cl = 'N'
num_beats = 10000

# cases = ['00', '01', '02', 'rl']
cases = ['00', '01', '02', 'rl', 'wgan']
case = cases[0]

drive = "E:"
myPath_base = os.path.join(drive, "\\UTSA")
for case in cases:
    if case == 'rl':
        exec("fname_gb_{} = 'mitbih_64_allN.json'".format(case, case))
    else:
        exec("fname_gb_{} = 'gb_dm_1d_case_{}.json'".format(case, case))

    path_aux = "paper3_DM\\paper3_data\\gb_dm_case_{}_rev1".format(case)
    exec("temp_path = os.path.join(myPath_base, path_aux, fname_gb_{})".format(case))
    with open(temp_path, 'r') as f:
        exec("gb_dm_{} = json.load(f)".format(case))

gb_dm_rl = random.choices(gb_dm_rl, k=10000)
gb_dm_wgan = random.choices(gb_dm_wgan, k=10000)

fname_template = "ecg_64_template_sample3.json"
path_aux = "PycharmProjects_F\\DM_post_processing"
temp_path = os.path.join(myPath_base, path_aux, fname_template)
with open(temp_path, 'r') as f:
    template = json.load(f)

for case in cases:
    print("\n\tcase: {}".format(case))
    dist_dtw = []
    dist_frechet = []
    exec("beats = gb_dm_{}".format(case))

    for beat in tqdm(beats[:num_beats]):
        dist_dtw.append(dtw(beat, template))
        dist_frechet.append(frechet(beat, template))

    exec("dist_dtw_mean_{} = mean(dist_dtw)".format(case))
    exec("dist_frechet_mean_{} = mean(dist_frechet)".format(case))

original_stdout = sys.stdout
# %%%%%%%%%%%%%%%%%%%%%%% begin: write all results to file %%%%%%%%%%%%%%%%%%%%%%%%%%%%
path = os.path.join(myPath_base, "paper3_DM\\paper3_data\\qualities")
with open(os.path.join(path, 'results_distances.txt'), 'w') as sys.stdout:
    print('\nNumber of beats in the class {}: {}'.format(cl, num_beats))
    print('template sample number: {}'.format(fname_template))

    print('\nAverage DTW distance:')
    print('\t case 00:      {:5.3f}'.format(dist_dtw_mean_00))
    print('\t case 01:      {:5.3f}'.format(dist_dtw_mean_01))
    print('\t case 02:      {:5.3f}'.format(dist_dtw_mean_02))
    print('\t case wgan:    {:5.3f}'.format(dist_dtw_mean_wgan))
    print('\t case rl:      {:5.3f}'.format(dist_dtw_mean_rl))

    print()
    print('%' * 60)

    print('\nAverage Frechet distance:')
    print('\t case 00:      {:5.3f}'.format(dist_frechet_mean_00))
    print('\t case 01:      {:5.3f}'.format(dist_frechet_mean_01))
    print('\t case 02:      {:5.3f}'.format(dist_frechet_mean_02))
    print('\t case wgan:    {:5.3f}'.format(dist_frechet_mean_wgan))
    print('\t case rl:      {:5.3f}'.format(dist_frechet_mean_rl))
# %%%%%%%%%%%%%%%%%%%%%%% end: write all results to file %%%%%%%%%%%%%%%%%%%%%%%%%%%%
sys.stdout = original_stdout

finish_time = datetime.now()
print(("\n\n\n" + "finish time = {0:02d}:{1:02d}:{2:02.0f}").format(
    finish_time.hour, finish_time.minute, finish_time.second))

laps = finish_time - start_time
tot_sec = laps.total_seconds()
h = int(tot_sec // 3600)
m = int((tot_sec % 3600) // 60)
s = int(tot_sec - (h * 3600 + m * 60))

print("total elapsed time = {:02d}:{:02d}:{:02d}".format(h, m, s))

brk = 'here'
