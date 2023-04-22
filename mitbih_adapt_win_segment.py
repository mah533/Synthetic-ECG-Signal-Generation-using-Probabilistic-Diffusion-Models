import datetime
import json
import os
from collections import Counter
import wfdb
import winsound
from scipy.signal import resample
from tqdm import tqdm

from blocks import normalize
from ekg_class import dicts
import numpy as np
from matplotlib import pyplot as plt

# *********************  Begin Read MITBIH data and segment it with fixed window **************************
drive = "E:\\"
myPath_base = os.path.join(drive, "UTSA")

records = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
           '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
           '209', '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231',
           '232', '233', '234']

# annot_ttl = ['!' '"' '+' '/' 'A' 'E' 'F' 'J' 'L' 'N' 'Q' 'R' 'S' 'V' '[' ']' 'a' 'e', 'f' 'j' 'x' '|' '~']

annot_nonbeat = ['!', '"', '+', '[', ']', 'x', '|', '~']
# records = ["208"]

key_fixadapt = "adaptwin"
resample_to = 64
# there are two channels in each record: 0 and 1
ch = 0
adapt_win_not_normalized_not_adjusted = {}

# for "fix_win segmentation"
# w_bef = 160
# w_aft = 180

# for "adaptive_win segmentation"
coeff_bef = 0.75
coeff_aft = 0.75

print('')
print("\nReading raw MIT-BIH data and creating the dictionary ...")
for rec in tqdm(records):
    annots_samples = []
    annots_symbols = []
    exec("rec_{} = []".format(rec))
    path = os.path.join(myPath_base, "mitbih-arrhyth_raw", rec)
    record_ttl = wfdb.rdrecord(path)
    annots_ttl = wfdb.rdann(path, "atr")

    for i in range(len(annots_ttl.symbol)):
        if annots_ttl.symbol[i] not in annot_nonbeat:
            annots_samples.extend([annots_ttl.sample[i]])
            annots_symbols.extend([annots_ttl.symbol[i]])

    for i, idx in enumerate(annots_samples, start=0):
        # for "fix_win segmentation"
        if i == 0:
            w_bef = idx
        else:
            w_bef = int((idx - annots_samples[i - 1]) * coeff_bef)
            # w_bef = 160

        if i == len(annots_samples) - 1:
            w_aft = len(record_ttl.p_signal[:, 0]) - idx
        else:
            w_aft = int((annots_samples[i + 1] - idx) * coeff_aft)
            # w_aft = 180

        bg = idx - w_bef
        en = idx + w_aft
        beat = record_ttl.p_signal[:, ch][bg:en]
        label = annots_symbols[i]
        # print(rec, idx, i)
        beat_resampled = resample(beat, resample_to).tolist()
        exec("rec_{}.append([beat_resampled, label])".format(rec))
    exec("adapt_win_not_normalized_not_adjusted[rec] = rec_{}".format(rec))
    exec("del rec_{}".format(rec))

path = os.path.join(myPath_base, "mitbih_{}_{}_bef{}_aft{}_Not_Normalized.json".
                    format(resample_to, key_fixadapt, str(coeff_bef)[2:], str(coeff_aft)[2:]))
# with open(path, "w") as f:
#    json.dump(adapt_win_not_normalized_not_adjusted, f)

"""
# %%%%%%%%%%        in case the file already exists: 
# %%%%%%%%%%%       reads from file, resamples and normalizes, and writes to another file
path = os.path.join(myPath_base, "mitbih_{}_{}_bef{}_aft{}_Not_Norm.json".
                    format(resample_to, key_fixadapt, str(coeff_bef)[2:], str(coeff_aft)[2:]))
with open(path, "r") as f:
    data = json.load(f)
"""

data = adapt_win_not_normalized_not_adjusted

data_normalized = {}
print("\n\nNormalizing beats between [-1, 1] ...")
for rec in tqdm(records):
    X_rec = np.asarray(data[rec], dtype=object)[:, 0]
    y_rec = np.asarray(data[rec], dtype=object)[:, 1]

    temp_beatlabel = [[normalize(X_rec[i]), y_rec[i]] for i in range(len(X_rec))]
    data_normalized[rec] = temp_beatlabel

path = os.path.join(myPath_base, "PycharmProjects_F\\DM_post_processing\\mitbih_{}_{}_bef{}_aft{}_Normalized.json".
                    format(resample_to, key_fixadapt, str(coeff_bef)[2:], str(coeff_aft)[2:]))
with open(path, "w") as f:
    json.dump(data_normalized, f)

cl = 'N'
print("\n\nPutting all '{}'-beats in one file ...".format(cl))
exec("beats_all{} = [[]]".format(cl))
# %%%%%%%%%%%%      Begin: All N Beats in One file       %%%%%%%%%%%%%%%%%
for rec in tqdm(records):
    temp_rec = data_normalized[rec]
    exec("temp_rec_all{} = []".format(cl))
    for beat_label in temp_rec:
        if beat_label[1] == cl:
            exec("temp_rec_all{}.append(beat_label[0])".format(cl))
    exec("beats_all{}.extend(temp_rec_all{})".format(cl, cl))
exec("beats_all{}.remove([])".format(cl))
# %%%%%%%%%%%%      End: All N Beats in one file       %%%%%%%%%%%%%%%%%
path = os.path.join(myPath_base, "PycharmProjects_F\\DM_post_processing\\mitbih_{}_allN.json".format(resample_to))
with open(path, "w") as f:
    exec("json.dump(beats_all{}, f)".format(cl))

template_num = 3
print('\n\nSaving template (sample {})...'.format(template_num))
path = os.path.join(myPath_base, "PycharmProjects_F\\DM_post_processing\\ecg_{}_template_sample{}.json".
                    format(resample_to, template_num))
with open(path, 'w') as f:
    exec("json.dump(beats_all{}[template_num], f)".format(cl))
a = 0
