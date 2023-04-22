"""

"""
import json
import math
import os
from copy import deepcopy
from textwrap import wrap
import imageio
import numpy as np
import tqdm
from matplotlib import pyplot as plt
import math
from numpy import arccos

PI = math.pi

drive = "F:\\"
myPath_base = "UTSA\\PycharmProjects_F"
myPath_save = os.path.join(drive, myPath_base, "")
myPath_read = os.path.join(drive, myPath_base, "DM_post_processing")

f_name = "mitbih_64_allN.json"
temp_path = os.path.join(myPath_read, f_name)

with open(temp_path, 'r') as f:
    beats = json.load(f)[:10000]

beat = beats[100]

phi_rad = [arccos(item) for item in beat]
phi_rad.insert(0, 0)
phi_deg = [item / PI * 180 for item in phi_rad]
radii = [idx / len(beat) for idx in range(len(beat) + 1)]

# from: https://matplotlib.org/stable/gallery/pie_and_polar_charts/polar_demo.html
r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(phi_rad, radii)
# ax.set_rmax(1)
# ax.set_rticks([0.25, 0.5, 0.75, 1])         # Less radial ticks
# ax.set_rlabel_position(-135)               # Move radial labels away from plotted line
# ax.grid(True)

# ax.set_title("Typical Normal ECG Heartbeat Plot on Polar Axis", va='bottom')
# plt.show()

# fig = plt.gcf()
ax1 = plt.subplot(121)
ax1.set_box_aspect(1)
ax2 = plt.subplot(122, projection='polar')

ax1.plot(beat)
ax1.grid()
ax1.set_title("Cartesian Coordinates", y=1.25)

ax2.plot(phi_rad, radii, color='r', linewidth=1.5)
ax2.set_rmax(1)
# ax2.set_rticks([0.25, 0.5, 0.75, 1])         # Less radial ticks

ax2.set_yticklabels([])

ax2.set_rlabel_position(-135)  # Move radial labels away from plotted line
ax2.grid(True)
ax2.set_title("Polar Coordinates", y=1.25)
plt.tight_layout(pad=4)
plt.show()

a = 0
