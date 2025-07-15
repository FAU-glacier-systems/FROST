#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import os
from PIL import Image
import numpy as np

# get file names in correct order
dir = '../../Plots/3D/'
files = os.listdir(dir)
files.sort()

# read all frames
frames = []
for image in files:
    if image.startswith('glacier') and image.endswith('.png'):
        frames.append(Image.open(dir + image))

# frames = frames + [frames[-1]]*5 + list(reversed(frames)) + [frames[0]]*5
# frame_one = Image.open(dir+'iterations_seed_111_2.png')
frame_one = Image.open(dir + 'glacier_surface2001.png')

frame_one.save("../../Plots/IterationKanderfirn_map.gif", format="GIF",
               append_images=frames,
               save_all=True,
               duration=500, loop=0)
