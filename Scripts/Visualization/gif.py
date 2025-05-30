#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import os
from PIL import Image
import numpy as np

# get file names in correct order
dir = '../../Experiments/RGI2000-v7.0-G-11-01706/hpc_hugonnet/'
files = os.listdir(dir)
files.sort()

# read all frames
frames = []
for image in files:
    if image.startswith('map') and image.endswith('.png'):
        frames.append(Image.open(dir + image))

# frames = frames + [frames[-1]]*5 + list(reversed(frames)) + [frames[0]]*5
# frame_one = Image.open(dir+'iterations_seed_111_2.png')
frame_one = Image.open(dir + 'maps_001_2020.png')

frame_one.save("../../Plots/IterationRhone_map.gif", format="GIF",
               append_images=frames,
               save_all=True,
               duration=1000, loop=0)
