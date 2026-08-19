#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import os
from PIL import Image
import numpy as np

# get file names in correct order
dir = 'Plots/'
dir = '/home/oskar//Desktop/HPC/FROST/data/results/Aletsch/glaciers/RGI2000-v7.0-G-11-02596/Monitor/'

files = os.listdir(dir)
files.sort()

# read all frames
frames = []
for image in files:
    if image.startswith('status') and image.endswith('.png'):

        img = Image.open(os.path.join(dir, image)).convert("RGBA")

        # keep transparency, do NOT add background
        frames.append(img)


# frames = frames + [frames[-1]]*5 + list(reversed(frames)) + [frames[0]]*5
# frame_one = Image.open(dir+'iterations_seed_111_2.png')
#frame_one = Image.open(dir + 'glacier_000_surface2020.png').convert("RGBA")

frame_one = Image.open(dir + 'status_001_2020.png').convert("RGBA")

frame_one.save(
    "Plots/Aletsch_calibration.gif",
    save_all=True,
    append_images=frames,
    duration=1000,
    loop=0,
    disposal=2,
    transparency=0
)