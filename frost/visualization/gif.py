#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# GIF of the per-iteration status plots of a calibration (monitor_plots: all).
#   python frost/visualization/gif.py data/results/<experiment>/<rgi_id>/Monitor

import argparse
import os
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='GIF of the Monitor status plots')
parser.add_argument('monitor_dir', help='Monitor folder of a glacier (relative path)')
dir = parser.parse_args().monitor_dir

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

frame_one = Image.open(os.path.join(dir, 'status_001_2020.png')).convert("RGBA")

os.makedirs("Plots/aletsch", exist_ok=True)
frame_one.save(
    "Plots/aletsch/Aletsch_calibration.gif",
    save_all=True,
    append_images=frames,
    duration=1000,
    loop=0,
    disposal=2,
    transparency=0
)