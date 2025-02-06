import os
from PIL import Image
import numpy as np

# get file names in correct order
dir = '../../Experiments/RGI2000-v7.0-G-11-01706/Experiment_20_30_1.0_1/'
files = os.listdir(dir)
files.sort()

# read all frames
frames = []
for image in files:
    if image.startswith('status'):
        frames.append(Image.open(dir + image))

# frames = frames + [frames[-1]]*5 + list(reversed(frames)) + [frames[0]]*5
# frame_one = Image.open(dir+'iterations_seed_111_2.png')
frame_one = Image.open(dir + 'status_1_2020.png')

frame_one.save("../../Plots/IterationRhone.gif", format="GIF",
               append_images=frames,
               save_all=True,
               duration=1000, loop=0)
