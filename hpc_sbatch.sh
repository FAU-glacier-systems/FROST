#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

#SBATCH --nodes=1
#SBATCH --time=00:59:59
#SBATCH --job-name=frost
#SBATCH --gres=gpu:a40:1
#SBATCH --output=data/results/Log/frost_%j.out
#SBATCH --error=data/results/Log/frost_%j.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm3

python frost_pipeline.py --config "$1" --rgi_id "$2"


