#!/bin/bash -l
# Zip the ContinuIX upload folder GROUP_FAU2 (3.8 GB of netCDF) into
# data/results/continuix/GROUP_FAU2.zip and test the archive. Too much I/O
# for the login node. Submit from an Alex login node, in the repo root:
#   sbatch experiments/continuix/zip_submission.sh

#SBATCH --job-name=continuix-zip
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/continuix_zip_%j.out

set -e
module load python
conda activate igm32-frost

cd data/results/continuix
python - <<'EOF'
import os
import zipfile

src, out = 'GROUP_FAU2', 'GROUP_FAU2.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, _, files in os.walk(src):
        for f in sorted(files):
            z.write(os.path.join(root, f))
with zipfile.ZipFile(out) as z:
    bad = z.testzip()
    names = z.namelist()
assert bad is None, f'corrupt member: {bad}'
print(f"{sum(n.endswith('.nc') for n in names)} nc, {len(names)} files, "
      f"{os.path.getsize(out) / 1e9:.2f} GB")
EOF
