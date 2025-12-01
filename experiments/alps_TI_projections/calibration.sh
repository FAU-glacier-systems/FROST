#!/bin/bash -l
#
#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

#SBATCH --nodes=1
#SBATCH --time=22:55:00
#SBATCH --job-name=frost
#SBATCH --output=Log/frost_batch_%j.out
#SBATCH --error=Log/frost_batch_%j.err


RGI_region=11

# Exit on error
set -e

module add python
#conda activate frost_env_igm3
conda activate igm3

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80


# setting
exec_script='frost_pipeline.py'
list_fname='todo_codeskel.txt'
PATH_exp_config='alps_TI_projections/eALPS'
FNAME_exp_config='config_TI'


while read rgi_id; 
do

  echo $rgi_id
  cd ..

  # Copy config file
  FNAME_in=${PATH_exp_config}"/"${FNAME_exp_config}"_skel.yml"
  FNAME_out=${PATH_exp_config}"/"${FNAME_exp_config}"_"${rgi_id}".yml"
  cp $FNAME_in $FNAME_out

  # change RGI ID
  sed -i "s/\"skel\"/\"$rgi_id\"/g" $FNAME_out

  cd ..
  python ${exec_script} --config "experiments/${FNAME_out}"
  cd -
  
done<${list_fname}




