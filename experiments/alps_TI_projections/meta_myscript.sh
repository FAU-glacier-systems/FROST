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
conda activate frost_env_IGM3

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80


# setting
exec_script='frost_pipeline.py'
list_fname='todo_codeskel.txt'
PATH_exp_config='./experiments/eALPS/'
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
  #cd ../
  #cp ${exec_script}_skel.sh ${exec_script}_$p.sh
  #sed -i "s/\"skel\"/\"$p\"/g" ${exec_script}_$p.sh
  ##/home/hpc/gwgi/gwgi17/cronjobs/bin/creation_chain_WOODY.sh ${RGI_region} $p
  #chmod u+x ${exec_script}_$p.sh
  #bash ./${exec_script}_$p.sh
  python ${exec_script} --config "${FNAME_out}"
  cd -

  ## Clean-up directory
  #cd ../Data/Glaciers/${rgi_id}/Ensemble/
  #pwd
  #ls
  #dir_names=`ls -d M*`
  #for adir in $dir_names ; do
  # #echo $adir $dir_names
  # rm -f ${adir}/climate_historical.nc ${adir}/input.nc
  # rm -f ${adir}/clim_1D-3D.py ${adir}/smb_oggm_TI_local.py
  # rm -f ${adir}/params_saved.json
  # rm -rf ${adir}/__pycache__
  # rm -rf ${adir}/iceflow-model
  # rm -f ${adir}/clean.sh
  #done
  #cd -
  
done<${list_fname}




