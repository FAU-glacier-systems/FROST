#!/bin/bash -l
#
#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --job-name=frost_proj
#SBATCH --output=Log/frost_batch_%j.out
#SBATCH --error=Log/frost_batch_%j.err

RGI_region=11

# Exit on error
set -e

module add python
conda activate igm3

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80


# setting
exec_script='frost_pipeline.py'
list_fname='todo_codeskel.txt'
PATH_exp_config='alps_TI_projections/eALPS'
PATH_results='../data/results/alps_TI_projections/glaciers'
FNAME_exp_config='config_TI'

# Folgender Befehl arbeitet alle Inputs aus sequence.txt ab und liest immer 6
# gleichzeitig laufen.
# Sie koennen als Befehl im Prinzip alles machen. Der Platxhalter {} steht fuer 
# die Zahl oder den String der aus dem input Listenfile kommt.
# Wenn alles abgearbeitet ist kehrt das parallel zurueck :-)
# Mit -u wir aller stdout und stdin in eine Pipe geleitet. Das ist schneller.
# Mehr dazu in der Doku
#parallel -a ${list_fname} -j 6 -u "./myscript.sh {}"
#parallel -a ${list_fname} -j 1 -u "./myscript.sh {}"

countit=0
while read rgi_id; 
do

  countit=0

  #cd $exec_dir
  cd ..
  # glacier ID selection
  echo $rgi_id


  #/home/hpc/gwgi/gwgi17/cronjobs/bin/creation_chain_WOODY.sh ${RGI_region} $p
  #chmod u+x ${exec_script}_$p.sh
  #bash ./${exec_script}_$p.sh

    # set calibration directory (where result.json is stored)
    cal_path=${PATH_results}/${rgi_id}
    JSON_cal_path="${cal_path}/calibration_results.json"

    # set calibration directory (where output of all ensemble members is stored) 
    OUT_cal_dir="${PATH_results}/${rgi_id}/Ensemble/"

    # retrieve ensemble size
    EnKF_ensemble_size=`ls -d ${OUT_cal_dir}/Member_* | wc -l`
    echo "ensemble size" $EnKF_ensemble_size

    # select SMB memebr

    SMBoption="best"  # best, mean (default)

    cal_fpath=${PATH_results}/${rgi_id}
    cal_fname="calibration_results.json"

    obs_fpath="${PATH_results}/${rgi_id}/"
    obs_fname="observations.nc"

    mod_fpath=${OUT_cal_dir}
    mod_fname="output.nc"

    pwd
    member_id=$(python ./alps_TI_projections/select_SMB_member.py --cal_fpath $cal_fpath --cal_fname $cal_fname --obs_fpath $obs_fpath --obs_fname $obs_fname --mod_fpath $mod_fpath --mod_fname $mod_fname --option $SMBoption)

    # TIME_PERIOD: 2000 - 2020
    echo "Simulating reference period for ... : "$rgi_id ${member_id}
    start_year=2000
    end_year=2020

    # Copy W5E5 climate forcing
    in_fpath="${cal_path}"
    in_fname="climate_historical_W5E5.nc"

    out_fpath="${cal_path}"
    out_fname="climate_historical.nc"

    if [ -f ${in_fpath}/${in_fname} ]; then
      cp ${in_fpath}/${in_fname} ${out_fpath}/${out_fname}
    else
      cp ${out_fpath}/${out_fname} ${in_fpath}/${in_fname}
    fi

    flag_3D_output=true
    echo "Start projections"
    # start projection for ensemble mean (as defined in result.json)
    python ../frost/glacier_model/igm_wrapper.py --rgi_id $rgi_id --JSON_path $JSON_cal_path --member_id=${member_id} --start_year ${start_year} --end_year ${end_year} --flag_3D_output ${flag_3D_output}


    # Copy results for reference period
    in_fpath="${PATH_results}/${rgi_id}/Projection/Member_${member_id}/outputs/"
    in_fname="output.nc"

    out_fpath="${PATH_results}/${rgi_id}/Projection/Member_${member_id}/outputs/"
    out_fname="output_W5E5_$SMBoption.nc"

    cp ${in_fpath}/${in_fname} ${out_fpath}/${out_fname}

    # Copy results for reference period
    in_fpath="${PATH_results}/${rgi_id}/Projection/Member_${member_id}/outputs/"
    in_fname="output_ts.nc"

    out_fpath="${PATH_results}/${rgi_id}/Projection/"
    out_fname="output_ts_W5E5_${SMBoption}.nc"

    cp ${in_fpath}/${in_fname} ${out_fpath}/${out_fname}
    cp ${in_fpath}/${in_fname} ${in_fpath}/${out_fname}

    # Loop over climate scenarios (ee=experiment) and climate models (mm=member)
  for ee in {0..3} #{0..3} #{0..0}
    do
      in_fpath="/home/vault/gwgi/gwgifu1h/data/input/11/1/glacier_ids/"${rgi_id: -5}"/climate/CORDEX_unbiased"
      in_fname="CORDEX_merged.nc"

      echo "python alps_TI_projections/extract_CMIP_representative.py --in_fpath $in_fpath --in_fname $in_fname --experiment ${ee}"
      pwd
      result=$(python alps_TI_projections/extract_CMIP_representative.py --in_fpath $in_fpath --in_fname $in_fname --experiment ${ee})

      # read output from python script
      read -r m3D experiment3D member3D <<< "$result"
#JJF start
      echo $m3D $experiment3D $member3D
#JJF end
      
      for mm in {0..69} #{0..69} #{3..3}
      do
        flag_IGM=0

        # Copy climatic input files
        #in_fpath="/home/hpc/gwgi/gwgifu1h/fragile/data/input/11/1/glacier_ids/"${rgi_id: -5}"/climate/CORDEX_unbiased/"
        #in_fpath="${cal_path}"
        in_fpath="/home/vault/gwgi/gwgifu1h/data/input/11/1/glacier_ids/"${rgi_id: -5}"/climate/CORDEX_unbiased"
        in_fname="CORDEX_merged.nc"

        out_fpath="${cal_path}"
        out_fname="climate_historical.nc"
        result=$(python alps_TI_projections/extract_CMIP_climate_historical.py --in_fpath $in_fpath --in_fname $in_fname --out_fpath $out_fpath --out_fname $out_fname --experiment_idx $ee --member_idx $mm)

        # read output from python script
        read -r flag_IGM experiment member <<< "$result"

#        python ./batchscripts/extract_CMIP_climate_historical.py --in_fpath $in_fpath --in_fname $in_fname --out_fpath $out_fpath --out_fname $out_fname --experiment_idx $ee --member_idx $mm

        if [ "$flag_IGM" == "1" ] ; then
          echo "Starting projection of ... : "$rgi_id
          start_year=2020
          end_year=2100

          echo "flag_3D_output : " $flag_3D_output $m3D $mm

          # check if representative run
          if [ $mm == ${m3D} ] ; then
              flag_3D_output=true
          else
              flag_3D_output=false
          fi

          echo "flag_3D_output : " $flag_3D_output $m3D $mm
          # start projection for ensemble mean (as defined in result.json)
          python ../frost/glacier_model/igm_wrapper.py --rgi_id $rgi_id --JSON_path $JSON_cal_path --member_id=${member_id} --start_year ${start_year} --end_year ${end_year} --flag_3D_output ${flag_3D_output}

          bpr_fpath=$in_fpath
          bpr_fname=$in_fname

          in_fpath="${PATH_results}/${rgi_id}/Projection/Member_${member_id}/outputs/"
          in_fname="output_ts.nc"

          out_fpath="${PATH_results}/${rgi_id}/Projection/"
          out_fname="CORDEX_${SMBoption}_output_ts.nc"
          python alps_TI_projections/compile_output_ts.py --counter $countit --blueprint_fpath $bpr_fpath --blueprint_fname $bpr_fname --in_fpath $in_fpath --in_fname $in_fname --out_fpath $out_fpath --out_fname $out_fname --experiment_idx $ee --member_idx $mm

          if [ $mm == ${m3D} ] ; then
            # save 3D output
            in_fpath="${PATH_results}/${rgi_id}/Projection/Member_${member_id}/outputs/"
            in_fname="output.nc"

            out_fpath="${PATH_results}/${rgi_id}/Projection/"
            out_fname="output_"${experiment3D}"_"${member3D}".nc"

            cp ${in_fpath}/${in_fname} ${out_fpath}/${out_fname}
            rm -f ${in_fpath}/${in_fname}
          fi

          # Increase counter of valid output entries
          countit=$((countit+1))
        fi 

      done
    done

##    experiment_idx=0
##    member_idx=3
##    result=$(python ./batchscripts/extract_CMIP_climate_historical.py --in_fpath $in_fpath --in_fname $in_fname --out_fpath $out_fpath --out_fname $out_fname --experiment_idx $experiment_idx --member_idx $member_idx)
##    # read output from python script
##    read -r flag_IGM experiment member <<< "$result"
##    
##    echo "Value 1: $flag_IGM"
##    echo "Value 2: $experiment"
##    echo "Value 3: $member"
##
##    if [ "$flag_IGM" == "1" ] ; then
##      echo "Starting projection of ... : "$rgi_id
##      # start projection for ensemble mean (as defined in result.json)
##      #python ./frost/glacier_model/igm_wrapper.py --rgi_id $rgi_id --JSON_path $JSON_cal_path --member_id=${SMBoption}
##    fi
##
##    bpr_fpath=$in_fpath
##    bpr_fname=$in_fname
##
##    in_fpath="${PATH_results}/${rgi_id}/Projection/Member_${member_id}/outputs/"
##    in_fname="output_ts.nc"
##
##    out_fpath="${PATH_results}/${rgi_id}/Projection/"
##    out_fname="CORDEX_${SMBoption}_output_ts.nc"
##    python ./batchscripts/compile_output_ts.py --counter $countit --blueprint_fpath $bpr_fpath --blueprint_fname $bpr_fname --in_fpath $in_fpath --in_fname $in_fname --out_fpath $out_fpath --out_fname $out_fname --experiment_idx $experiment_idx --member_idx $member_idx
##
   
   #


    ## start projections for each ensemble member
    #for ((ii=1;ii<=EnKF_ensemble_size;ii++)); do
    #    jj=$((ii-1))
    #    # reset counter for output
    #    countit=0
    #    echo "Count : " $ii "(Member_$jj)"
    # 
    #    python ./Scripts/IGM_wrapper.py --rgi_id $rgi_id --JSON_path $JSON_cal_path --member_id=$jj
    #done
  #rm -f ./${exec_script}_$p.sh
  cd -

  # Clean-up directory
  #cd ../Data/Glaciers/${rgi_id}/Projection/
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

#echo "I am in"

#rm -f $list_fname

# Der folgende Aufruf kann dann z.B. in myscript.sh sein mit den Befehlen oder dem Skriptaufruf das vorher
# das Verzeichnis und alle notwendigen Dateinen vorbereitet. Am Besten alles nach Woody Home packen
# und bitte das Parallele Filesystem nicht verwenden. NAch dem Aufruf kann man dann mit einem Skript das Ergebnis
# extrahieren und alles aufraeumen.
#mpiexec.hydra -n 2  -envall ElmerSolver_mpi


#IDs=`cat ${list_fname}`
#
#for anID in $IDs
#do
#
#  # Remove Elmer grid files
#  rm -rf ${storage_path}/${anID}/mesh/grid
#
#
#  cd ${pwd_path}
#  # Reproject and clean up
#  ./closure_chain_WOODY.sh ${anID} ${RGI_region}
#
#done


