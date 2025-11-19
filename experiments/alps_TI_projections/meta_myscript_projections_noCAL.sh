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
#conda activate frost_env_igm3
conda activate frost_env_IGM3

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80


# setting
exec_script='frost_pipeline.py'
list_fname='todo_codeskel_noCAL.txt'
PATH_exp_config='./experiments/eALPS/'
PATH_results='./data/results/eALPS/glaciers/'
FNAME_exp_config='config_TI'

# Folgender Befehl arbeitet alle Inputs aus sequence.txt ab und laest immer 6
# gleichzeitig laufen.
# Sie koennen als Befehl im Prinzip alles machen. Der Platxhalter {} steht fuer 
# die Zahl oder den String der aus dem input Listenfile kommt.
# Wenn alles abgearbeitet ist kehrt das parallel zurueck :-)
# Mit -u wir aller stdout und stdin in eine Pipe geleitet. Das ist schneller.
# Mehr dazu in der Doku
#parallel -a ${list_fname} -j 6 -u "./myscript.sh {}"
#parallel -a ${list_fname} -j 1 -u "./myscript.sh {}"

countit=0
while read p; 
do

  countit=0

  my_string=`echo $p`
  # Set IFS to comma and read into an array
  IFS=',' read -ra IDs <<< "$my_string"
  # Print the result
  rgi_id=`echo ${IDs[0]}`
  rgi_id_bigbrother=`echo ${IDs[1]}`
  echo $rgi_id $rgi_id_bigbrother

  cd ..

  # Copy SMB calibration from nearest big-brother glacier
  #dir_in=`ls -d ./Experiments/${rgi_id_bigbrother}/Ensemble_*`
  #bn_dir_in=`basename $dir_in`
  #dir_out="./Experiments/${rgi_id}/${bn_dir_in}"
  dir_in=${PATH_results}/${rgi_id_bigbrother}
  dir_out=${PATH_results}/${rgi_id}

  if [ -d ${dir_out}/calibration_results.json ] ; then
    echo "Ensemble results folder exists already."
    echo "Nothing is copied."
  else
    echo "Let it go ... let it go ... "

    ## create directory
    #mkdir $dir_out

    # copy JSON file
    cp ${dir_in}/calibration_results.json $dir_out
  fi

  #/home/hpc/gwgi/gwgi17/cronjobs/bin/creation_chain_WOODY.sh ${RGI_region} $p
  #chmod u+x ${exec_script}_$p.sh
  #bash ./${exec_script}_$p.sh

    # set calibration directory (where result.json is stored)
    cal_path=${PATH_results}/${rgi_id}
    JSON_cal_path="${cal_path}/calibration_results.json"

    # set calibration directory (where output of all ensemble members is stored) 
    OUT_cal_dir="${PATH_results}/${rgi_id_bigbrother}/Ensemble/"

    # retrieve ensemble size
    EnKF_ensemble_size=`ls -d ${OUT_cal_dir}/Member_* | wc -l`
    echo "ensemble size" $EnKF_ensemble_size

    # TIME_PERIOD: 2000 - 2020
    echo "Simulating reference period for ... : "$rgi_id
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
 
    #cp ${in_fpath}/${in_fname} ${out_fpath}/${out_fname} 

    flag_3D_output=true

    # start projection for ensemble mean (as defined in result.json)
    python ./frost/glacier_model/igm_wrapper.py --rgi_id $rgi_id --JSON_path $JSON_cal_path --member_id="mean" --start_year ${start_year} --end_year ${end_year} --flag_3D_output ${flag_3D_output}


    # Copy results for reference period
    in_fpath="${PATH_results}/${rgi_id}/Projection/Member_mean/outputs/"
    in_fname="output.nc"

    out_fpath="${PATH_results}/${rgi_id}/Projection/Member_mean/outputs/"
    out_fname="output_W5E5_mean.nc"

    cp ${in_fpath}/${in_fname} ${out_fpath}/${out_fname}

    # Copy results for reference period
    in_fpath="${PATH_results}/${rgi_id}/Projection/Member_mean/outputs/"
    in_fname="output_ts.nc"

    out_fpath="${PATH_results}/${rgi_id}/Projection/"
    out_fname="output_ts_W5E5_mean.nc"

    cp ${in_fpath}/${in_fname} ${out_fpath}/${out_fname}
    cp ${in_fpath}/${in_fname} ${in_fpath}/${out_fname}

    ## Check if volume is already zero in 2020
    #flag_VOL=$(python ./batchscripts/check_volume2020.py --in_fpath $in_fpath --in_fname $in_fname)


    # Loop over climate scenarios (ee=experiment) and climate models (mm=member)
    for ee in {0..3} #{0..3} #{0..0}
    do

      in_fpath="/home/vault/gwgi/gwgifu1h/data/input/11/1/glacier_ids/"${rgi_id: -5}"/climate/CORDEX_unbiased"
      in_fname="CORDEX_merged.nc"

      echo "python ./batchscripts/extract_CMIP_representative.py --in_fpath $in_fpath --in_fname $in_fname --experiment ${ee}"

      result=$(python ./batchscripts/extract_CMIP_representative.py --in_fpath $in_fpath --in_fname $in_fname --experiment ${ee})

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
        result=$(python ./batchscripts/extract_CMIP_climate_historical.py --in_fpath $in_fpath --in_fname $in_fname --out_fpath $out_fpath --out_fname $out_fname --experiment_idx $ee --member_idx $mm)

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

          #if [ $mm == ${m3D} ] ; then
              # start projection for ensemble mean (as defined in result.json)
              python ./frost/glacier_model/igm_wrapper.py --rgi_id $rgi_id --JSON_path $JSON_cal_path --member_id="mean" --start_year ${start_year} --end_year ${end_year} --flag_3D_output ${flag_3D_output} 
          #else
          #  if [ "flag_VOL" == 1 ]; then
          #    # start projection for ensemble mean (as defined in result.json)
          #    python ./frost/glacier_model/igm_wrapper.py --rgi_id $rgi_id --JSON_path $JSON_cal_path --member_id="mean" --start_year ${start_year} --end_year ${end_year} --flag_3D_output ${flag_3D_output}
          #  fi
          #fi

          bpr_fpath=$in_fpath
          bpr_fname=$in_fname

          in_fpath="${PATH_results}/${rgi_id}/Projection/Member_mean/outputs/"
          in_fname="output_ts.nc"

          out_fpath="${PATH_results}/${rgi_id}/Projection/"
          out_fname="CORDEX_mean_output_ts.nc"
          python ./batchscripts/compile_output_ts.py --counter $countit --blueprint_fpath $bpr_fpath --blueprint_fname $bpr_fname --in_fpath $in_fpath --in_fname $in_fname --out_fpath $out_fpath --out_fname $out_fname --experiment_idx $ee --member_idx $mm

          if [ $mm == ${m3D} ] ; then
            # save 3D output
            in_fpath="${PATH_results}/${rgi_id}/Projection/Member_mean/outputs/"
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

#    experiment_idx=0
#    member_idx=3
#    result=$(python ./batchscripts/extract_CMIP_climate_historical.py --in_fpath $in_fpath --in_fname $in_fname --out_fpath $out_fpath --out_fname $out_fname --experiment_idx $experiment_idx --member_idx $member_idx)
#    # read output from python script
#    read -r flag_IGM experiment member <<< "$result"
#    
#    echo "Value 1: $flag_IGM"
#    echo "Value 2: $experiment"
#    echo "Value 3: $member"
#
#    if [ "$flag_IGM" == "1" ] ; then
#      echo "Starting projection of ... : "$rgi_id
#      # start projection for ensemble mean (as defined in result.json)
#      #python ./frost/glacier_model/igm_wrapper.py --rgi_id $rgi_id --JSON_path $JSON_cal_path --member_id="mean"
#    fi
#
#    bpr_fpath=$in_fpath
#    bpr_fname=$in_fname
#
#    in_fpath="${PATH_results}/${rgi_id}/Projection/Member_mean/outputs/"
#    in_fname="output_ts.nc"
#
#    out_fpath="${PATH_results}/${rgi_id}/Projection/"
#    out_fname="CORDEX_mean_output_ts.nc"
#    python ./batchscripts/compile_output_ts.py --counter $countit --blueprint_fpath $bpr_fpath --blueprint_fname $bpr_fname --in_fpath $in_fpath --in_fname $in_fname --out_fpath $out_fpath --out_fname $out_fname --experiment_idx $experiment_idx --member_idx $member_idx

    
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


