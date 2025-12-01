#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
import shutil
import os
import subprocess
from netCDF4 import Dataset
import datetime
import numpy as np
import sys


def extract_repo(in_fpath,in_fname,experiment):
    '''
    Extract single climate forcing from CORDEX netCDF.

    Authors: Johannes J. Fürst

    Args:
        in_fpath(str)              - file path to input netCDF
        in_fname(str)              - file name of input netCDF
        experiment(int)            - integer indicating experiment
        

    Returns: 
    '''

    infile  = os.path.join(in_fpath,in_fname)
    #print(experiment)
    experiment=int(experiment)


    with Dataset(infile, "r") as src:
        # Read dimensions
        time_len = len(src.dimensions["time"])

        # Extract slices for the given experiment and member
        time = src.variables["time"][:]
        temp = src.variables["temp"][:,:,:]
        temp_std = src.variables["temp_std"][:,:,:]
        prcp = src.variables["prcp"][:,:,:]
        experiments = src.variables["experiment"][:]
        members = src.variables["member"][:]

        # Check for outliers
        temp[np.abs(temp)>100]=np.nan
        prcp[np.abs(prcp)>100]=np.nan

        ## BIAS CORRECTION
        ## extract time baselines
        #print(np.asarray(temp_W5E5))
        time_units      = "days since 1801-01-01 00:00:00"

        ## reference period
        year_0 = 1970
        year_1 = 1999

        date_ref = datetime.datetime(1800, 1, 1)
        date_10   = datetime.datetime(1970, 1, 1)
        date_11   = datetime.datetime(1999, 12, 31)
        date_20   = datetime.datetime(2090, 1, 1)
        date_21   = datetime.datetime(2099, 12, 31)
        #print('DATE', date_0, date_1)

        days_10 = (date_10 - date_ref).days
        days_11 = (date_11 - date_ref).days
        days_20 = (date_20 - date_ref).days
        days_21 = (date_21 - date_ref).days
        #print('NORMAL', days_0,days_1)

        mask1 = (time >= days_10) & (time <= days_11)
        mask2 = (time >= days_20) & (time <= days_21)
        #print(np.nansum(mask),np.nansum(mask_W5E5))

        # mean temp/prcp per scenario

        #print(np.shape(temp[experiment,:,mask2]))
        #print(np.shape(prcp[experiment,:,mask2]))
        temp_select = np.asarray(temp[experiment,:,mask2])
        prcp_select = np.asarray(prcp[experiment,:,mask2])

        temp_mean  = np.nanmean(np.nanmean(temp_select,axis=-1),axis=-1)
        prcp_mean  = np.nanmean(np.nansum(prcp_select,axis=0),axis=0)/np.nansum(mask2)

        # stdev temp/prcp per scenario
        temp_std   = np.nanstd(np.nanmean(temp_select,axis=0),axis=0)
        prcp_std   = np.nanstd(np.nansum(prcp_select,axis=0),axis=0)/np.nansum(mask2)

        #print(np.shape(temp[experiment,:,mask2]))
        temp_dist  = np.abs(np.nanmean(temp_select,axis=0)-temp_mean)
        temp_index = np.where(temp_dist == np.nanmin(temp_dist))

        prcp_dist  = np.abs(np.nansum(prcp_select,axis=0)/np.nansum(mask2)-prcp_mean)
        prcp_index = np.where(prcp_dist == np.nanmin(prcp_dist))
        
        tp_dist    = temp_dist/temp_std+prcp_dist/prcp_std
        tp_index   = np.where(tp_dist == np.nanmin(tp_dist))

        if experiments[experiment] == "historical" :
            member_idx    = -9999
            member_select = "NaN"
        else:
            member_idx    = int(tp_index[0][0])
            member_select = members[member_idx]
      
        #print(np.shape(temp_dist),np.shape(prcp_dist),np.shape(tp_dist),np.nansum(mask2))
        #print(np.shape(temp_std),np.shape(prcp_std))
        #print('DISTIES : ',np.nanmin(temp_dist/temp_std),np.nanmin(prcp_dist/prcp_std))
        #print('INDIES  : ',temp_index, prcp_index, tp_index)
        #print('INDIE', member_idx)
        #print(np.shape(temp))
        #print('TEMPS : ',temp_mean,np.nanmean(temp[experiment,member_idx,mask2]),np.nanmean(temp[experiment,temp_index,mask2]),np.nanmean(temp[experiment,prcp_index,mask2]))
        #print('PRCPS : ',prcp_mean,np.nanmean(prcp[experiment,member_idx,mask2]),np.nanmean(prcp[experiment,temp_index,mask2]),np.nanmean(prcp[experiment,prcp_index,mask2]))
        #print(np.shape(temp))
        #for ee in np.arange(0,np.shape(temp)[0],1):
        #    if ee == 2:
        #        continue
        #    temp_dist  = np.abs(np.nanmean(temp[ee,:,mask2],axis=-1)-temp_mean[ee])
        #    temp_index = np.where(temp_dist == np.nanmin(temp_dist))
        #
        #    prcp_dist  = np.abs(np.nansum(prcp[ee,:,mask2],axis=-1)/np.nansum(mask2)-prcp_mean[ee])
        #    prcp_index = np.where(prcp_dist == np.nanmin(prcp_dist))
        #
        ##    print(np.shape(temp_dist),np.shape(prcp_dist))
        ##    print(np.shape(temp_std),np.shape(prcp_std))
        #
        #    tp_dist    = temp_dist/temp_std[ee]+prcp_dist/prcp_std[ee]
        #    tp_index   = np.where(tp_dist == np.nanmin(tp_dist))

        #    print('DISTIES : ',np.nanmin(temp_dist/temp_std[ee]),np.nanmin(prcp_dist/prcp_std[ee]))
        #    print('INDIES  : ',temp_index, prcp_index, tp_index)

            #print(np.shape(temp[ee,:,mask2]),np.shape(temp[ee,:,mask1]))
            #print(np.nanmax(np.nanmax(prcp[ee,:,:])))
            #print(np.nanmean(np.nanmean(np.asarray(prcp[ee,:,mask2]),axis=0),axis=0)*12/1000.)
            ##print(np.nanmean(np.nansum(np.asarray(prcp[ee,:,mask2]),axis=0)/10,axis=0))
            #temp_diff=np.nanmean(np.asarray(temp[ee,:,mask2]),axis=0)-np.nanmean(np.asarray(temp[ee,:,mask1]),axis=0)
            #prcp_diff=np.nanmean(np.asarray(prcp[ee,:,mask2]),axis=0)-np.nanmean(np.asarray(prcp[ee,:,mask1]),axis=0)*12
            #print(np.shape(np.asarray(temp[ee,:,:])))
            #temp_mean      = np.nanmean(np.nanmean(np.asarray(temp[ee,:,mask2]),axis=0),axis=0)
            #prcp_mean      = np.nansum(np.nanmean(np.asarray(prcp[ee,:,mask2]),axis=0),axis=0)/np.nansum(mask2))
            #print(np.shape(temp_mean))
            #temp_diff_mean = np.nanmean(temp_mean[mask2])#-np.nanmean(temp_mean[mask1])
            #prcp_fac_mean  = (np.nansum(prcp_mean[mask2])/np.nansum(mask2))#/(np.nansum(prcp_mean[mask1])/np.nansum(mask1))
            #temp_diff_mean = np.nanmean(temp_diff,axis=0)
            #prcp_diff_mean = np.nanmean(prcp_diff,axis=0)
            #temp_diff_mean=np.nanmean(np.nanmean(np.asarray(temp[ee,:,mask2]),axis=0)-np.nanmean(np.asarray(temp[ee,:,mask1]),axis=0),axis=0)
            #prcp_diff_mean=np.nanmean(np.nanmean(np.asarray(prcp[ee,:,mask2]),axis=0)-np.nanmean(np.asarray(prcp[ee,:,mask1]),axis=0),axis=0)*12
            #print(np.nansum(mask2),temp_diff_mean,prcp_fac_mean)

        print(member_idx, experiments[experiment], member_select)
        return member_idx, experiments[experiment], member_select

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run glacier projection experiments.')

    # Add arguments for parameters
    parser.add_argument('--in_fpath', type=str,
                        default='./',
                        help='File path of input netCDF.')

    parser.add_argument('--in_fname', type=str,
                        default='./',
                        help='File name of input netCDF.')

    parser.add_argument('--experiment', type=int,
                        default=0,
                        help='Experiment integer.')

    # Parse arguments
    args = parser.parse_args()

    member_id = extract_repo(args.in_fpath, args.in_fname, args.experiment)

