#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:05:29 2024

@author: danl
"""

import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Time setup (global to all datasets)
start_datetime_data = pd.Timestamp('2022-07-19 00:00:00')
time_interval = pd.Timedelta(hours=1)
start_time_desired = pd.Timestamp('2022-07-20 00:00:00')
end_time_desired = pd.Timestamp('2022-07-23 00:00:00')
start_index = int((start_time_desired - start_datetime_data) / time_interval)
end_index = int((end_time_desired - start_datetime_data) / time_interval)

variables_to_read = ["TC_URB", "TB_URB", "TG_URB", "TA_URB", "TR_URB", "TS_URB", 
                     "ALPHAC_URB2D", "ALPHAB_URB2D", "ALPHAG_URB2D", "ALPHAR_URB2D", 
                     "UTYPE_URB", "LU_INDEX", "FRC_URB2D",
                     "T2", "TSK", "HFX"]


# Organize folders and AH values for cases
folder_paths_cases = {
    "Case 1": [
        "T2_test/T2_test_no_AH",
        "T2_test/T2_test_AH_option5",
        "T2_test/T2_test_AH_option3",
        "T2_test/T2_test_AH_option2"
    ],
    "Case 2": [
        "T2_test/T2_test_rtc_no_AH",
        "T2_test/T2_test_AH_option5_rtc",
        "T2_test/T2_test_AH_option3_rtc",
        "T2_test/T2_test_AH_option2_rtc"
    ],
    "Case 3": [
        "T2_test/T2_test_ch100_no_AH",
        "T2_test/T2_test_AH_option5_ch100",
        "T2_test/T2_test_AH_option3_ch100",
        "T2_test/T2_test_AH_option2_ch100"
    ],
    "Case 4": [
        "T2_test/T2_test_rtc_ch100_no_AH",
        "T2_test/T2_test_AH_option5_rtc_ch100",
        "T2_test/T2_test_AH_option3_rtc_ch100",
        "T2_test/T2_test_AH_option2_rtc_ch100"    ]    
}

AH_value = 100

AH_values_cases = {
    "Case 1": [0, 0, 0, AH_value],
    "Case 2": [0, 0, 0, AH_value],
    "Case 3": [0, 0, 0, AH_value],
    "Case 4": [0, 0, 0, AH_value],   
}

converstion_factor = 0.001 * 0.24 / 1004.5


#%% read in the data

def process_dataset(folder_path, AH):
    # Read in the data
    ds = xr.open_mfdataset(os.path.join(folder_path, "wrfout_d03*"), combine='nested', concat_dim='Time')
    
    # Select the time range
    ds_selected = ds.sel(Time=slice(start_index, end_index))
    ds_ana = ds_selected[variables_to_read].load()

    # Assigning ROOF_WIDTH, ROAD_WIDTH, ZR_TBLs ...
    
    ds_ana['ROOF_WIDTH'] = xr.full_like(ds_ana['UTYPE_URB'], fill_value=np.nan, dtype=float)
    ds_ana['ROOF_WIDTH'] = ds_ana['ROOF_WIDTH'].where(ds_ana['UTYPE_URB'] != 1, 8.3)
    ds_ana['ROOF_WIDTH'] = ds_ana['ROOF_WIDTH'].where(ds_ana['UTYPE_URB'] != 2, 9.4)
    ds_ana['ROOF_WIDTH'] = ds_ana['ROOF_WIDTH'].where(ds_ana['UTYPE_URB'] != 3, 10)
    
    ds_ana['ROAD_WIDTH'] = xr.full_like(ds_ana['UTYPE_URB'], fill_value=np.nan, dtype=float)
    ds_ana['ROAD_WIDTH'] = ds_ana['ROAD_WIDTH'].where(ds_ana['UTYPE_URB'] != 1, 8.3)
    ds_ana['ROAD_WIDTH'] = ds_ana['ROAD_WIDTH'].where(ds_ana['UTYPE_URB'] != 2, 9.4)
    ds_ana['ROAD_WIDTH'] = ds_ana['ROAD_WIDTH'].where(ds_ana['UTYPE_URB'] != 3, 10)

    ds_ana['ZR_TBL'] = xr.full_like(ds_ana['UTYPE_URB'], fill_value=np.nan, dtype=float)
    ds_ana['ZR_TBL'] = ds_ana['ZR_TBL'].where(ds_ana['UTYPE_URB'] != 1, 5)
    ds_ana['ZR_TBL'] = ds_ana['ZR_TBL'].where(ds_ana['UTYPE_URB'] != 2, 7.5)
    ds_ana['ZR_TBL'] = ds_ana['ZR_TBL'].where(ds_ana['UTYPE_URB'] != 3, 10)

    # Calculation for HGT_TBL
    ds_ana['HGT_TBL'] = ds_ana['ZR_TBL'] / (ds_ana['ROAD_WIDTH'] + ds_ana['ROOF_WIDTH'])

    # Calculation for R_TBL
    ds_ana['R_TBL'] = ds_ana['ROOF_WIDTH'] / (ds_ana['ROAD_WIDTH'] + ds_ana['ROOF_WIDTH'])

    # Calculation for RW_TBL
    ds_ana['RW_TBL'] = 1.0 - ds_ana['R_TBL']

    # Calculation for W_TBL
    ds_ana['W_TBL'] = 2.0 * 1.0 * ds_ana['HGT_TBL']  
    
    # Calculation for TC_URB_DIAG

    ds_ana['TC_URB_DIAG'] = (ds_ana['RW_TBL']*ds_ana['TA_URB']*ds_ana['ALPHAC_URB2D'] + \
                             ds_ana['RW_TBL']*ds_ana['TG_URB']*ds_ana['ALPHAG_URB2D'] + \
                             ds_ana['W_TBL']*ds_ana['TB_URB']*ds_ana['ALPHAB_URB2D'] + AH*converstion_factor) / \
                             (ds_ana['RW_TBL']*ds_ana['ALPHAC_URB2D'] + 
                              ds_ana['RW_TBL']*ds_ana['ALPHAG_URB2D'] + 
                              ds_ana['W_TBL']*ds_ana['ALPHAB_URB2D'])
                             
    ds_ana['dTC_URB_DIAG_dAH'] = (converstion_factor) / \
                             (ds_ana['RW_TBL']*ds_ana['ALPHAC_URB2D'] + 
                              ds_ana['RW_TBL']*ds_ana['ALPHAG_URB2D'] + 
                              ds_ana['W_TBL']*ds_ana['ALPHAB_URB2D'])                             
                             
                             
    # create time average outputs for TC_URB_DIAG and TC_URB (masked)
                            
    
    #mask = ds_ana['UTYPE_URB'] != 0
    ds_ana['TC_URB_MASK'] = ds_ana['TC_URB'].where(ds_ana['UTYPE_URB'] != 0)
    ds_ana['tc_urb_avg_time'] = ds_ana['TC_URB_MASK'].mean(dim='Time')
    ds_ana['tc_urb_diag_avg_time'] = ds_ana['TC_URB_DIAG'].mean(dim='Time') 
    
    #ds_ana['dTC_URB_DIAG_dAH_time'] = ds_ana['dTC_URB_DIAG_dAH'].mean(dim='Time')
    
    #ds_ana['ALPHAC_URB2D_time'] = ds_ana['ALPHAC_URB2D'].mean(dim='Time')/converstion_factor 
    #ds_ana['ALPHAG_URB2D_time'] = ds_ana['ALPHAG_URB2D'].mean(dim='Time')/converstion_factor 
    #ds_ana['ALPHAB_URB2D_time'] = ds_ana['ALPHAB_URB2D'].mean(dim='Time')/converstion_factor 
     
    return ds_ana


def process_dataset_rtc(folder_path, AH):
    # Read in the data
    ds = xr.open_mfdataset(os.path.join(folder_path, "wrfout_d03*"), combine='nested', concat_dim='Time')
    
    # Select the time range
    ds_selected = ds.sel(Time=slice(start_index, end_index))
    ds_ana = ds_selected[variables_to_read].load()

    # Assigning ROOF_WIDTH, ROAD_WIDTH, ZR_TBLs ...
    
    ds_ana['ROOF_WIDTH'] = xr.full_like(ds_ana['UTYPE_URB'], fill_value=np.nan, dtype=float)
    ds_ana['ROOF_WIDTH'] = ds_ana['ROOF_WIDTH'].where(ds_ana['UTYPE_URB'] != 1, 8.3)
    ds_ana['ROOF_WIDTH'] = ds_ana['ROOF_WIDTH'].where(ds_ana['UTYPE_URB'] != 2, 9.4)
    ds_ana['ROOF_WIDTH'] = ds_ana['ROOF_WIDTH'].where(ds_ana['UTYPE_URB'] != 3, 10)
    
    ds_ana['ROAD_WIDTH'] = xr.full_like(ds_ana['UTYPE_URB'], fill_value=np.nan, dtype=float)
    ds_ana['ROAD_WIDTH'] = ds_ana['ROAD_WIDTH'].where(ds_ana['UTYPE_URB'] != 1, 8.3)
    ds_ana['ROAD_WIDTH'] = ds_ana['ROAD_WIDTH'].where(ds_ana['UTYPE_URB'] != 2, 9.4)
    ds_ana['ROAD_WIDTH'] = ds_ana['ROAD_WIDTH'].where(ds_ana['UTYPE_URB'] != 3, 10)

    ds_ana['ZR_TBL'] = xr.full_like(ds_ana['UTYPE_URB'], fill_value=np.nan, dtype=float)
    ds_ana['ZR_TBL'] = ds_ana['ZR_TBL'].where(ds_ana['UTYPE_URB'] != 1, 5)
    ds_ana['ZR_TBL'] = ds_ana['ZR_TBL'].where(ds_ana['UTYPE_URB'] != 2, 7.5)
    ds_ana['ZR_TBL'] = ds_ana['ZR_TBL'].where(ds_ana['UTYPE_URB'] != 3, 10)

    # Calculation for HGT_TBL
    ds_ana['HGT_TBL'] = ds_ana['ZR_TBL'] / (ds_ana['ROAD_WIDTH'] + ds_ana['ROOF_WIDTH'])

    # Calculation for R_TBL
    ds_ana['R_TBL'] = ds_ana['ROOF_WIDTH'] / (ds_ana['ROAD_WIDTH'] + ds_ana['ROOF_WIDTH'])

    # Calculation for RW_TBL
    ds_ana['RW_TBL'] = 1.0 - ds_ana['R_TBL']

    # Calculation for W_TBL
    ds_ana['W_TBL'] = 2.0 * 1.0 * ds_ana['HGT_TBL']  
    
    # Calculation for TC_URB_DIAG

    ds_ana['TC_URB_DIAG'] = (ds_ana['TA_URB']*ds_ana['ALPHAC_URB2D'] + \
                             ds_ana['RW_TBL']*ds_ana['TG_URB']*ds_ana['ALPHAG_URB2D'] + \
                             ds_ana['R_TBL']*ds_ana['TR_URB']*ds_ana['ALPHAR_URB2D'] + \
                             ds_ana['W_TBL']*ds_ana['TB_URB']*ds_ana['ALPHAB_URB2D'] + AH*converstion_factor) / \
                             (ds_ana['ALPHAC_URB2D'] +
                              ds_ana['RW_TBL']*ds_ana['ALPHAG_URB2D'] + 
                              ds_ana['R_TBL']*ds_ana['ALPHAR_URB2D'] +                              
                              ds_ana['W_TBL']*ds_ana['ALPHAB_URB2D'])
                  
    denominator = (ds_ana['ALPHAC_URB2D'] +
               ds_ana['RW_TBL']*ds_ana['ALPHAG_URB2D'] + 
               ds_ana['R_TBL']*ds_ana['ALPHAR_URB2D'] +                              
               ds_ana['W_TBL']*ds_ana['ALPHAB_URB2D'])

    # Check if there are any zeros in the denominator
    if (denominator == 0).any():
        print("Warning: Zero in the denominator found!")    
                                               
    ds_ana['TC_URB_DIAG'] = ds_ana['TC_URB_DIAG'].where(np.isfinite(ds_ana['TC_URB_DIAG']), np.nan)
                         
    # create time average outputs for TC_URB_DIAG and TC_URB (masked)                               
    #mask = ds_ana['UTYPE_URB'] != 0
    ds_ana['TC_URB_MASK'] = ds_ana['TC_URB'].where(ds_ana['UTYPE_URB'] != 0)
    ds_ana['tc_urb_avg_time'] = ds_ana['TC_URB_MASK'].mean(dim='Time')
    ds_ana['tc_urb_diag_avg_time'] = ds_ana['TC_URB_DIAG'].mean(dim='Time') 
    
    #ds_ana['dTC_URB_DIAG_dAH_time'] = ds_ana['dTC_URB_DIAG_dAH'].mean(dim='Time')
    
    #ds_ana['ALPHAC_URB2D_time'] = ds_ana['ALPHAC_URB2D'].mean(dim='Time')/converstion_factor 
    #ds_ana['ALPHAG_URB2D_time'] = ds_ana['ALPHAG_URB2D'].mean(dim='Time')/converstion_factor 
    #ds_ana['ALPHAB_URB2D_time'] = ds_ana['ALPHAB_URB2D'].mean(dim='Time')/converstion_factor 
     
    return ds_ana

# Process datasets for each case
datasets = {case: [] for case in folder_paths_cases}

for case, folder_paths in folder_paths_cases.items():
    for i, folder in enumerate(folder_paths):
        AH = AH_values_cases[case][i]
        # Use the original process_dataset for Case 1 and Case 2
        if case in ["Case 1", "Case 3"]:
            ds_processed = process_dataset(folder, AH)
        # Use the new process_dataset_v2 for Case 3 and Case 4
        else:
            ds_processed = process_dataset_rtc(folder, AH)
        datasets[case].append(ds_processed)
        
  

#%% 
T2_values = {}
T2_std_dev = {}
TSK_values = {}
TSK_std_dev = {}

utypes = ["all", 1, 2, 3]  # List of urban types to analyze

for case, ds_list in datasets.items():
    baseline_ds = ds_list[0]  # Baseline dataset

    for utype in utypes:
        # Filtering based on urban type if utype is not 'all'
        baseline_filtered = baseline_ds.where(baseline_ds['UTYPE_URB'] != 0, drop=True) if utype == "all" else baseline_ds.where(baseline_ds['UTYPE_URB'] == utype, drop=True)
        
        baseline_t2_urb_diag_avg = baseline_filtered['T2'].mean(dim='Time')
        baseline_ts_urb_diag_avg = baseline_filtered['TSK'].mean(dim='Time')
        
        for i, ds in enumerate(ds_list[1:], 1):  # Starting from the second dataset in the list
            ds_filtered = ds.where(ds['UTYPE_URB'] != 0, drop=True) if utype == "all" else ds.where(ds['UTYPE_URB'] == utype, drop=True)            
            
            overall_diff_time_avg = ds_filtered['T2'].mean(dim='Time') 
            overall_spatial_avg_diff = ((overall_diff_time_avg - baseline_t2_urb_diag_avg) / AH_value).mean(['south_north', 'west_east']).values
            overall_spatial_std_dev = ((overall_diff_time_avg - baseline_t2_urb_diag_avg) / AH_value).std(['south_north', 'west_east']).values

            T2_values[(case, i, utype)] = overall_spatial_avg_diff
            T2_std_dev[(case, i, utype)] = overall_spatial_std_dev
            
            overall_diff_time_avg = ds_filtered['TSK'].mean(dim='Time') 
            overall_spatial_avg_diff = ((overall_diff_time_avg - baseline_ts_urb_diag_avg) / AH_value).mean(['south_north', 'west_east']).values
            overall_spatial_std_dev = ((overall_diff_time_avg - baseline_ts_urb_diag_avg) / AH_value).std(['south_north', 'west_east']).values

            TSK_values[(case, i, utype)] = overall_spatial_avg_diff
            TSK_std_dev[(case, i, utype)] = overall_spatial_std_dev

    
#%%fig, axes = plt.subplots(3, 2, figsize=(12, 18))

# Assuming datasets, TSK_values, TSK_std_dev, T2_values, T2_std_dev are defined somewhere in your code

labels = ["CASE 1", "CASE 2", "CASE 3", "CASE 4"]
colors = ['red', 'blue', 'green', 'purple']
axis_label_font_size = 14
title_font_size = 14
tick_label_font_size = 14
#urban_types = ["all", 1, 2, 3]  # List of urban types
cases = list(datasets.keys())

titles = ['(a) revised method 1', '(b) revised method 1', '(c) method 2', '(d) method 2', '(e) method 3', '(f) method 3']

for utype in utypes:  # Loop through each urban type
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))  # Create a new figure for each urban type
    for i in range(3):  # Assuming 3 different i values (1, 2, 3)
        # Plot TSK values and standard deviations for this i
        tsk_values = [TSK_values[(case, i+1, utype)] for case in cases]
        tsk_std_devs = [TSK_std_dev[(case, i+1, utype)] for case in cases]
        axes[i, 0].bar(labels, tsk_values, color=colors, yerr=tsk_std_devs, capsize=5)
        axes[i, 0].set_title(titles[2*(i)+0], fontsize=title_font_size)
        axes[i, 0].set_ylabel('dT$_{S}$/dQ$_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
        axes[i, 0].tick_params(axis='x', labelrotation=45)
        axes[i, 0].set_ylim([0, 0.05])  # Adjust as necessary
        axes[i, 0].tick_params(axis='both', which='major', labelsize=tick_label_font_size)
        
        # Plot T2 values and standard deviations for this i
        t2_values = [T2_values[(case, i+1, utype)] for case in cases]
        t2_std_devs = [T2_std_dev[(case, i+1, utype)] for case in cases]
        axes[i, 1].bar(labels, t2_values, color=colors, yerr=t2_std_devs, capsize=5)
        axes[i, 1].set_title(titles[2*(i)+1], fontsize=title_font_size)
        axes[i, 1].set_ylabel('dT$_{2}$/dQ$_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
        axes[i, 1].tick_params(axis='x', labelrotation=45)
        axes[i, 1].set_ylim([0, 0.04])  # Adjust as necessary
        axes[i, 1].tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    
    plt.tight_layout()
    plt.show()
    
    # Save each figure with a unique name that includes the urban type
    fig.savefig(f"T2_figures/T2_test_4cases_utype_{utype}.png", dpi=300)

