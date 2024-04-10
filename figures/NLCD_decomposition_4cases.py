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

variables_to_read = ["TC_URB", "TB_URB", "TG_URB","TR_URB", "TA_URB", 
                     "ALPHAC_URB2D", "ALPHAB_URB2D", "ALPHAG_URB2D", "ALPHAR_URB2D",
                      "UTYPE_URB", "LU_INDEX"]


# Organize folders and AH values for cases
folder_paths_cases = {
    "Case 1": [
        "NLCD/outputs_no_AH",
        "NLCD/outputs_AH_option2"
    ],
    "Case 2": [
        "NLCD/outputs_no_AH_rtc",
        "NLCD/outputs_AH_option2_rtc"
    ],
    "Case 3": [
        "NLCD/outputs_no_AH_ch100",
        "NLCD/outputs_AH_option2_ch100"
    ],
    "Case 4": [
        "NLCD/outputs_no_AH_rtc_ch100",
        "NLCD/outputs_AH_option2_rtc_ch100"
    ]    
}

AH_values_cases = {
    "Case 1": [0, 10],
    "Case 2": [0, 10],
    "Case 3": [0, 10],
    "Case 4": [0, 10]    
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
  
def compute_TC_URB_DIAG(ds, TA_URB=None, TG_URB=None, TB_URB=None, ALPHAC_URB2D=None, ALPHAG_URB2D=None, ALPHAB_URB2D=None, AH=None):
    '''Compute TC_URB_DIAG given a dataset and optionally replace input variables.'''
    
    # Use the provided replacements if available
    TA_URB = ds['TA_URB'] if TA_URB is None else TA_URB
    TG_URB = ds['TG_URB'] if TG_URB is None else TG_URB
    TB_URB = ds['TB_URB'] if TB_URB is None else TB_URB
    ALPHAC_URB2D = ds['ALPHAC_URB2D'] if ALPHAC_URB2D is None else ALPHAC_URB2D
    ALPHAG_URB2D = ds['ALPHAG_URB2D'] if ALPHAG_URB2D is None else ALPHAG_URB2D
    ALPHAB_URB2D = ds['ALPHAB_URB2D'] if ALPHAB_URB2D is None else ALPHAB_URB2D
    AH = 0 if AH is None else AH
    
    # Now use the above variables to compute TC_URB_DIAG
    TC_URB_DIAG = (ds['RW_TBL']*TA_URB*ALPHAC_URB2D + \
                   ds['RW_TBL']*TG_URB*ALPHAG_URB2D + \
                   ds['W_TBL']*TB_URB*ALPHAB_URB2D + AH*converstion_factor) / \
                  (ds['RW_TBL']*ALPHAC_URB2D + 
                   ds['RW_TBL']*ALPHAG_URB2D + 
                   ds['W_TBL']*ALPHAB_URB2D)
                   
    return TC_URB_DIAG

def compute_TC_URB_DIAG_rtc(ds, TA_URB=None, TG_URB=None, TR_URB=None, TB_URB=None, ALPHAC_URB2D=None, ALPHAG_URB2D=None, ALPHAR_URB2D=None, ALPHAB_URB2D=None, AH=None):
    '''Compute TC_URB_DIAG given a dataset and optionally replace input variables.'''
    
    # Use the provided replacements if available
    TA_URB = ds['TA_URB'] if TA_URB is None else TA_URB
    TG_URB = ds['TG_URB'] if TG_URB is None else TG_URB
    TR_URB = ds['TR_URB'] if TR_URB is None else TR_URB    
    TB_URB = ds['TB_URB'] if TB_URB is None else TB_URB
    ALPHAC_URB2D = ds['ALPHAC_URB2D'] if ALPHAC_URB2D is None else ALPHAC_URB2D
    ALPHAG_URB2D = ds['ALPHAG_URB2D'] if ALPHAG_URB2D is None else ALPHAG_URB2D
    ALPHAR_URB2D = ds['ALPHAR_URB2D'] if ALPHAR_URB2D is None else ALPHAR_URB2D
    ALPHAB_URB2D = ds['ALPHAB_URB2D'] if ALPHAB_URB2D is None else ALPHAB_URB2D
    AH = 0 if AH is None else AH
    
    # Now use the above variables to compute TC_URB_DIAG
    TC_URB_DIAG = (TA_URB*ALPHAC_URB2D + \
                   ds['RW_TBL']*TG_URB*ALPHAG_URB2D + \
                   ds['R_TBL']*TR_URB*ALPHAR_URB2D + \
                   ds['W_TBL']*TB_URB*ALPHAB_URB2D + AH*converstion_factor) / \
                  (ALPHAC_URB2D + 
                   ds['RW_TBL']*ALPHAG_URB2D +
                   ds['R_TBL']*ALPHAR_URB2D +                    
                   ds['W_TBL']*ALPHAB_URB2D)

    TC_URB_DIAG=TC_URB_DIAG.where(np.isfinite(TC_URB_DIAG), np.nan)
          
    return TC_URB_DIAG
#%% 
decomposed_values = {}
decomposed_std_dev = {}

utypes = ["all", 1, 2, 3]  # List of urban types to analyze

for case, ds_list in datasets.items():
    baseline_ds = ds_list[0]  # Baseline dataset
    baseline_AH = AH_values_cases[case][0]

    for utype in utypes:
        # Filtering based on urban type if utype is not 'all'
        baseline_filtered = baseline_ds if utype == "all" else baseline_ds.where(baseline_ds['UTYPE_URB'] == utype, drop=True)
        
        if case in ["Case 1", "Case 3"]:
            baseline_tc_urb_diag_avg = compute_TC_URB_DIAG(baseline_filtered, AH=baseline_AH).mean(dim='Time')
        else:
            baseline_tc_urb_diag_avg = compute_TC_URB_DIAG_rtc(baseline_filtered, AH=baseline_AH).mean(dim='Time')

        for i, ds in enumerate(ds_list[1:], 1):  # Starting from the second dataset in the list
            ds_filtered = ds if utype == "all" else ds.where(ds['UTYPE_URB'] == utype, drop=True)
            AH = AH_values_cases[case][i]

            results = []
            std_devs = []
            
            
            if case in ["Case 1", "Case 3"]:

                for compute_func, params in [
                    (compute_TC_URB_DIAG, {"AH": AH}),
                    (compute_TC_URB_DIAG, {"ALPHAB_URB2D": ds_filtered['ALPHAB_URB2D'], "ALPHAG_URB2D": ds_filtered['ALPHAG_URB2D']}),
                    (compute_TC_URB_DIAG, {"TB_URB": ds_filtered['TB_URB'], "TG_URB": ds_filtered['TG_URB']}),
                    (compute_TC_URB_DIAG, {"ALPHAC_URB2D": ds_filtered['ALPHAC_URB2D']}),
                    (compute_TC_URB_DIAG, {"TA_URB": ds_filtered['TA_URB']}),
                ]:
                    diff_time_avg = (compute_func(baseline_filtered, **params).mean(dim='Time') - baseline_tc_urb_diag_avg) / AH
                    spatial_avg_diff = diff_time_avg.mean(['south_north', 'west_east']).values
                    spatial_std_dev = diff_time_avg.std(['south_north', 'west_east']).values

                    results.append(spatial_avg_diff)
                    std_devs.append(spatial_std_dev)
            else:
                
                for compute_func, params in [
                    (compute_TC_URB_DIAG_rtc, {"AH": AH}),
                    (compute_TC_URB_DIAG_rtc, {"ALPHAB_URB2D": ds_filtered['ALPHAB_URB2D'], "ALPHAG_URB2D": ds_filtered['ALPHAG_URB2D'], "ALPHAR_URB2D": ds_filtered['ALPHAR_URB2D']}),
                    (compute_TC_URB_DIAG_rtc, {"TB_URB": ds_filtered['TB_URB'], "TG_URB": ds_filtered['TG_URB'], "TR_URB": ds_filtered['TR_URB']}),
                    (compute_TC_URB_DIAG_rtc, {"ALPHAC_URB2D": ds_filtered['ALPHAC_URB2D']}),
                    (compute_TC_URB_DIAG_rtc, {"TA_URB": ds_filtered['TA_URB']}),
                ]:
                    diff_time_avg = (compute_func(baseline_filtered, **params).mean(dim='Time') - baseline_tc_urb_diag_avg) / AH
                    spatial_avg_diff = diff_time_avg.mean(['south_north', 'west_east']).values
                    spatial_std_dev = diff_time_avg.std(['south_north', 'west_east']).values
    
                    results.append(spatial_avg_diff)
                    std_devs.append(spatial_std_dev)

            overall_diff_time_avg = ds_filtered['TC_URB_DIAG'].mean(dim='Time') 
            overall_spatial_avg_diff = ((overall_diff_time_avg - baseline_tc_urb_diag_avg) / AH).mean(['south_north', 'west_east']).values
            overall_spatial_std_dev = ((overall_diff_time_avg - baseline_tc_urb_diag_avg) / AH).std(['south_north', 'west_east']).values

            cumulative_sum = sum(results)
            cumulative_sum_std_dev = np.sqrt(sum([std_dev**2 for std_dev in std_devs]))

            decomposed_values[(case, utype)] = [overall_spatial_avg_diff, cumulative_sum] + results
            decomposed_std_dev[(case, utype)] = [overall_spatial_std_dev, cumulative_sum_std_dev] + std_devs
            

#for key, value in decomposed_values.items():
#    print(f"{key}: {value}")
#%%


# Creating a 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()  # Flattening the 2D axes array into 1D for easier iteration

labels1 = ["Direct", "Sum", "Baseline", "r$_W$, r$_G$", "T$_W$, $T_G$", "r$_C$", "T$_A$"]
labels2 = ["Direct", "Sum", "Baseline", "r$_R$, r$_W$, r$_G$", "T$_R$, T$_W$, $T_G$", "r$_C$", "T$_A$"]

colors = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'black']
utype = "all"
subplot_labels = ['(a)', '(b)', '(c)', '(d)']

# Set font sizes
axis_label_font_size = 14  # Font size for x and y axis labels
title_font_size = 14       # Font size for the subplot titles
tick_label_font_size = 14  # Font size for tick labels
legend_font_size = 14      # Font size for legend

# Assuming you have exactly 4 cases to plot
for i, (case, ax) in enumerate(zip(datasets.keys(), axes)):
    # Fetch the values and standard deviations for 'all' utype for this case
    values = decomposed_values[(case, utype)]
    std_devs = decomposed_std_dev[(case, utype)]
    
    ax.bar(labels1 if i in [0, 2] else labels2, values, color=colors, yerr=std_devs, capsize=5)
    full_title = f"{subplot_labels[i]} {case}"
    ax.set_title(full_title, fontsize=title_font_size)    
    ax.set_xticklabels(labels1 if i in [0, 2] else labels2, rotation=45, ha="right", fontsize=tick_label_font_size)
    ax.set_ylim([-0.02, 0.12])  # Adjust as needed based on your data
    ax.set_ylabel('dT$_{C}$/dQ$_{AH}$', fontsize=axis_label_font_size)

    # Adjusting the font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

plt.tight_layout()
plt.show()

fig.savefig("figures/NLCD_decomposition_all_4cases.png", dpi=300)

