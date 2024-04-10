#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:47:08 2023

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
variables_to_read = ["TC_URB", "TB_URB", "TG_URB", "TA_URB", "ALPHAC_URB2D", "ALPHAB_URB2D", 
                     "ALPHAG_URB2D", "UTYPE_URB", "LU_INDEX", "HFX"]


folder_paths = [
    "T2_test/T2_test_no_AH",
    "T2_test/T2_test_AH_option4", 
    "T2_test/T2_test_AH_option3",    
    "T2_test/T2_test_AH_option2"
]

AH_value = 100

AH_values = {
    "T2_test/T2_test_no_AH": 0,
    "T2_test/T2_test_AH_option4": 0,   
    "T2_test/T2_test_AH_option3": 0,       
    "T2_test/T2_test_AH_option2": AH_value
}

method_map = {
    'T2_test/T2_test_AH_option4': 'method 1',
    'T2_test/T2_test_AH_option3': 'method 2',
    'T2_test/T2_test_AH_option2': 'method 3',
    # Add more if needed
}

converstion_factor = 0.001*0.24/1004.5

# Set font sizes
axis_label_font_size = 16  # Font size for x and y axis labels
title_font_size = 14       # Font size for the subplot titles
tick_label_font_size = 14  # Font size for tick labels
legend_font_size = 14      # Font size for legend

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


datasets = {}
for folder in folder_paths:
    ds_processed = process_dataset(folder, AH_values[folder])
    datasets[folder] = ds_processed
    
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

#%% 

folder_1_ds = datasets[folder_paths[0]]

decomposed_values = {}
decomposed_std_dev = {}

utypes = ["all", 1, 2, 3]

for utype in utypes:
    if utype == "all":
        folder_1_filtered = folder_1_ds
    else:
        folder_1_filtered = folder_1_ds.where(folder_1_ds['UTYPE_URB'] == utype)
        
    folder_1_tc_urb_diag_baseline_time_avg = compute_TC_URB_DIAG(folder_1_filtered, AH=AH_values[folder_paths[0]]).mean(dim='Time')

    for folder, ds in datasets.items():
        if folder == folder_paths[0]:
            continue  # We don't decompose the base folder

        if utype == "all":
            ds_filtered = ds
        else:
            ds_filtered = ds.where(ds['UTYPE_URB'] == utype)
            
        results = []
        std_devs = []

        for compute_func, params in [
            (compute_TC_URB_DIAG, {"AH": AH_values[folder]}),
            (compute_TC_URB_DIAG, {"ALPHAB_URB2D": ds_filtered['ALPHAB_URB2D'], "ALPHAG_URB2D": ds_filtered['ALPHAG_URB2D']}),
            (compute_TC_URB_DIAG, {"TB_URB": ds_filtered['TB_URB'], "TG_URB": ds_filtered['TG_URB']}),
            (compute_TC_URB_DIAG, {"ALPHAC_URB2D": ds_filtered['ALPHAC_URB2D']}),
            (compute_TC_URB_DIAG, {"TA_URB": ds_filtered['TA_URB']}),
        ]:
            diff_time_avg = (compute_func(folder_1_filtered, **params).mean(dim='Time') - folder_1_tc_urb_diag_baseline_time_avg) / AH_value
            spatial_avg_diff = diff_time_avg.mean(['south_north', 'west_east']).values
            spatial_std_dev = diff_time_avg.std(['south_north', 'west_east']).values
            
            results.append(spatial_avg_diff)
            std_devs.append(spatial_std_dev)

        overall_diff_time_avg = ds_filtered['TC_URB_DIAG'].mean(dim='Time') 
        overall_spatial_avg_diff = ((overall_diff_time_avg - folder_1_tc_urb_diag_baseline_time_avg)/ AH_value).mean(['south_north', 'west_east']).values
        overall_spatial_std_dev = ((overall_diff_time_avg - folder_1_tc_urb_diag_baseline_time_avg)/ AH_value).std(['south_north', 'west_east']).values

        cumulative_sum = sum(results)
        cumulative_sum_std_dev = np.sqrt(sum([std_dev**2 for std_dev in std_devs]))


        decomposed_values[(folder, utype)] = [overall_spatial_avg_diff, cumulative_sum] + results
        decomposed_std_dev[(folder, utype)] = [overall_spatial_std_dev, cumulative_sum_std_dev] + std_devs

#%% 

# fig, axes = plt.subplots(4, 3, figsize=(24, 24))  # Setting up a 4x3 grid
# axes = axes.ravel()  # Flattening the axes to easily index them

# labels = ["Direct", "Sum", "Baseline", "r$_B$, r$_G$", "T$_B$, $T_G$", "r$_C$", "T$_A$"]
# colors = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'black']  # Modify this to your desired colors

# titles = ["(a) method 1", "(b) method 2", "(c) method 3"]


# for idx, ((folder, utype), values) in enumerate(decomposed_values.items()):
#     ax = axes[idx]
    
#     # Fetch the standard deviations for the current folder and utype
#     std_devs = decomposed_std_dev[(folder, utype)]
    
#     ax.bar(labels, values, color=colors, yerr=std_devs, capsize=10)  # Added yerr for error bars and capsize for cap size on error bars
#     #ax.set_ylabel('Mean Value')
    
#     method_name = method_map.get(folder, folder)  # Use folder name if method name not found in the dictionary
#     ax.set_title(f'{method_name} with Urban type = {utype}')
#     #ax.set_title(titles[idx], fontsize=title_font_size)
#     ax.set_xticklabels(labels, rotation=45, ha="right")  # Added ha="right" for better label alignment
#     ax.set_ylim(-0.02, 0.2)  # Set the same y-limits for each subplot
#     ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
#     ax.set_ylabel('$dT_{C}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
    
# plt.tight_layout()
# plt.show()

# fig.savefig("figures/NLCD_decomposition_100.png", dpi=300)


#%%

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
axes = axes.ravel()  # Flattening the axes to easily index them

labels = ["Direct", "Sum", "Baseline", "r$_W$, r$_G$", "T$_W$, $T_G$", "r$_C$", "T$_A$"]
colors = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'black']  # Modify this to your desired colors

utype = "all"

titles = ["(g) method 1", "(h) method 2", "(i) method 3"]

plot_idx = 0

for idx, ((folder, utype), values) in enumerate(decomposed_values.items()):
    
    if utype != 'all' or plot_idx >= len(axes):
    # Skip this iteration if utype is not 'all' or if we have no more subplots
        continue    
    
    ax = axes[plot_idx]
    
    # Fetch the standard deviations for the current folder and utype
    std_devs = decomposed_std_dev[(folder, utype)]
    
    ax.bar(labels, values, color=colors, yerr=std_devs, capsize=10)  # Added yerr for error bars and capsize for cap size on error bars
    print(values)
    ax.set_title(titles[plot_idx], fontsize=title_font_size)
    ax.set_xticklabels(labels, rotation=45, ha="right")  # Added ha="right" for better label alignment
    ax.set_ylim(-0.02, 0.12)  # Set the same y-limits for each subplot
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dT_{C}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
    
    plot_idx += 1
    #print(plot_idx)
    
plt.tight_layout()
plt.show()

fig.savefig("figures/NLCD_decomposition_all_AH100.png", dpi=300)

#%%

# titles = ["Roof", "Road", "Wall"]

# # Set up the figure and axes for the 3-panel plot
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# utypes = [1, 2, 3]
# labels = [f"Urban type = {ut}" for ut in utypes]

# # Selecting the first time step value for R_TBL_temp
# temp1 = datasets[folder_paths[0]]['R_TBL'].isel(Time=0)
# temp2 = datasets[folder_paths[0]]['RW_TBL'].isel(Time=0)
# temp3 = datasets[folder_paths[0]]['W_TBL'].isel(Time=0)

# data_list = [temp1, temp2, temp3]

# for i, data in enumerate(data_list):
#     avg_diffs = [data.where(datasets[folder]['UTYPE_URB'] == ut).mean().values for ut in utypes]
    
#     # Plotting
#     axes[i].bar(labels, avg_diffs, color=['blue', 'green', 'red'], capsize=10)
#     axes[i].set_title(titles[i])
#     axes[i].set_ylim([0, 1.0])

# plt.tight_layout()
# plt.show()

# fig.savefig("figures/Building_morphology.png", dpi=300)

#%%

titles = ["Atmosphere", "Road", "Wall"]

# Set up the figure and axes for the 3-panel plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

utypes = [1, 2, 3]
labels = [f"Urban type = {ut}" for ut in utypes]

# Selecting the first time step value for R_TBL_temp
temp1 = datasets[folder_paths[0]]['ALPHAC_URB2D'].mean(dim='Time')/converstion_factor
temp2 = datasets[folder_paths[0]]['ALPHAG_URB2D'].mean(dim='Time')/converstion_factor
temp3 = datasets[folder_paths[0]]['ALPHAB_URB2D'].mean(dim='Time')/converstion_factor

data_list = [temp1, temp2, temp3]

for i, data in enumerate(data_list):
    avg_diffs = [data.where(datasets[folder]['UTYPE_URB'] == ut).mean().values for ut in utypes]
    
    # Plotting
    axes[i].bar(labels, avg_diffs, color=['blue', 'green', 'red'], capsize=10)
    axes[i].set_title(titles[i])
    axes[i].set_ylim([0, 25])

plt.tight_layout()
plt.show()

fig.savefig("figures/Conductances.png", dpi=300)


#%%

# titles = ["Atmosphere", "Road", "Wall", "Sum of All"]

# # Set up the figure and axes for the 3-panel plot
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

# utypes = [1, 2, 3]
# labels = [f"Urban type = {ut}" for ut in utypes]

# # Your existing data calculations
# temp1 = datasets[folder_paths[0]]['RW_TBL']*datasets[folder_paths[0]]['ALPHAC_URB2D']/converstion_factor
# temp2 = datasets[folder_paths[0]]['RW_TBL']*datasets[folder_paths[0]]['ALPHAG_URB2D']/converstion_factor
# temp3 = datasets[folder_paths[0]]['W_TBL']*datasets[folder_paths[0]]['ALPHAB_URB2D']/converstion_factor

# data_list = [temp1.mean(dim='Time'), temp2.mean(dim='Time'), temp3.mean(dim='Time')]

# summed_data = []

# subplot_indices = [(0, 0), (0, 1), (1, 0)]

# for i, (row, col) in enumerate(subplot_indices):
#     data = data_list[i]
#     avg_diffs = [data.where(datasets[folder]['UTYPE_URB'] == ut).mean().values for ut in utypes]
#     summed_data.append(avg_diffs)
    
#     # Plotting
#     axes[row, col].bar(labels, avg_diffs, color=['blue', 'green', 'red'], capsize=10)
#     axes[row, col].set_title(titles[i])
#     axes[row, col].set_ylim([0, 10])

# # Summing data from the first three panels and plotting in the fourth panel
# summed_values = [sum(x) for x in zip(*summed_data)]
# axes[1, 1].bar(labels, summed_values, color=['blue', 'green', 'red'], capsize=10)
# axes[1, 1].set_title(titles[3])
# axes[1, 1].set_ylim([0, 20])  # Adjusting the y-limit for the summed values

# plt.tight_layout()
# plt.show()

# fig.savefig("figures/Morphologymultipliedbyconductances.png", dpi=300)

