#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:42:56 2023

@author: danl
"""

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
variables_to_read = ["TC_URB", "TB_URB", "TG_URB", "TR_URB", "TA_URB", "ALPHAC_URB2D", "ALPHAB_URB2D", "ALPHAG_URB2D", "ALPHAR_URB2D", "UTYPE_URB", "LU_INDEX"]


folder_paths = [
    "NLCD/outputs_no_AH_rtc",
    "NLCD/outputs_AH_option2_rtc"
]

AH_values = {
    "NLCD/outputs_no_AH_rtc": 0, 
    "NLCD/outputs_AH_option2_rtc": 10       
}

converstion_factor = 0.001*0.24/1004.5

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


datasets = {}
for folder in folder_paths:
    ds_processed = process_dataset(folder, AH_values[folder])
    datasets[folder] = ds_processed
    
#%% 
  
def compute_TC_URB_DIAG(ds, TA_URB=None, TG_URB=None, TR_URB=None, TB_URB=None, ALPHAC_URB2D=None, ALPHAG_URB2D=None, ALPHAR_URB2D=None, ALPHAB_URB2D=None, AH=None):
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
            (compute_TC_URB_DIAG, {"ALPHAB_URB2D": ds_filtered['ALPHAB_URB2D'], "ALPHAG_URB2D": ds_filtered['ALPHAG_URB2D'], "ALPHAR_URB2D": ds_filtered['ALPHAR_URB2D']}),
            (compute_TC_URB_DIAG, {"TB_URB": ds_filtered['TB_URB'], "TG_URB": ds_filtered['TG_URB'], "TR_URB": ds_filtered['TR_URB']}),
            (compute_TC_URB_DIAG, {"ALPHAC_URB2D": ds_filtered['ALPHAC_URB2D']}),
            (compute_TC_URB_DIAG, {"TA_URB": ds_filtered['TA_URB']}),
        ]:
            diff_time_avg = (compute_func(folder_1_filtered, **params).mean(dim='Time') - folder_1_tc_urb_diag_baseline_time_avg) / 10.0
            spatial_avg_diff = diff_time_avg.mean(['south_north', 'west_east']).values
            spatial_std_dev = diff_time_avg.std(['south_north', 'west_east']).values
            
            results.append(spatial_avg_diff)
            std_devs.append(spatial_std_dev)

        overall_diff_time_avg = ds_filtered['TC_URB_DIAG'].mean(dim='Time') 
        overall_spatial_avg_diff = ((overall_diff_time_avg - folder_1_tc_urb_diag_baseline_time_avg)/ 10.0).mean(['south_north', 'west_east']).values
        overall_spatial_std_dev = ((overall_diff_time_avg - folder_1_tc_urb_diag_baseline_time_avg)/ 10.0).std(['south_north', 'west_east']).values

        cumulative_sum = sum(results)
        cumulative_sum_std_dev = np.sqrt(sum([std_dev**2 for std_dev in std_devs]))


        decomposed_values[(folder, utype)] = [overall_spatial_avg_diff, cumulative_sum] + results
        decomposed_std_dev[(folder, utype)] = [overall_spatial_std_dev, cumulative_sum_std_dev] + std_devs

#%%
method_map = {
    'NLCD/outputs_AH_option2_forcerestore_roof_to_canopy_air': 'Roof to Canopy Air',
    # Add more if needed
}


fig, ax = plt.subplots(figsize=(12, 8))  # Use a single axis

labels = ["Direct", "Sum", "Baseline", "c$_s$", "T$_s$", "c$_a$", "T$_a$"]
colors = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'black']  # Modify this to your desired colors

utype = "all"

# Only get the values and std_devs for the 'all' urban type
values = decomposed_values[folder_paths[1], utype]  # Assuming you want to plot for the second folder path
std_devs = decomposed_std_dev[folder_paths[1], utype]

ax.bar(labels, values, color=colors, yerr=std_devs, capsize=10)
ax.set_title('Case 2')
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylim(-0.02, 0.12)

plt.tight_layout()
plt.show()

fig.savefig("figures/NLCD_decomposition_rtc.png", dpi=300)

#%%

titles = ["Atmosphere", "Road", "Wall"]

# Set up the figure and axes for the 3-panel plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

utypes = [1, 2, 3]
labels = [f"Urban type = {ut}" for ut in utypes]

# Selecting the first time step value for R_TBL_temp
temp1 = datasets[folder_paths[1]]['ALPHAC_URB2D'].mean(dim='Time')/converstion_factor
temp2 = datasets[folder_paths[1]]['ALPHAG_URB2D'].mean(dim='Time')/converstion_factor
temp3 = datasets[folder_paths[1]]['ALPHAB_URB2D'].mean(dim='Time')/converstion_factor

data_list = [temp1, temp2, temp3]

for i, data in enumerate(data_list):
    avg_diffs = [data.where(datasets[folder]['UTYPE_URB'] == ut).mean().values for ut in utypes]
    
    # Plotting
    axes[i].bar(labels, avg_diffs, color=['blue', 'green', 'red'], capsize=10)
    axes[i].set_title(titles[i])
    axes[i].set_ylim([0, 25])

plt.tight_layout()
plt.show()

fig.savefig("figures/Conductances_rtc.png", dpi=300)


#%%

# titles = ["Atmosphere", "Road and Roof", "Wall", "Sum of All"]

# # Set up the figure and axes for the 4-panel plot
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

# utypes = [1, 2, 3]
# labels = [f"Urban type = {ut}" for ut in utypes]

# # Your existing data calculations
# temp1 = datasets[folder_paths[1]]['ALPHAC_URB2D']/converstion_factor
# temp2 = datasets[folder_paths[1]]['ALPHAG_URB2D']/converstion_factor
# temp3 = datasets[folder_paths[1]]['W_TBL']*datasets[folder_paths[1]]['ALPHAB_URB2D']/converstion_factor

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
#     axes[row, col].set_ylim([0, 16])

# # Summing data from the first three panels and plotting in the fourth panel
# summed_values = [sum(x) for x in zip(*summed_data)]
# axes[1, 1].bar(labels, summed_values, color=['blue', 'green', 'red'], capsize=10)
# axes[1, 1].set_title(titles[3])
# axes[1, 1].set_ylim([0,40])  # Adjusting the y-limit for the summed values

# plt.tight_layout()
# plt.show()

# fig.savefig("Morphologymultipliedbyconductances_rooftocanopyair.png", dpi=300)



