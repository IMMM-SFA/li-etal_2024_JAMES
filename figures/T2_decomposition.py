#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:45:23 2024

@author: danl
"""


import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Time setup (global to all datasets)
start_datetime_data = pd.Timestamp('2022-07-19 00:00:00')
time_interval = pd.Timedelta(hours=1)
start_time_desired = pd.Timestamp('2022-07-20 00:00:00')
end_time_desired = pd.Timestamp('2022-07-23 00:00:00')
start_index = int((start_time_desired - start_datetime_data) / time_interval)
end_index = int((end_time_desired - start_datetime_data) / time_interval)

variables_to_read = ["TC_URB", "TB_URB", "TG_URB", "TA_URB", "TR_URB",  "TSK", "TS_URB", "T2",
                     "ALPHAC_URB2D", "ALPHAB_URB2D", "ALPHAG_URB2D", "ALPHAR_URB2D",
                     "UTYPE_URB", "LU_INDEX", "FRC_URB2D",
                     "HFX","SH_URB","CHS_URB2D", "CHS2_URB2D"]


folder_paths = [
    "T2_test/T2_test_no_AH",
    "T2_test/T2_test_AH_option5", 
    "T2_test/T2_test_AH_option3",    
    "T2_test/T2_test_AH_option2"
]

AH_value = 100

AH_values = {
    "T2_test/T2_test_no_AH": 0,
    "T2_test/T2_test_AH_option5": 0,   
    "T2_test/T2_test_AH_option3": 0,       
    "T2_test/T2_test_AH_option2": AH_value
}

AH_T2_values = {
    "T2_test/T2_test_no_AH": 0,
    "T2_test/T2_test_AH_option5": AH_value,   
    "T2_test/T2_test_AH_option3": 0,       
    "T2_test/T2_test_AH_option2": 0
}

method_map = {
    'T2_test/T2_test_AH_option5': 'revised method 1',
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

def process_dataset(folder_path, AH, AH_T2_value):
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
    
    ds_ana['TS_RUL'] =  (ds_ana['TSK'] -   ds_ana['TS_URB']*ds_ana['FRC_URB2D'] ) / (1 - ds_ana['FRC_URB2D'])            

    ds_ana['Qu'] = ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TA_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                             ds_ana['RW_TBL']*(ds_ana['TC_URB']-ds_ana['TA_URB'])*ds_ana['ALPHAC_URB2D']/converstion_factor 
                             
    ds_ana['Qu2'] = ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TA_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                   ds_ana['W_TBL']*(ds_ana['TB_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAB_URB2D']/converstion_factor + \
                   ds_ana['RW_TBL']*(ds_ana['TG_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAG_URB2D']/converstion_factor + \
                       + AH
                       
        
    #ds_ana['QT'] = ds_ana['Q_RUL']* (1 - ds_ana['FRC_URB2D'])  + ds_ana['Qu']*ds_ana['FRC_URB2D']
                         
    
    ds_ana['Tu'] = ds_ana['Qu']/ds_ana['CHS_URB2D'] + ds_ana['TA_URB']
    
    ds_ana['TS_URB_DIAG'] = ds_ana['TS_RUL']* (1 - ds_ana['FRC_URB2D']) + ((ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TA_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                   ds_ana['W_TBL']*(ds_ana['TB_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAB_URB2D']/converstion_factor + \
                   ds_ana['RW_TBL']*(ds_ana['TG_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAG_URB2D']/converstion_factor + \
                       + AH)/ds_ana['CHS_URB2D'] + ds_ana['TA_URB']) *ds_ana['FRC_URB2D']

    ds_ana['HFX_corrected'] = ds_ana['HFX']-AH_T2_value*ds_ana['FRC_URB2D']
    
    ds_ana['Q_RUL'] =  (ds_ana['HFX_corrected'] - ds_ana['SH_URB']*ds_ana['FRC_URB2D']) / (1 - ds_ana['FRC_URB2D'])

    
    ds_ana['T2_URB_DIAG'] = ds_ana['TS_RUL']* (1 - ds_ana['FRC_URB2D']) + ((ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TA_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                   ds_ana['W_TBL']*(ds_ana['TB_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAB_URB2D']/converstion_factor + \
                   ds_ana['RW_TBL']*(ds_ana['TG_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAG_URB2D']/converstion_factor + \
                       + AH)/ds_ana['CHS_URB2D'] + ds_ana['TA_URB']) *ds_ana['FRC_URB2D'] -  \
        (ds_ana['Q_RUL']*(1 - ds_ana['FRC_URB2D'])+(ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TA_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                                      ds_ana['W_TBL']*(ds_ana['TB_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAB_URB2D']/converstion_factor + \
                                      ds_ana['RW_TBL']*(ds_ana['TG_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAG_URB2D']/converstion_factor + \
                                          + AH)*ds_ana['FRC_URB2D'])/ds_ana['CHS2_URB2D']

    ds_ana['T2_URB_DIAG2'] = ds_ana['TS_RUL']* (1 - ds_ana['FRC_URB2D']) + ((ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TA_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                   ds_ana['W_TBL']*(ds_ana['TB_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAB_URB2D']/converstion_factor + \
                   ds_ana['RW_TBL']*(ds_ana['TG_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAG_URB2D']/converstion_factor + \
                       + AH)/ds_ana['CHS_URB2D'] + ds_ana['TA_URB']) *ds_ana['FRC_URB2D'] -  \
        ((ds_ana['TS_RUL']-ds_ana['TA_URB'])*ds_ana['CHS_URB2D']*(1 - ds_ana['FRC_URB2D'])+(ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TA_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                                      ds_ana['W_TBL']*(ds_ana['TB_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAB_URB2D']/converstion_factor + \
                                      ds_ana['RW_TBL']*(ds_ana['TG_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAG_URB2D']/converstion_factor + \
                                          + AH)*ds_ana['FRC_URB2D'])/ds_ana['CHS2_URB2D']
    
    # create time average outputs for TC_URB_DIAG and TC_URB (masked)
                            
    
    #mask = ds_ana['UTYPE_URB'] != 0
    ds_ana['TC_URB_MASK'] = ds_ana['TC_URB'].where(ds_ana['UTYPE_URB'] != 0)
    ds_ana['tc_urb_avg_time'] = ds_ana['TC_URB_MASK'].mean(dim='Time')
    ds_ana['tc_urb_diag_avg_time'] = ds_ana['TC_URB_DIAG'].mean(dim='Time') 
    
    ds_ana['Qu_avg_time'] = ds_ana['SH_URB'].mean(dim='Time') 
    ds_ana['Qu_diag_avg_time'] = ds_ana['Qu'].mean(dim='Time') 
    ds_ana['Qu2_diag_avg_time'] = ds_ana['Qu2'].mean(dim='Time') 
    
    ds_ana['TSK_URB_MASK'] = ds_ana['TSK'].where(ds_ana['UTYPE_URB'] != 0)
    ds_ana['ts_urb_avg_time'] = ds_ana['TSK_URB_MASK'].mean(dim='Time')
    ds_ana['ts_urb_diag_avg_time'] = ds_ana['TS_URB_DIAG'].mean(dim='Time') 

    ds_ana['T2_URB_MASK'] = ds_ana['T2'].where(ds_ana['UTYPE_URB'] != 0)
    ds_ana['t2_urb_avg_time'] = ds_ana['T2_URB_MASK'].mean(dim='Time')
    ds_ana['t2_urb_diag_avg_time'] = ds_ana['T2_URB_DIAG'].mean(dim='Time') 
    ds_ana['t2_urb_diag2_avg_time'] = ds_ana['T2_URB_DIAG2'].mean(dim='Time') 

    return ds_ana


datasets = {}
for folder in folder_paths:
    ds_processed = process_dataset(folder, AH_values[folder], AH_T2_values[folder])
    datasets[folder] = ds_processed

ref_ds = datasets[folder_paths[0]]

for i, folder in enumerate(folder_paths[1:]):
    datasets[folder]['tc_sensitivity'] = (datasets[folder]['tc_urb_diag_avg_time'] - ref_ds['tc_urb_diag_avg_time']) / AH_value
    datasets[folder]['ts_sensitivity'] = (datasets[folder]['TSK'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time') - ref_ds['TSK'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time'))/AH_value
    datasets[folder]['t2_sensitivity'] = (datasets[folder]['T2'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time') - ref_ds['T2'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time'))/AH_value
    datasets[folder]['H_sensitivity'] = (datasets[folder]['HFX_corrected'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time') - ref_ds['HFX_corrected'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time'))/AH_value



#%% plot SH_URB and QU_DIAG
titles_4 = ["(a) no $Q_{AH}$","(b) revised method 1", "(c) method 2", "(d) method 3"]

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
axes = axes.flatten()  # Flatten the 2x2 matrix to make indexing easier

# Iterate over the datasets and plot
for i, (folder, ds) in enumerate(datasets.items()):
    ax = axes[i]
    ax.scatter(ds['Qu_avg_time'].values.flatten(),
                ds['Qu2_diag_avg_time'].values.flatten(),
                marker='o', alpha=0.5)
    
    # You can customize the scatter plot further, e.g., with colors, marker sizes, etc.
    
    # # Adding 1:1 line
    ax.plot([50, 250], [50, 250], 'r-', label="1:1 Line")

    # Setting the axis limits
    ax.set_xlim([50, 250])
    ax.set_ylim([50, 250])
    ax.set_title(titles_4[i], fontsize=title_font_size)
    ax.set_xlabel("Q$_U$ (W/m$^2$)", fontsize=axis_label_font_size)
    ax.set_ylabel("Diagnosed Q$_U$ (W/m$^2$)", fontsize=axis_label_font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

# Adjust spacing between plots
plt.tight_layout()
plt.show()


#%% plot TSK and TS_URB_DIAG
titles_4 = ["(a) no $Q_{AH}$","(b) revised method 1", "(c) method 2", "(d) method 3"]

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
axes = axes.flatten()  # Flatten the 2x2 matrix to make indexing easier

# Iterate over the datasets and plot
for i, (folder, ds) in enumerate(datasets.items()):
    ax = axes[i]
    ax.scatter(ds['ts_urb_avg_time'].values.flatten(),
                ds['ts_urb_diag_avg_time'].values.flatten(),
                marker='o', alpha=0.5)
    
    # You can customize the scatter plot further, e.g., with colors, marker sizes, etc.
    
    # Adding 1:1 line
    ax.plot([290, 310], [290, 310], 'r-', label="1:1 Line")

    # Setting the axis limits
    ax.set_xlim([290, 310])
    ax.set_ylim([290, 310])
    
    ax.set_title(titles_4[i], fontsize=title_font_size)
    ax.set_xlabel("T$_S$ (K)", fontsize=axis_label_font_size)
    ax.set_ylabel("Diagnosed T$_S$ (K)", fontsize=axis_label_font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

# Adjust spacing between plots
plt.tight_layout()
plt.show()
fig.savefig("T2_figures/validation_TS.png", dpi=300)


#%% plot T2 and T2_URB_DIAG

titles_4 = ["(a) no $Q_{AH}$","(b) revised method 1", "(c) method 2", "(d) method 3"]

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
axes = axes.flatten()  # Flatten the 2x2 matrix to make indexing easier

# Iterate over the datasets and plot
for i, (folder, ds) in enumerate(datasets.items()):
    ax = axes[i]
    ax.scatter(ds['t2_urb_avg_time'].values.flatten(),
                ds['t2_urb_diag_avg_time'].values.flatten(),
                marker='o', alpha=0.5)
    
    # You can customize the scatter plot further, e.g., with colors, marker sizes, etc.
    
    # Adding 1:1 line
    ax.plot([290, 310], [290, 310], 'r-', label="1:1 Line")

    # Setting the axis limits
    ax.set_xlim([290, 310])
    ax.set_ylim([290, 310])
    
    ax.set_title(titles_4[i], fontsize=title_font_size)
    ax.set_xlabel("T$_2$ (K)", fontsize=axis_label_font_size)
    ax.set_ylabel("Diagnosed T$_2$ (K)", fontsize=axis_label_font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

# Adjust spacing between plots
plt.tight_layout()
plt.show()
fig.savefig("T2_figures/validation_T2.png", dpi=300)


#%%
# Set colormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

colorbar_ticks = np.arange(-0.01, 0.051, 0.01)

def create_colormap():
    # Calculate the proportion of the colorbar that should be blue
    negative_proportion = 0.01 / (0.04 + 0.01)
    positive_proportion = 1 - negative_proportion
    #print(negative_proportion)
    # Define the starting point for the Reds and Blues_r to match the intensity
    blue_start = 1 - negative_proportion  # Start from the lightest blue
    red_start = 0.0  # Start from a lighter red to match the blue intensity
    
    # Sample colors from the Blues_r and Reds colormaps
    blues = plt.cm.Blues_r(np.linspace(blue_start, 1.0, int(256 * negative_proportion)))  # Light to dark blue
    whites = np.array([1.0, 1.0, 1.0, 1.0]).reshape(1,4)  # Pure white color
    reds = plt.cm.Reds(np.linspace(red_start, 1.0, int(256 * positive_proportion)))  # Light to dark red
    
    # Combine them into a single array
    colors = np.vstack((blues, whites, reds))
    
    # Create a new colormap
    cmap = LinearSegmentedColormap.from_list('CustomBluesReds', colors)
    
    return cmap

custom_cmap = create_colormap()



titles = ["(a) revised method 1", "(b) method 2", "(c) method 3","(d) revised method 1", "(e) method 2", "(f) method 3"]


# Create a 2x3 subplot layout
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axes = axes.flatten()  # Flatten the axes array for easier indexing

# Plot the first set of figures (first row)
for i, folder in enumerate(folder_paths[1:]):
    ax = axes[i]
    
    ax.set_aspect('equal', adjustable='box')
    norm = matplotlib.colors.Normalize(vmin=-0.01, vmax=0.04)
    im = datasets[folder]['ts_sensitivity'].plot(ax=ax, cmap=custom_cmap, norm=norm, add_colorbar=False)
    ax.set_title(titles[i], fontsize=title_font_size)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=colorbar_ticks)
    cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in colorbar_ticks], fontsize=legend_font_size)
    
    ax.set_xlim(0, 150)
    ax.set_xticks(range(0, 151, 50))
    ax.set_ylim(0, 150)
    ax.set_yticks(range(0, 151, 50))
    ax.set_xlabel('x', fontsize=axis_label_font_size)
    ax.set_ylabel('y', fontsize=axis_label_font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

utypes = [1, 2, 3]
labels = [f"Urban type {ut}" for ut in utypes]

# Plot the second set of figures (second row)
for i, folder in enumerate(folder_paths[1:]):
    ax = axes[i + 3]  # Adjust index to target the second row
    
    avg_diffs = [datasets[folder]['ts_sensitivity'].where(datasets[folder]['UTYPE_URB'] == ut).mean().values for ut in utypes]
    std_devs = [datasets[folder]['ts_sensitivity'].where(datasets[folder]['UTYPE_URB'] == ut).std().values for ut in utypes]
    
    
    ax.bar(labels, avg_diffs, yerr=std_devs, color=['blue', 'green', 'red'], capsize=10)
    ax.set_title(titles[i + 3], fontsize=title_font_size)
    ax.set_ylim([0, 0.04])  # Adjust as necessary
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dT_{S}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)

plt.tight_layout()
plt.show()
fig.savefig(f"T2_figures/TS_AH100.png", dpi=300)


#%%

colorbar_ticks = np.arange(-0.01, 0.021, 0.01)

def create_colormap_air():
    # Calculate the proportion of the colorbar that should be blue
    negative_proportion = 0.01 / (0.02 + 0.01)
    positive_proportion = 1 - negative_proportion
    #print(negative_proportion)
    # Define the starting point for the Reds and Blues_r to match the intensity
    blue_start = 1 - negative_proportion  # Start from the lightest blue
    red_start = 0.0  # Start from a lighter red to match the blue intensity
    
    # Sample colors from the Blues_r and Reds colormaps
    blues = plt.cm.Blues_r(np.linspace(blue_start, 1.0, int(256 * negative_proportion)))  # Light to dark blue
    whites = np.array([1.0, 1.0, 1.0, 1.0]).reshape(1,4)  # Pure white color
    reds = plt.cm.Reds(np.linspace(red_start, 1.0, int(256 * positive_proportion)))  # Light to dark red
    
    # Combine them into a single array
    colors = np.vstack((blues, whites, reds))
    
    # Create a new colormap
    cmap = LinearSegmentedColormap.from_list('CustomBluesReds', colors)
    
    return cmap

custom_cmap_air = create_colormap_air()

titles = ["(a) revised method 1", "(b) method 2", "(c) method 3","(d) revised method 1", "(e) method 2", "(f) method 3"]

# Create a 2x3 subplot layout
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axes = axes.flatten()  # Flatten the axes array for easier indexing

# Plot the first set of figures (first row)
for i, folder in enumerate(folder_paths[1:]):
    ax = axes[i]
    
    ax.set_aspect('equal', adjustable='box')
    norm = matplotlib.colors.Normalize(vmin=-0.01, vmax=0.02)
    im = datasets[folder]['t2_sensitivity'].plot(ax=ax, cmap=custom_cmap_air, norm=norm, add_colorbar=False)
    ax.set_title(titles[i], fontsize=title_font_size)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=colorbar_ticks)
    cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in colorbar_ticks], fontsize=legend_font_size)
    
    ax.set_xlim(0, 150)
    ax.set_xticks(range(0, 151, 50))
    ax.set_ylim(0, 150)
    ax.set_yticks(range(0, 151, 50))
    ax.set_xlabel('x', fontsize=axis_label_font_size)
    ax.set_ylabel('y', fontsize=axis_label_font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

utypes = [1, 2, 3]
labels = [f"Urban type {ut}" for ut in utypes]

# Plot the second set of figures (second row)
for i, folder in enumerate(folder_paths[1:]):
    ax = axes[i + 3]  # Adjust index to target the second row
    
    avg_diffs = [datasets[folder]['t2_sensitivity'].where(datasets[folder]['UTYPE_URB'] == ut).mean().values for ut in utypes]
    std_devs = [datasets[folder]['t2_sensitivity'].where(datasets[folder]['UTYPE_URB'] == ut).std().values for ut in utypes]
    
    
    ax.bar(labels, avg_diffs, yerr=std_devs, color=['blue', 'green', 'red'], capsize=10)
    ax.set_title(titles[i + 3], fontsize=title_font_size)
    ax.set_ylim([0, 0.02])  # Adjust as necessary
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dT_{2}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)

plt.tight_layout()
plt.show()

fig.savefig(f"T2_figures/T2_AH100.png", dpi=300)

#%% The first one has TC while the second one does not have TC
  
def compute_TS_URB_DIAG(ds, TS_RUL=None, TR_URB=None, TB_URB=None, TC_URB=None, TG_URB=None, TA_URB=None, ALPHAR_URB2D=None, ALPHAB_URB2D=None, ALPHAG_URB2D=None, AH=None, CHS_URB2D=None):
    TS_RUL = ds['TS_RUL'] if TS_RUL is None else TS_RUL
    TR_URB = ds['TR_URB'] if TR_URB is None else TR_URB
    TB_URB = ds['TB_URB'] if TB_URB is None else TB_URB
    TC_URB = ds['TC_URB'] if TC_URB is None else TC_URB
    TG_URB = ds['TG_URB'] if TG_URB is None else TG_URB
    TA_URB = ds['TA_URB'] if TA_URB is None else TA_URB
    ALPHAR_URB2D = ds['ALPHAR_URB2D'] if ALPHAR_URB2D is None else ALPHAR_URB2D
    ALPHAB_URB2D = ds['ALPHAB_URB2D'] if ALPHAB_URB2D is None else ALPHAB_URB2D
    ALPHAG_URB2D = ds['ALPHAG_URB2D'] if ALPHAG_URB2D is None else ALPHAG_URB2D
    CHS_URB2D = ds['CHS_URB2D'] if CHS_URB2D is None else CHS_URB2D
    AH = 0 if AH is None else AH

    TS_URB_DIAG = TS_RUL * (1 - ds['FRC_URB2D']) + \
        ((ds['R_TBL'] * (TR_URB - TA_URB) * ALPHAR_URB2D / converstion_factor + \
          ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + \
          ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + \
          AH) / CHS_URB2D + TA_URB) * ds['FRC_URB2D']

    return TS_URB_DIAG

def compute_TS_URB_DIAG_TC_NOT_INCLUDED(ds, TS_RUL=None, TR_URB=None, TB_URB=None, TG_URB=None, TA_URB=None, ALPHAR_URB2D=None, ALPHAB_URB2D=None, ALPHAG_URB2D=None, ALPHAC_URB2D=None, AH=None, CHS_URB2D=None):
    TS_RUL = ds['TS_RUL'] if TS_RUL is None else TS_RUL
    TR_URB = ds['TR_URB'] if TR_URB is None else TR_URB
    TB_URB = ds['TB_URB'] if TB_URB is None else TB_URB
    TG_URB = ds['TG_URB'] if TG_URB is None else TG_URB
    TA_URB = ds['TA_URB'] if TA_URB is None else TA_URB
    ALPHAR_URB2D = ds['ALPHAR_URB2D'] if ALPHAR_URB2D is None else ALPHAR_URB2D
    ALPHAB_URB2D = ds['ALPHAB_URB2D'] if ALPHAB_URB2D is None else ALPHAB_URB2D
    ALPHAG_URB2D = ds['ALPHAG_URB2D'] if ALPHAG_URB2D is None else ALPHAG_URB2D
    ALPHAC_URB2D = ds['ALPHAC_URB2D'] if ALPHAC_URB2D is None else ALPHAC_URB2D
    CHS_URB2D = ds['CHS_URB2D'] if CHS_URB2D is None else CHS_URB2D
    AH = 0 if AH is None else AH
    
    TC_URB = (ds['RW_TBL']*TA_URB*ALPHAC_URB2D + \
                   ds['RW_TBL']*TG_URB*ALPHAG_URB2D + \
                   ds['W_TBL']*TB_URB*ALPHAB_URB2D + AH*converstion_factor) / \
                  (ds['RW_TBL']*ALPHAC_URB2D + 
                   ds['RW_TBL']*ALPHAG_URB2D + 
                   ds['W_TBL']*ALPHAB_URB2D)

    TS_URB_DIAG = TS_RUL * (1 - ds['FRC_URB2D']) + \
        ((ds['R_TBL'] * (TR_URB - TA_URB) * ALPHAR_URB2D / converstion_factor + \
          ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + \
          ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + \
          AH) / CHS_URB2D + TA_URB) * ds['FRC_URB2D']

    return TS_URB_DIAG

#%%
decomposed_values = {}
decomposed_std_dev = {}
utypes = ["all", 1, 2, 3]

for utype in utypes:
    if utype == "all":
        folder_1_filtered = datasets[folder_paths[0]]
    else:
        folder_1_filtered = datasets[folder_paths[0]].where(datasets[folder_paths[0]]['UTYPE_URB'] == utype)

    folder_1_ts_urb_diag_baseline_time_avg = folder_1_filtered['TS_URB_DIAG'].mean(dim='Time') 

    for folder, ds in datasets.items():
        if folder == folder_paths[0]:
            continue

        if utype == "all":
            ds_filtered = ds
        else:
            ds_filtered = ds.where(ds['UTYPE_URB'] == utype)

        results = []
        std_devs = []

        for compute_func, params in [
            (compute_TS_URB_DIAG_TC_NOT_INCLUDED, {"AH": AH_values[folder]}),
            (compute_TS_URB_DIAG_TC_NOT_INCLUDED, {"ALPHAR_URB2D": ds_filtered['ALPHAR_URB2D'], "ALPHAB_URB2D": ds_filtered['ALPHAB_URB2D'], "ALPHAG_URB2D": ds_filtered['ALPHAG_URB2D'], "ALPHAC_URB2D": ds_filtered['ALPHAC_URB2D'],  "CHS_URB2D": ds_filtered['CHS_URB2D']}),
            (compute_TS_URB_DIAG_TC_NOT_INCLUDED, {"TS_RUL": ds_filtered['TS_RUL'], "TR_URB": ds_filtered['TR_URB'], "TB_URB": ds_filtered['TB_URB'], "TG_URB": ds_filtered['TG_URB']}),
            (compute_TS_URB_DIAG_TC_NOT_INCLUDED, {"TA_URB": ds_filtered['TA_URB']})
        ]:
            diff_time_avg = (compute_func(folder_1_filtered, **params).mean(dim='Time') - folder_1_ts_urb_diag_baseline_time_avg) / AH_value
            spatial_avg_diff = diff_time_avg.mean(['south_north', 'west_east']).values
            spatial_std_dev = diff_time_avg.std(['south_north', 'west_east']).values

            results.append(spatial_avg_diff)
            std_devs.append(spatial_std_dev)

        overall_diff_time_avg = ds_filtered['TS_URB_DIAG'].mean(dim='Time') 
        overall_spatial_avg_diff = ((overall_diff_time_avg - folder_1_ts_urb_diag_baseline_time_avg)/ AH_value).mean(['south_north', 'west_east']).values
        overall_spatial_std_dev = ((overall_diff_time_avg - folder_1_ts_urb_diag_baseline_time_avg)/ AH_value).std(['south_north', 'west_east']).values

        cumulative_sum = sum(results)
        cumulative_sum_std_dev = np.sqrt(sum([std_dev**2 for std_dev in std_devs]))

        decomposed_values[(folder, utype)] = [overall_spatial_avg_diff, cumulative_sum] + results
        decomposed_std_dev[(folder, utype)] = [overall_spatial_std_dev, cumulative_sum_std_dev] + std_devs

        #print(decomposed_values)
#%%

fig, axes = plt.subplots(4, 3, figsize=(24, 24))  # Setting up a 4x3 grid
axes = axes.ravel()  # Flattening the axes to easily index them

labels = ["Direct", "Sum", "Baseline", "r", "T","T$_A$"]
colors = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'black']  # Modify this to your desired colors


method_ylims = {
    'revised method 1': (-0.01, 0.04),
    'method 2': (-0.01, 0.04),
    'method 3': (-0.01, 0.04),
    # Add more methods and their ylims as needed
}

subplot_labels = [f'({chr(97 + i)})' for i in range(12)]  # Generates labels (a), (b), ..., (l)

for idx, ((folder, utype), values) in enumerate(decomposed_values.items()):
    ax = axes[idx]
    
    # Fetch the standard deviations for the current folder and utype
    std_devs = decomposed_std_dev[(folder, utype)]
    
    ax.bar(labels, values, color=colors, yerr=std_devs, capsize=10)  # Added yerr for error bars and capsize for cap size on error bars
    #ax.set_ylabel('Mean Value')
    
    method_name = method_map.get(folder, folder)  # Use folder name if method name not found in the dictionary
    ax.set_title(f'{subplot_labels[idx]} {method_name} with urban type = {utype}', fontsize=title_font_size)
    ax.set_xticklabels(labels, rotation=45, ha="right")  # Added ha="right" for better label alignment
    ylim_values = method_ylims.get(method_name, (-0.01, 0.04))  # Default ylim if method_name not in dictionary
    ax.set_ylim(*ylim_values)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dT_{S}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
    
plt.tight_layout()
plt.show()
fig.savefig("T2_figures/TS_decomposition.png", dpi=300)

#%%
def compute_T2_URB_DIAG(ds, TS_RUL=None, TR_URB=None, TB_URB=None, TC_URB=None, TG_URB=None, TA_URB=None, ALPHAR_URB2D=None, ALPHAB_URB2D=None, ALPHAG_URB2D=None, AH=None, CHS_URB2D=None, CHS2_URB2D=None, Q_RUL=None):
    # Use the provided replacements if available or default to dataset values
    TS_RUL = ds['TS_RUL'] if TS_RUL is None else TS_RUL
    TR_URB = ds['TR_URB'] if TR_URB is None else TR_URB
    TB_URB = ds['TB_URB'] if TB_URB is None else TB_URB
    TC_URB = ds['TC_URB'] if TC_URB is None else TC_URB
    TG_URB = ds['TG_URB'] if TG_URB is None else TG_URB
    TA_URB = ds['TA_URB'] if TA_URB is None else TA_URB
    ALPHAR_URB2D = ds['ALPHAR_URB2D'] if ALPHAR_URB2D is None else ALPHAR_URB2D
    ALPHAB_URB2D = ds['ALPHAB_URB2D'] if ALPHAB_URB2D is None else ALPHAB_URB2D
    ALPHAG_URB2D = ds['ALPHAG_URB2D'] if ALPHAG_URB2D is None else ALPHAG_URB2D
    CHS_URB2D = ds['CHS_URB2D'] if CHS_URB2D is None else CHS_URB2D
    CHS2_URB2D = ds['CHS2_URB2D'] if CHS2_URB2D is None else CHS2_URB2D
    Q_RUL = ds['Q_RUL'] if Q_RUL is None else Q_RUL
    AH = 0 if AH is None else AH

    # # Check if any variable is None after attempting to get from ds
    # if any(v is None for v in [TS_RUL, TR_URB, TB_URB, TC_URB, TG_URB, TA_URB, ALPHAR_URB2D, ALPHAB_URB2D, ALPHAG_URB2D, CHS_URB2D, CHS2_URB2D, Q_RUL]):
    #     raise ValueError("One or more required variables are missing from the dataset.")

    # Now use the above variables to compute T2_URB_DIAG
    # T2_URB_DIAG = (TS_RUL * (1 - ds['FRC_URB2D']) + 
    #                ((ds['R_TBL'] * (TR_URB - TA_URB) * ALPHAR_URB2D / converstion_factor + 
    #                  ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + 
    #                  ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + 
    #                  AH) / CHS_URB2D + TA_URB) * ds['FRC_URB2D']) - \
    #                ((Q_RUL * (1 - ds['FRC_URB2D']) + 
    #                  (ds['R_TBL'] * (TR_URB - TA_URB) * ALPHAR_URB2D / converstion_factor + 
    #                   ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + 
    #                   ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + 
    #                   AH) * ds['FRC_URB2D']) / CHS2_URB2D)
                   
    T2_URB_DIAG = (TS_RUL * (1 - ds['FRC_URB2D']) + 
                    ((ds['R_TBL'] * (TR_URB - TA_URB) * ALPHAR_URB2D / converstion_factor + 
                      ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + 
                      ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + 
                      AH) / CHS_URB2D + TA_URB) * ds['FRC_URB2D']) - \
                    (((TS_RUL-TA_URB)*CHS_URB2D * (1 - ds['FRC_URB2D']) + 
                      (ds['R_TBL'] * (TR_URB - TA_URB) * ALPHAR_URB2D / converstion_factor + 
                      ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + 
                      ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + 
                      AH) * ds['FRC_URB2D']) / CHS2_URB2D)
    return T2_URB_DIAG

def compute_T2_URB_DIAG_TC_NOT_INCLUDED(ds, TS_RUL=None, TR_URB=None, TB_URB=None, TG_URB=None, TA_URB=None, ALPHAR_URB2D=None, ALPHAB_URB2D=None, ALPHAG_URB2D=None, ALPHAC_URB2D=None, AH=None, CHS_URB2D=None, CHS2_URB2D=None, Q_RUL=None):
    # Use the provided replacements if available or default to dataset values
    TS_RUL = ds['TS_RUL'] if TS_RUL is None else TS_RUL
    TR_URB = ds['TR_URB'] if TR_URB is None else TR_URB
    TB_URB = ds['TB_URB'] if TB_URB is None else TB_URB
    TG_URB = ds['TG_URB'] if TG_URB is None else TG_URB
    TA_URB = ds['TA_URB'] if TA_URB is None else TA_URB
    ALPHAR_URB2D = ds['ALPHAR_URB2D'] if ALPHAR_URB2D is None else ALPHAR_URB2D
    ALPHAB_URB2D = ds['ALPHAB_URB2D'] if ALPHAB_URB2D is None else ALPHAB_URB2D
    ALPHAG_URB2D = ds['ALPHAG_URB2D'] if ALPHAG_URB2D is None else ALPHAG_URB2D
    ALPHAC_URB2D = ds['ALPHAC_URB2D'] if ALPHAC_URB2D is None else ALPHAC_URB2D
    CHS_URB2D = ds['CHS_URB2D'] if CHS_URB2D is None else CHS_URB2D
    CHS2_URB2D = ds['CHS2_URB2D'] if CHS2_URB2D is None else CHS2_URB2D
    Q_RUL = ds['Q_RUL'] if Q_RUL is None else Q_RUL
    AH = 0 if AH is None else AH

    # # Check if any variable is None after attempting to get from ds
    # if any(v is None for v in [TS_RUL, TR_URB, TB_URB, TG_URB, TA_URB, ALPHAR_URB2D, ALPHAB_URB2D, ALPHAG_URB2D, ALPHAC_URB2D, CHS_URB2D, CHS2_URB2D, Q_RUL]):
    #     raise ValueError("One or more required variables are missing from the dataset.")

    TC_URB = (ds['RW_TBL']*TA_URB*ALPHAC_URB2D + \
                    ds['RW_TBL']*TG_URB*ALPHAG_URB2D + \
                    ds['W_TBL']*TB_URB*ALPHAB_URB2D + AH*converstion_factor) / \
                  (ds['RW_TBL']*ALPHAC_URB2D + 
                    ds['RW_TBL']*ALPHAG_URB2D + 
                    ds['W_TBL']*ALPHAB_URB2D)
                  
    T2_URB_DIAG = (TS_RUL * (1 - ds['FRC_URB2D']) + 
                    ((ds['R_TBL'] * (TR_URB - TA_URB) * ALPHAR_URB2D / converstion_factor + 
                      ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + 
                      ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + 
                      AH) / CHS_URB2D + TA_URB) * ds['FRC_URB2D']) - \
                    ((Q_RUL * (1 - ds['FRC_URB2D']) + 
                      (ds['R_TBL'] * (TR_URB - TA_URB) * ALPHAR_URB2D / converstion_factor + 
                      ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + 
                      ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + 
                      AH) * ds['FRC_URB2D']) / CHS2_URB2D)
               
    # T2_URB_DIAG = (TS_RUL * (1 - ds['FRC_URB2D']) + 
    #                 ((ds['R_TBL'] * (TR_URB - TA_URB) * ALPHAR_URB2D / converstion_factor + 
    #                   ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + 
    #                   ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + 
    #                   AH) / CHS_URB2D + TA_URB) * ds['FRC_URB2D']) - \
    #                 (((TS_RUL-TA_URB)*CHS_URB2D * (1 - ds['FRC_URB2D']) + 
    #                   (ds['R_TBL'] * (TR_URB - TA_URB) * ALPHAR_URB2D / converstion_factor + 
    #                   ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + 
    #                   ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + 
    #                   AH) * ds['FRC_URB2D']) / CHS2_URB2D)
    return T2_URB_DIAG

#%%
decomposed_values = {}
decomposed_std_dev = {}
utypes = ["all", 1, 2, 3]

for utype in utypes:
    if utype == "all":
        folder_1_filtered = datasets[folder_paths[0]]
    else:
        folder_1_filtered = datasets[folder_paths[0]].where(datasets[folder_paths[0]]['UTYPE_URB'] == utype)

    folder_1_ts_urb_diag_baseline_time_avg = folder_1_filtered['T2_URB_DIAG'].mean(dim='Time') 

    for folder, ds in datasets.items():
        if folder == folder_paths[0]:
            continue

        if utype == "all":
            ds_filtered = ds
        else:
            ds_filtered = ds.where(ds['UTYPE_URB'] == utype)

        results = []
        std_devs = []

        for compute_func, params in [
            (compute_T2_URB_DIAG_TC_NOT_INCLUDED, {"AH": AH_values[folder]}),
            (compute_T2_URB_DIAG_TC_NOT_INCLUDED, {"ALPHAR_URB2D": ds_filtered['ALPHAR_URB2D'], "ALPHAB_URB2D": ds_filtered['ALPHAB_URB2D'], "ALPHAG_URB2D": ds_filtered['ALPHAG_URB2D'], "ALPHAC_URB2D": ds_filtered['ALPHAC_URB2D'], "CHS_URB2D": ds_filtered['CHS_URB2D'], "CHS2_URB2D": ds_filtered['CHS2_URB2D']}),
            (compute_T2_URB_DIAG_TC_NOT_INCLUDED, {"Q_RUL": ds_filtered['Q_RUL'], "TS_RUL": ds_filtered['TS_RUL'], "TR_URB": ds_filtered['TR_URB'], "TB_URB": ds_filtered['TB_URB'], "TG_URB": ds_filtered['TG_URB']}),
            (compute_T2_URB_DIAG_TC_NOT_INCLUDED, {"TA_URB": ds_filtered['TA_URB']})
        ]:
            diff_time_avg = (compute_func(folder_1_filtered, **params).mean(dim='Time') - folder_1_ts_urb_diag_baseline_time_avg) / AH_value
            spatial_avg_diff = diff_time_avg.mean(['south_north', 'west_east']).values
            spatial_std_dev = diff_time_avg.std(['south_north', 'west_east']).values

            results.append(spatial_avg_diff)
            std_devs.append(spatial_std_dev)

        overall_diff_time_avg = ds_filtered['T2_URB_DIAG'].mean(dim='Time') 
        overall_spatial_avg_diff = ((overall_diff_time_avg - folder_1_ts_urb_diag_baseline_time_avg)/ AH_value).mean(['south_north', 'west_east']).values
        overall_spatial_std_dev = ((overall_diff_time_avg - folder_1_ts_urb_diag_baseline_time_avg)/ AH_value).std(['south_north', 'west_east']).values

        cumulative_sum = sum(results)
        cumulative_sum_std_dev = np.sqrt(sum([std_dev**2 for std_dev in std_devs]))

        decomposed_values[(folder, utype)] = [overall_spatial_avg_diff, cumulative_sum] + results
        decomposed_std_dev[(folder, utype)] = [overall_spatial_std_dev, cumulative_sum_std_dev] + std_devs

#%%

fig, axes = plt.subplots(4, 3, figsize=(24, 24))  # Setting up a 4x3 grid
axes = axes.ravel()  # Flattening the axes to easily index them

labels = ["Direct", "Sum", "Baseline", "r", "T", "T$_A$"]
colors = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'black']  # Modify this to your desired colors

subplot_labels = [f'({chr(97 + i)})' for i in range(12)]  # Generates labels (a), (b), ..., (l)

for idx, ((folder, utype), values) in enumerate(decomposed_values.items()):
    ax = axes[idx]
    
    # Fetch the standard deviations for the current folder and utype
    std_devs = decomposed_std_dev[(folder, utype)]
    
    ax.bar(labels, values, color=colors, yerr=std_devs, capsize=10)  # Added yerr for error bars and capsize for cap size on error bars
    #ax.set_ylabel('Mean Value')
    
    method_name = method_map.get(folder, folder)  # Use folder name if method name not found in the dictionary
    ax.set_title(f'{subplot_labels[idx]} {method_name} with urban type = {utype}', fontsize=title_font_size)
    ax.set_xticklabels(labels, rotation=45, ha="right")  # Added ha="right" for better label alignment
    ax.set_ylim(-0.01, 0.02)  # Set the same y-limits for each subplot
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dT_{2}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
    
plt.tight_layout()
plt.show()
fig.savefig(f"T2_figures/T2_decomposition.png", dpi=300)


#%%
titles = ["CHS", "CHS2", "CHS2 - CHS"]

# Set up the figure and axes for the 3-panel plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

utypes = [1, 2, 3]
labels = [f"Urban type = {ut}" for ut in utypes]

# Selecting the first time step value for R_TBL_temp
temp1 = datasets[folder_paths[0]]['CHS_URB2D'].mean(dim='Time')
temp2 = datasets[folder_paths[0]]['CHS2_URB2D'].mean(dim='Time')
temp3 = temp2 - temp1 

data_list = [temp1, temp2, temp3]

for i, data in enumerate(data_list):
    avg_diffs = [data.where(datasets[folder]['UTYPE_URB'] == ut).mean().values for ut in utypes]
    
    # Plotting
    axes[i].bar(labels, avg_diffs, color=['blue', 'green', 'red'], capsize=10)
    axes[i].set_title(titles[i])
    axes[i].set_ylim([0, 25])

plt.tight_layout()
plt.show()

#fig.savefig("figures/Conductances.png", dpi=300)


#%%
titles = ["(a) revised method 1", "(b) method 2", "(c) method 3", "(d) comparison"]

sensitivity_vars = ['H_sensitivity']
colorbar_ticks = np.arange(-0.1, 1.1, 0.1)

def create_colormap():
    # Calculate the proportion of the colorbar that should be blue
    negative_proportion = 0.01 / (0.1 + 0.01)
    positive_proportion = 1 - negative_proportion
    #print(negative_proportion)
    # Define the starting point for the Reds and Blues_r to match the intensity
    blue_start = 1 - negative_proportion  # Start from the lightest blue
    red_start = 0.0  # Start from a lighter red to match the blue intensity
    
    # Sample colors from the Blues_r and Reds colormaps
    blues = plt.cm.Blues_r(np.linspace(blue_start, 1.0, int(256 * negative_proportion)))  # Light to dark blue
    whites = np.array([1.0, 1.0, 1.0, 1.0]).reshape(1,4)  # Pure white color
    reds = plt.cm.Reds(np.linspace(red_start, 1.0, int(256 * positive_proportion)))  # Light to dark red
    
    # Combine them into a single array
    colors = np.vstack((blues, whites, reds))
    
    # Create a new colormap
    cmap = LinearSegmentedColormap.from_list('CustomBluesReds', colors)
    
    return cmap

custom_cmap = create_colormap()

for sensitivity in sensitivity_vars:
    # Create a 1x5 subplot layout for each sensitivity variable
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 6))  # Adjust the figsize as needed
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # Plot the first set of figures (first four panels)
    for i, folder in enumerate(folder_paths[1:]):
        ax = axes[i]
        #ax.set_aspect('equal', adjustable='box')
        norm = matplotlib.colors.Normalize(vmin=-0.1, vmax=1)
        im = datasets[folder][sensitivity].plot(ax=ax, cmap=custom_cmap, norm=norm, add_colorbar=False)
        ax.set_title(titles[i], fontsize=title_font_size)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, ticks=colorbar_ticks)
        cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in colorbar_ticks], fontsize=legend_font_size)

        ax.set_xlim(0, 150)
        ax.set_xticks(range(0, 151, 50))
        ax.set_ylim(0, 150)
        ax.set_yticks(range(0, 151, 50))
        ax.set_xlabel('x', fontsize=axis_label_font_size)
        ax.set_ylabel('y', fontsize=axis_label_font_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

    # Plot the fourth panel as a bar plot for UTYPE_URB != 0
    ax = axes[3]  # The fourth panel
    avg_diffs = [datasets[folder][sensitivity].where(datasets[folder]['UTYPE_URB'] != 0).mean().values for folder in folder_paths[1:]]
    std_devs = [datasets[folder][sensitivity].where(datasets[folder]['UTYPE_URB'] != 0).std().values for folder in folder_paths[1:]]
    ax.bar(methods[0:], avg_diffs, yerr=std_devs, color=['navy', 'green', 'red'], capsize=10)
    print(avg_diffs)
    #x_labels = [f"Method {i+1}" for i in range(len(folder_paths) - 1)]
    #ax.bar(x_labels, avg_diffs, yerr=std_devs, color=['navy', 'lightblue', 'green', 'red'], capsize=10)
    ax.set_title(titles[3], fontsize=title_font_size)
    ax.set_ylim([-0.1, 0.6])  # Adjust as necessary
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dH/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
                     
    plt.tight_layout()
    plt.show()