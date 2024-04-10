#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:29:59 2023

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
variables_to_read = ["TC_URB", "TB_URB", "TG_URB", "TA_URB", "ALPHAC_URB2D", "ALPHAB_URB2D", "ALPHAG_URB2D", "UTYPE_URB", "LU_INDEX"]


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


converstion_factor = 0.001*0.24/1004.5

# Set font sizes
axis_label_font_size = 16  # Font size for x and y axis labels
title_font_size = 14       # Font size for the subplot titles
tick_label_font_size = 14  # Font size for tick labels
legend_font_size = 14      # Font size for legend

# Set colormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

def create_colormap():
    # Calculate the proportion of the colorbar that should be blue
    negative_proportion = 0.02 / (0.2 + 0.02)
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
                             
    # create time average outputs for TC_URB_DIAG and TC_URB (masked)
                            
    
    #mask = ds_ana['UTYPE_URB'] != 0
    ds_ana['TC_URB_MASK'] = ds_ana['TC_URB'].where(ds_ana['UTYPE_URB'] != 0)
    ds_ana['tc_urb_avg_time'] = ds_ana['TC_URB_MASK'].mean(dim='Time')
    ds_ana['tc_urb_diag_avg_time'] = ds_ana['TC_URB_DIAG'].mean(dim='Time') 
    
    
    return ds_ana


datasets = {}
for folder in folder_paths:
    ds_processed = process_dataset(folder, AH_values[folder])
    datasets[folder] = ds_processed

ref_ds = datasets[folder_paths[0]]

for i, folder in enumerate(folder_paths[1:]):
    datasets[folder]['tc_sensitivity'] = (datasets[folder]['tc_urb_diag_avg_time'] - ref_ds['tc_urb_diag_avg_time']) / AH_value
    
    

#%% plot UTYPE and LUINDEX
# Create a 2-panel figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Plot LU_INDEX at Time = 0 in the second panel
im1 = ref_ds['LU_INDEX'].isel(Time=0).plot(ax=axes[0], cmap='tab20b', levels=list(range(20, 40)), 
                                               cbar_kwargs={'label': '', 'ticks': list(range(20, 40))})  
axes[0].set_title('(a) Land use index', fontsize=title_font_size)

cbar1 = im1.colorbar
cbar1.ax.tick_params(labelsize=tick_label_font_size)  # Set font size of tick labels

# Plot UTYPE_URB at Time = 0 in the first panel
filtered_data = ref_ds['UTYPE_URB'].where(ref_ds['UTYPE_URB'] != 0)
im2 = filtered_data.isel(Time=0).plot(ax=axes[1], cmap='tab20', levels=[0.5, 1.5, 2.5, 3.5], extend='neither', cbar_kwargs={'label': ''})
axes[1].set_title('(b) Urban type', fontsize=title_font_size)

# Set x-axis limits and ticks for both subplots
for ax in axes:
    ax.set_xlim(0, 150)
    ax.set_xticks(list(range(0, 151, 50)))
    ax.set_ylim(0, 150)
    ax.set_yticks(list(range(0, 151, 50)))
    ax.set_xlabel('x', fontsize=axis_label_font_size)
    ax.set_ylabel('y', fontsize=axis_label_font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

# Get the colorbar and set custom ticks and labels
cbar = im2.colorbar
cbar.set_ticks([1, 2, 3])
cbar.set_ticklabels(['1', '2', '3'], fontsize=tick_label_font_size)

plt.tight_layout()
plt.show()    
#fig.savefig("figures/NLCD_landuse.png", dpi=300)
#%% plot TC_URB and TC_URB_DIAG

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
axes = axes.flatten()  # Flatten the 2x2 matrix to make indexing easier

# Iterate over the datasets and plot
for i, (folder, ds) in enumerate(datasets.items()):
    ax = axes[i]
    ax.scatter(ds['tc_urb_avg_time'].values.flatten(),
                ds['tc_urb_diag_avg_time'].values.flatten(),
                marker='o', alpha=0.5)
    
    # You can customize the scatter plot further, e.g., with colors, marker sizes, etc.
    
    # Adding 1:1 line
    ax.plot([290, 310], [290, 310], 'r-', label="1:1 Line")

    # Setting the axis limits
    ax.set_xlim([290, 310])
    ax.set_ylim([290, 310])
    
    ax.set_title(folder.split("/")[-1])  # Setting title as the folder's last name (e.g., "outputs_no_AH")
    ax.set_xlabel("tc_urb_avg_time")
    ax.set_ylabel("tc_urb_diag_avg_time")

# Adjust spacing between plots
plt.tight_layout()
plt.show()
#fig.savefig("figures/NLCD_validation_TC.png", dpi=300)

#%% spatial plot changes in TC_URB_DIAG normalzied by the amount of anthropogenic heat flux 

titles = ["(a) method 1", "(b) method 2", "(c) method 3"]

# Setup the plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

colorbar_ticks = np.arange(-0.02, 0.21, 0.02)

for i, folder in enumerate(folder_paths[1:]):
    #diff_tc_urb_diag_avg_time = (datasets[folder]['tc_urb_diag_avg_time'] - ref_ds['tc_urb_diag_avg_time']) / AH_value
    
    ax = axes[i]
    ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
    # Use the custom colormap with the adjusted scale
    norm = matplotlib.colors.Normalize(vmin=-0.02, vmax=0.2)
    im = datasets[folder]['tc_sensitivity'].plot(ax=ax, cmap=custom_cmap, norm=norm, add_colorbar=False)
    print(title_font_size)
    ax.set_title(titles[i], fontsize=title_font_size)

    # Create a separate colorbar for each subplot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=colorbar_ticks)
    cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in colorbar_ticks], fontsize=legend_font_size)  # Change fontsize here

# Set x-axis limits and ticks for all subplots
for ax in axes:
    ax.set_xlim(0, 150)
    ax.set_xticks(range(0, 151, 50))
    ax.set_ylim(0, 150)
    ax.set_yticks(range(0, 151, 50))
    ax.set_xlabel('x', fontsize=axis_label_font_size)
    ax.set_ylabel('y', fontsize=axis_label_font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

plt.tight_layout()
plt.show()
#fig.savefig("figures/NLCD_spatial_TC.png", dpi=300)

#%% bar plot changes in TC_URB_DIAG normalzied by the amount of anthropogenic heat flux

# Get the reference dataset from folder 1
ref_ds = datasets[folder_paths[0]]
titles = ["method 1", "method 2", "method 3"]

# Set up the figure and axes for a 2x2 grid plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

utypes = ["all", 3, 2, 1]  # 'all' represents all urban types combined
labels = [f"Urban type = {ut}" for ut in utypes]
subplot_labels = ['(a)', '(b)', '(c)', '(d)']

colors = ['blue', 'green', 'red']  # Define the colors for each method

# Flatten the axes array for easier indexing
axes = axes.flatten()


for j, ut in enumerate(utypes):
    ax = axes[j]
    for i, folder in enumerate(folder_paths[1:]):  # Starting from the 2nd folder
        
        if ut == "all":
            # Calculate for all urban types
            mask = datasets[folder]['UTYPE_URB'] != 0
        else:
            # Calculate for a specific urban type
            mask = datasets[folder]['UTYPE_URB'] == ut

        avg_diff = datasets[folder]['tc_sensitivity'].where(mask).mean().values
        print(avg_diff)
        std_dev = datasets[folder]['tc_sensitivity'].where(mask).std().values

        # Plotting
        ax.bar(i, avg_diff, yerr=std_dev, capsize=10, color=colors[i])

    # Setting the titles and labels with increased font sizes
    full_title = f"{subplot_labels[j]} {labels[j]}"
    ax.set_title(full_title, fontsize=title_font_size)
    ax.set_xticks(range(len(folder_paths) - 1))
    ax.set_xticklabels([f'method {i+1}' for i in range(len(folder_paths) - 1)], fontsize=tick_label_font_size)
    ax.set_ylabel('$dT_{C}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
    ax.set_ylim([0, 0.12])  # Adjust as necessary

    # Adjusting the font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

    # Setting the legend with a specific font size
    #ax.legend(fontsize=legend_font_size)

plt.tight_layout()
plt.show()

#fig.savefig("figures/NLCD_bar_TC.png", dpi=300) 


#%%

titles = ["(a) method 1", "(b) method 2", "(c) method 3","(d) method 1", "(e) method 2", "(f) method 3"]

colorbar_ticks = np.arange(-0.02, 0.21, 0.02)

# Create a 2x3 subplot layout
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axes = axes.flatten()  # Flatten the axes array for easier indexing

# Plot the first set of figures (first row)
for i, folder in enumerate(folder_paths[1:]):
    ax = axes[i]
    
    ax.set_aspect('equal', adjustable='box')
    norm = matplotlib.colors.Normalize(vmin=-0.02, vmax=0.2)
    im = datasets[folder]['tc_sensitivity'].plot(ax=ax, cmap=custom_cmap, norm=norm, add_colorbar=False)
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
    
    avg_diffs = [datasets[folder]['tc_sensitivity'].where(datasets[folder]['UTYPE_URB'] == ut).mean().values for ut in utypes]
    std_devs = [datasets[folder]['tc_sensitivity'].where(datasets[folder]['UTYPE_URB'] == ut).std().values for ut in utypes]
    
    
    ax.bar(labels, avg_diffs, yerr=std_devs, color=['blue', 'green', 'red'], capsize=10)
    ax.set_title(titles[i + 3], fontsize=title_font_size)
    ax.set_ylim([0, 0.12])  # Adjust as necessary
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dT_{C}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)

plt.tight_layout()
plt.show()
fig.savefig("figures/NLCD_combined_TC_AH100.png", dpi=300)

