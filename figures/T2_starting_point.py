#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:07:49 2024

@author: danl
"""

#%%
import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import matplotlib


# Time setup (global to all datasets)
start_datetime_data = pd.Timestamp('2022-07-19 00:00:00')
time_interval = pd.Timedelta(hours=1)
start_time_desired = pd.Timestamp('2022-07-20 00:00:00')
end_time_desired = pd.Timestamp('2022-07-23 00:00:00')
start_index = int((start_time_desired - start_datetime_data) / time_interval)
end_index = int((end_time_desired - start_datetime_data) / time_interval)
variables_to_read = ["TC_URB", "TB_URB", "TG_URB", "TA_URB", "TR_URB", "TS_URB", 
                     "ALPHAC_URB2D", "ALPHAB_URB2D", "ALPHAG_URB2D", 
                     "UTYPE_URB", "LU_INDEX", "FRC_URB2D",
                     "T2", "TSK", "HFX"]


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

# folder_paths = [
#     "NLCD/outputs_no_AH",
#     "NLCD/outputs_AH_option5",
#     "NLCD/outputs_AH_option3",    
#     "NLCD/outputs_AH_option2"
# ]

# AH_value = 10

# AH_values = {
#     "NLCD/outputs_no_AH": 0,
#     "NLCD/outputs_AH_option5": 0,   
#     "NLCD/outputs_AH_option3": 0,       
#     "NLCD/outputs_AH_option2": AH_value
# }


converstion_factor = 0.001*0.24/1004.5

# Set font sizes
axis_label_font_size = 16  # Font size for x and y axis labels
title_font_size = 14       # Font size for the subplot titles
tick_label_font_size = 14  # Font size for tick labels
legend_font_size = 14      # Font size for legend

methods = ["revised method 1", "method 2", "method 3"]


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
                             
                            
                             
    ds_ana['TS_RUL'] =  (ds_ana['TSK'] -   ds_ana['TS_URB']*ds_ana['FRC_URB2D'] ) / (1 - ds_ana['FRC_URB2D'])            
                             
    # create time average outputs for TC_URB_DIAG and TC_URB (masked)
                            
    
    #mask = ds_ana['UTYPE_URB'] != 0
    ds_ana['TC_URB_MASK'] = ds_ana['TC_URB'].where(ds_ana['UTYPE_URB'] != 0)
    ds_ana['tc_urb_avg_time'] = ds_ana['TC_URB_MASK'].mean(dim='Time')
    ds_ana['tc_urb_diag_avg_time'] = ds_ana['TC_URB_DIAG'].mean(dim='Time') 
    # ds_ana['CHS_URB2D_avg_time'] = ds_ana['CHS_URB2D'].mean(dim='Time')
    # ds_ana['CHS2_URB2D_avg_time'] = ds_ana['CHS2_URB2D'].mean(dim='Time')     
    
    return ds_ana


datasets = {}
for folder in folder_paths:
    ds_processed = process_dataset(folder, AH_values[folder])
    datasets[folder] = ds_processed

ref_ds = datasets[folder_paths[0]]

for i, folder in enumerate(folder_paths[1:]):
    datasets[folder]['tc_sensitivity'] = (datasets[folder]['tc_urb_diag_avg_time'] - ref_ds['tc_urb_diag_avg_time']) / AH_value
    datasets[folder]['t2_sensitivity'] = (datasets[folder]['T2'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time') - ref_ds['T2'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time'))/AH_value
    datasets[folder]['ts_sensitivity'] = (datasets[folder]['TSK'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time') - ref_ds['TSK'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time'))/AH_value
    datasets[folder]['ta_sensitivity'] = (datasets[folder]['TA_URB'].mean(dim='Time') - ref_ds['TA_URB'].mean(dim='Time'))/AH_value
    datasets[folder]['tw_sensitivity'] = (datasets[folder]['TB_URB'].mean(dim='Time') - ref_ds['TB_URB'].mean(dim='Time'))/AH_value
    datasets[folder]['tg_sensitivity'] = (datasets[folder]['TG_URB'].mean(dim='Time') - ref_ds['TG_URB'].mean(dim='Time'))/AH_value
    datasets[folder]['tr_sensitivity'] = (datasets[folder]['TR_URB'].mean(dim='Time') - ref_ds['TR_URB'].mean(dim='Time'))/AH_value
    datasets[folder]['tsrural_sensitivity'] = (datasets[folder]['TS_RUL'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time') - ref_ds['TS_RUL'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time'))/AH_value
    datasets[folder]['tsurban_sensitivity'] = (datasets[folder]['TS_URB'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time') - ref_ds['TS_URB'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time'))/AH_value
    
    datasets[folder]['t2_sensitivity_scaled'] = (datasets[folder]['T2'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time') - ref_ds['T2'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time'))/(AH_value*ref_ds['FRC_URB2D'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time'))
    datasets[folder]['ts_sensitivity_scaled'] = (datasets[folder]['TSK'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time') - ref_ds['TSK'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time'))/(AH_value*ref_ds['FRC_URB2D'].where(ref_ds['UTYPE_URB'] != 0).mean(dim='Time'))    


#%% Loop through each sensitivity variable and create plots

titles = ["(a) revised method 1", "(b) method 2", "(c) method 3", "(d) comparison"]

sensitivity_vars = ['tc_sensitivity']
colorbar_ticks = np.arange(-0.01, 0.11, 0.01)

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
        norm = matplotlib.colors.Normalize(vmin=-0.01, vmax=0.1)
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
    ax.set_ylim([0, 0.1])  # Adjust as necessary
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dT_{C}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
                     
    plt.tight_layout()
    plt.show()
    fig.savefig(f"T2_figures/{sensitivity}_option5.png", dpi=300)


#%% Loop through each sensitivity variable and create plots

titles = ["(e) revised method 1", "(f) method 2", "(g) method 3", "(h) comparison"]

sensitivity_vars = ['ts_sensitivity']

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

for sensitivity in sensitivity_vars:
    # Create a 1x5 subplot layout for each sensitivity variable
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 6))  # Adjust the figsize as needed
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # Plot the first set of figures (first four panels)
    for i, folder in enumerate(folder_paths[1:]):
        ax = axes[i]
        #ax.set_aspect('equal', adjustable='box')
        norm = matplotlib.colors.Normalize(vmin=-0.01, vmax=0.04)
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
    ax.set_ylim([0, 0.05])  # Adjust as necessary
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dT_{S}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)

                 
    plt.tight_layout()
    plt.show()
    fig.savefig(f"T2_figures/{sensitivity}_option5.png", dpi=300)


#%% Loop through each sensitivity variable and create plots

# titles = ["(e) revised method 1", "(f) method 2", "(g) method 3", "(h) comparison"]

# sensitivity_vars = ['ts_sensitivity_scaled']

# colorbar_ticks = np.arange(-0.01, 0.051, 0.01)

# def create_colormap():
#     # Calculate the proportion of the colorbar that should be blue
#     negative_proportion = 0.01 / (0.04 + 0.01)
#     positive_proportion = 1 - negative_proportion
#     #print(negative_proportion)
#     # Define the starting point for the Reds and Blues_r to match the intensity
#     blue_start = 1 - negative_proportion  # Start from the lightest blue
#     red_start = 0.0  # Start from a lighter red to match the blue intensity
    
#     # Sample colors from the Blues_r and Reds colormaps
#     blues = plt.cm.Blues_r(np.linspace(blue_start, 1.0, int(256 * negative_proportion)))  # Light to dark blue
#     whites = np.array([1.0, 1.0, 1.0, 1.0]).reshape(1,4)  # Pure white color
#     reds = plt.cm.Reds(np.linspace(red_start, 1.0, int(256 * positive_proportion)))  # Light to dark red
    
#     # Combine them into a single array
#     colors = np.vstack((blues, whites, reds))
    
#     # Create a new colormap
#     cmap = LinearSegmentedColormap.from_list('CustomBluesReds', colors)
    
#     return cmap

# custom_cmap = create_colormap()

# for sensitivity in sensitivity_vars:
#     # Create a 1x5 subplot layout for each sensitivity variable
#     fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 6))  # Adjust the figsize as needed
#     axes = axes.flatten()  # Flatten the axes array for easier indexing

#     # Plot the first set of figures (first four panels)
#     for i, folder in enumerate(folder_paths[1:]):
#         ax = axes[i]
#         #ax.set_aspect('equal', adjustable='box')
#         norm = matplotlib.colors.Normalize(vmin=-0.01, vmax=0.04)
#         im = datasets[folder][sensitivity].plot(ax=ax, cmap=custom_cmap, norm=norm, add_colorbar=False)
#         ax.set_title(titles[i], fontsize=title_font_size)

#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.1)
#         cbar = plt.colorbar(im, cax=cax, ticks=colorbar_ticks)
#         cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in colorbar_ticks], fontsize=legend_font_size)

#         ax.set_xlim(0, 150)
#         ax.set_xticks(range(0, 151, 50))
#         ax.set_ylim(0, 150)
#         ax.set_yticks(range(0, 151, 50))
#         ax.set_xlabel('x', fontsize=axis_label_font_size)
#         ax.set_ylabel('y', fontsize=axis_label_font_size)
#         ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

#     # Plot the fourth panel as a bar plot for UTYPE_URB != 0
#     ax = axes[3]  # The fourth panel
#     avg_diffs = [datasets[folder][sensitivity].where(datasets[folder]['UTYPE_URB'] != 0).mean().values for folder in folder_paths[1:]]
#     std_devs = [datasets[folder][sensitivity].where(datasets[folder]['UTYPE_URB'] != 0).std().values for folder in folder_paths[1:]]
#     ax.bar(methods[0:], avg_diffs, yerr=std_devs, color=['navy', 'green', 'red'], capsize=10)
#     print(avg_diffs)
#     #x_labels = [f"Method {i+1}" for i in range(len(folder_paths) - 1)]
#     #ax.bar(x_labels, avg_diffs, yerr=std_devs, color=['navy', 'lightblue', 'green', 'red'], capsize=10)
#     ax.set_title(titles[3], fontsize=title_font_size)
#     ax.set_ylim([0, 0.2])  # Adjust as necessary
#     ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
#     ax.set_ylabel('$dT_{S}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)

                 
#     plt.tight_layout()
#     plt.show()
    
#%% bar plot changes in dTs/dQ_AH when anthropogenic heat flux is scaled with FRC_URB2D

# # Get the reference dataset from folder 1
# ref_ds = datasets[folder_paths[0]]
# titles = ["revised method 1", "method 2", "method 3"]

# # Set up the figure and axes for a 2x2 grid plot
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

# utypes = ["all", 3, 2, 1]  # 'all' represents all urban types combined
# labels = [f"Urban type = {ut}" for ut in utypes]
# subplot_labels = ['(a)', '(b)', '(c)', '(d)']

# colors = ['blue', 'green', 'red']  # Define the colors for each method

# # Flatten the axes array for easier indexing
# axes = axes.flatten()


# for j, ut in enumerate(utypes):
#     ax = axes[j]
#     for i, folder in enumerate(folder_paths[1:]):  # Starting from the 2nd folder
        
#         if ut == "all":
#             # Calculate for all urban types
#             mask = datasets[folder]['UTYPE_URB'] != 0
#         else:
#             # Calculate for a specific urban type
#             mask = datasets[folder]['UTYPE_URB'] == ut

#         avg_diff = datasets[folder]['ts_sensitivity_scaled'].where(mask).mean().values
#         print(avg_diff)
#         std_dev = datasets[folder]['ts_sensitivity_scaled'].where(mask).std().values

#         # Plotting
#         ax.bar(i, avg_diff, yerr=std_dev, capsize=10, color=colors[i])

#     # Setting the titles and labels with increased font sizes
#     full_title = f"{subplot_labels[j]} {labels[j]}"
#     ax.set_title(full_title, fontsize=title_font_size)
#     ax.set_xticks(range(len(folder_paths) - 1))
#     ax.set_xticklabels([f'method {i+1}' for i in range(len(folder_paths) - 1)], fontsize=tick_label_font_size)
#     ax.set_ylabel('$dT_{S}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
#     ax.set_ylim([0, 0.15])  # Adjust as necessary

#     # Adjusting the font size of tick labels
#     ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

#     # Setting the legend with a specific font size
#     #ax.legend(fontsize=legend_font_size)

# plt.tight_layout()
# plt.show()

#%%

titles = ["(i) revised method 1", "(j) method 2", "(k) method 3", "(l) comparison"]

sensitivity_vars = ['t2_sensitivity']
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

for sensitivity in sensitivity_vars:
    # Create a 2x3 subplot layout for each sensitivity variable
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 6))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # Plot the first set of figures (first row)
    for i, folder in enumerate(folder_paths[1:]):
        ax = axes[i]
        #ax.set_aspect('equal', adjustable='box')
        norm = matplotlib.colors.Normalize(vmin=-0.01, vmax=0.02)
        im = datasets[folder][sensitivity].plot(ax=ax, cmap=custom_cmap_air, norm=norm, add_colorbar=False)
        ax.set_title(titles[i], fontsize=title_font_size)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, ticks=colorbar_ticks)
        cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in colorbar_ticks], fontsize=legend_font_size)
        #cbar.set_ticklabels(['{:.3f}'.format(tick) for tick in np.arange(-0.01, 0.02, 0.005)])

        ax.set_xlim(0, 150)
        ax.set_xticks(range(0, 151, 50))
        ax.set_ylim(0, 150)
        ax.set_yticks(range(0, 151, 50))
        ax.set_xlabel('x', fontsize=axis_label_font_size)
        ax.set_ylabel('y', fontsize=axis_label_font_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

    ax = axes[3]  # The fourth panel
    avg_diffs = [datasets[folder][sensitivity].where(datasets[folder]['UTYPE_URB'] != 0).mean().values for folder in folder_paths[1:]]
    std_devs = [datasets[folder][sensitivity].where(datasets[folder]['UTYPE_URB'] != 0).std().values for folder in folder_paths[1:]]
    ax.bar(methods[0:], avg_diffs, yerr=std_devs, color=['navy', 'green', 'red'], capsize=10)
    
    # Set y-ticks and optionally y-tick labels
    yticks = np.arange(0, 0.021, 0.01)  # Adjust the range and step as needed
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:.2f}'.format(tick) for tick in yticks], fontsize=tick_label_font_size)
    print(avg_diffs)
    #x_labels = [f"Method {i+1}" for i in range(len(folder_paths) - 1)]
    #ax.bar(x_labels, avg_diffs, yerr=std_devs, color=['navy', 'lightblue', 'green', 'red'], capsize=10)
    ax.set_title(titles[3], fontsize=title_font_size)
    ax.set_ylim([0, 0.02])  # Adjust as necessary
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dT_{2}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)

                 
    plt.tight_layout()
    plt.show()
    fig.savefig(f"T2_figures/{sensitivity}_option5.png", dpi=300)

#%%

titles = ["(i) revised method 1", "(j) method 2", "(k) method 3", "(l) comparison"]

sensitivity_vars = ['t2_sensitivity_scaled']
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

for sensitivity in sensitivity_vars:
    # Create a 2x3 subplot layout for each sensitivity variable
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 6))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # Plot the first set of figures (first row)
    for i, folder in enumerate(folder_paths[1:]):
        ax = axes[i]
        #ax.set_aspect('equal', adjustable='box')
        norm = matplotlib.colors.Normalize(vmin=-0.01, vmax=0.02)
        im = datasets[folder][sensitivity].plot(ax=ax, cmap=custom_cmap_air, norm=norm, add_colorbar=False)
        ax.set_title(titles[i], fontsize=title_font_size)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, ticks=colorbar_ticks)
        cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in colorbar_ticks], fontsize=legend_font_size)
        #cbar.set_ticklabels(['{:.3f}'.format(tick) for tick in np.arange(-0.01, 0.02, 0.005)])

        ax.set_xlim(0, 150)
        ax.set_xticks(range(0, 151, 50))
        ax.set_ylim(0, 150)
        ax.set_yticks(range(0, 151, 50))
        ax.set_xlabel('x', fontsize=axis_label_font_size)
        ax.set_ylabel('y', fontsize=axis_label_font_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

    ax = axes[3]  # The fourth panel
    avg_diffs = [datasets[folder][sensitivity].where(datasets[folder]['UTYPE_URB'] != 0).mean().values for folder in folder_paths[1:]]
    std_devs = [datasets[folder][sensitivity].where(datasets[folder]['UTYPE_URB'] != 0).std().values for folder in folder_paths[1:]]
    ax.bar(methods[0:], avg_diffs, yerr=std_devs, color=['navy', 'green', 'red'], capsize=10)
    
    # Set y-ticks and optionally y-tick labels
    yticks = np.arange(0, 0.11, 0.01)  # Adjust the range and step as needed
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:.2f}'.format(tick) for tick in yticks], fontsize=tick_label_font_size)
    print(avg_diffs)
    #x_labels = [f"Method {i+1}" for i in range(len(folder_paths) - 1)]
    #ax.bar(x_labels, avg_diffs, yerr=std_devs, color=['navy', 'lightblue', 'green', 'red'], capsize=10)
    ax.set_title(titles[3], fontsize=title_font_size)
    ax.set_ylim([0, 0.1])  # Adjust as necessary
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dT_{2}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)

                 
    plt.tight_layout()
    plt.show()
#%%
titles = ["(a) revised method 1", "(b) method 2", "(c) method 3", "(d) comparison"]

sensitivity_vars = ['tsurban_sensitivity']
colorbar_ticks = np.arange(-0.01, 0.11, 0.01)

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
        norm = matplotlib.colors.Normalize(vmin=-0.01, vmax=0.1)
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
    ax.bar(methods, avg_diffs, yerr=std_devs, color=['navy', 'green', 'red'], capsize=10)
    
    #x_labels = [f"Method {i+1}" for i in range(len(folder_paths) - 1)]
    #ax.bar(x_labels, avg_diffs, yerr=std_devs, color=['navy', 'lightblue', 'green', 'red'], capsize=10)
    ax.set_title('comparison', fontsize=title_font_size)
    ax.set_ylim([0, 0.1])  # Adjust as necessary
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    ax.set_ylabel('$dT_{U}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size) 
                     
    plt.tight_layout()
    plt.show()
    fig.savefig(f"T2_figures/{sensitivity}_option5.png", dpi=300)
    
#%%    

titles = ["(a) revised method 1", "(b) method 2", "(c) method 3", "(d) comparison"]


sensitivity_vars = ['ta_sensitivity']
colorbar_ticks = np.arange(-0.005, 0.021, 0.005)

def create_colormap_air():
    # Calculate the proportion of the colorbar that should be blue
    negative_proportion = 0.005 / (0.02 + 0.005)
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

for sensitivity in sensitivity_vars:
    # Create a 2x3 subplot layout for each sensitivity variable
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 6))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # Plot the first set of figures (first row)
    for i, folder in enumerate(folder_paths[1:]):
        ax = axes[i]
        #ax.set_aspect('equal', adjustable='box')
        norm = matplotlib.colors.Normalize(vmin=-0.005, vmax=0.02)
        im = datasets[folder][sensitivity].plot(ax=ax, cmap=custom_cmap_air, norm=norm, add_colorbar=False)
        
        ax.set_title(titles[i], fontsize=title_font_size)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, ticks=colorbar_ticks)
        cbar.ax.set_yticklabels(['{:.3f}'.format(tick) for tick in colorbar_ticks], fontsize=legend_font_size)
        #cbar.set_ticklabels(['{:.3f}'.format(tick) for tick in np.arange(-0.01, 0.02, 0.005)])

        ax.set_xlim(0, 150)
        ax.set_xticks(range(0, 151, 50))
        ax.set_ylim(0, 150)
        ax.set_yticks(range(0, 151, 50))
        ax.set_xlabel('x', fontsize=axis_label_font_size)
        ax.set_ylabel('y', fontsize=axis_label_font_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

    ax = axes[3]  # The fourth panel
    avg_diffs = [datasets[folder][sensitivity].where(datasets[folder]['UTYPE_URB'] != 0).mean().values for folder in folder_paths[1:]]
    std_devs = [datasets[folder][sensitivity].where(datasets[folder]['UTYPE_URB'] != 0).std().values for folder in folder_paths[1:]]
    
    ax.bar(methods[0:], avg_diffs, yerr=std_devs, color=['navy', 'green', 'red'], capsize=10)
        
    #x_labels = [f"Method {i+1}" for i in range(len(folder_paths) - 1)]
    #ax.bar(x_labels, avg_diffs, yerr=std_devs, color=['navy', 'lightblue', 'green', 'red'], capsize=10)
    ax.set_title(titles[3], fontsize=title_font_size)   
    ax.set_ylim([0, 0.02])  # Adjust as necessary
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
    if sensitivity == 'ta_sensitivity':
        ax.set_ylabel('$dT_{A}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
    elif sensitivity == 't2_sensitivity':
        ax.set_ylabel('$dT_{2}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
                 
    plt.tight_layout()
    plt.show()
    fig.savefig(f"T2_figures/{sensitivity}_option5.png", dpi=300)
    
        
#%% Loop through each sensitivity variable and create plots

titles_tr = ["(a) revised method 1", "(b) method 2", "(c) method 3", "(d) comparison"]
titles_tw = ["(e) revised method 1", "(f) method 2", "(g) method 3", "(h) comparison"]
titles_tg = ["(i) revised method 1", "(j) method 2", "(k) method 3", "(l) comparison"]
titles_tsrural = ["(m) revised method 1", "(n) method 2", "(o) method 3", "(p) comparison"]

sensitivity_vars = ['tr_sensitivity', 'tw_sensitivity','tg_sensitivity', 'tsrural_sensitivity']

colorbar_ticks = np.arange(-0.02, 0.21, 0.02)

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
        norm = matplotlib.colors.Normalize(vmin=-0.02, vmax=0.2)
        im = datasets[folder][sensitivity].plot(ax=ax, cmap=custom_cmap, norm=norm, add_colorbar=False)
        if sensitivity == 'tr_sensitivity':
            ax.set_title(titles_tr[i], fontsize=title_font_size)
        elif sensitivity == 'tw_sensitivity':
            ax.set_title(titles_tw[i], fontsize=title_font_size)
        elif sensitivity == 'tg_sensitivity':
            ax.set_title(titles_tg[i], fontsize=title_font_size)
        elif sensitivity == 'tsrural_sensitivity':
            ax.set_title(titles_tsrural[i], fontsize=title_font_size)
                
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
    
    #x_labels = [f"Method {i+1}" for i in range(len(folder_paths) - 1)]
    #ax.bar(x_labels, avg_diffs, yerr=std_devs, color=['navy', 'lightblue', 'green', 'red'], capsize=10)
    if sensitivity == 'tr_sensitivity':
        ax.set_title(titles_tr[3], fontsize=title_font_size)
    elif sensitivity == 'tw_sensitivity':
        ax.set_title(titles_tw[3], fontsize=title_font_size)
    elif sensitivity == 'tg_sensitivity':
        ax.set_title(titles_tg[3], fontsize=title_font_size)
    elif sensitivity == 'tsrural_sensitivity':
        ax.set_title(titles_tsrural[3], fontsize=title_font_size)
    
    ax.set_ylim([0, 0.2])  # Adjust as necessary
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

    if sensitivity == 'tr_sensitivity':
        ax.set_ylabel('$dT_{R}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
    elif sensitivity == 'tw_sensitivity':
        ax.set_ylabel('$dT_{W}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
    elif sensitivity == 'tg_sensitivity':
        ax.set_ylabel('$dT_{G}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)
    elif sensitivity == 'tsrural_sensitivity':
        ax.set_ylabel('$dT_{GRASS}/dQ_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size) 
        
    plt.tight_layout()
    plt.show()
    fig.savefig(f"T2_figures/{sensitivity}_option5.png", dpi=300)