#%%
import xarray as xr
import pandas as pd
import os
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# Read in the Excel data
df = pd.read_excel('NCEI_CDO/OBS.xlsx', engine='openpyxl')
df['DATE'] = pd.to_datetime(df['DATE'])


# List of station IDs to exclude
exclude_stations = {}
exclude_stations = ['72509854704', # NORWOOD MEMORIAL AIRPORT, MA US, not urban
                    '74490714753', # BLUE HILL LCD, MA US, no data and not urban
                    ]  # Replace with your actual station IDs


# Convert TMP to the correct format
df['TMP'] = df['TMP'].str.replace(r'\+', '', regex=True).str.split(',', expand=True)[0].astype(float)

# Convert +9999 to NaN and temperatures to Kelvin
df.loc[df['TMP'] == 9999, 'TMP'] = np.nan
df['TMP'] = (df['TMP'] / 10) + 273.15

# Create a dictionary to store dataframes for each unique station
station_dfs = {}
for station_id, station_data in df.groupby('STATION'):
    station_data.set_index('DATE', inplace=True)
    station_dfs[station_id] = station_data
    
    
all_hourly_temps = []

for station_id, station_data in station_dfs.items():
    if str(station_id) in exclude_stations:
        print(f"Excluding station: {station_id}")
        continue  # Skip this station
    
    # print(f"Data for station {station_id} before resampling:")
    # print("TMP Values:", station_data['TMP'].values)
    # print("\n")  # For better readability

    # Resample to hourly and interpolate
    hourly_data = station_data['TMP'].resample('1H').mean()

    # Shift the time by 1 hour
    hourly_data.index = hourly_data.index + pd.DateOffset(hours=1)
        
    # print(f"Data after station {station_id} before resampling:")
    # print("TMP Values:", hourly_data)
    # print("\n")  # For better readability        
    
    all_hourly_temps.append(hourly_data)


# Concatenate all the hourly dataframes along the columns axis
all_data = pd.concat(all_hourly_temps, axis=1)

# Compute the mean temperature for each hour across all stations
all_data['AVG_TMP'] = all_data.mean(axis=1)

all_data['STD_TMP'] = all_data.std(axis=1)

avg_df = all_data[['AVG_TMP','STD_TMP']].reset_index()
#%%

# Time setup (global to all datasets)
start_datetime_data = pd.Timestamp('2022-07-19 00:00:00')
time_interval = pd.Timedelta(hours=1)
start_time_desired = pd.Timestamp('2022-07-19 00:00:00')
end_time_desired = pd.Timestamp('2022-07-23 00:00:00')
start_index = int((start_time_desired - start_datetime_data) / time_interval)
end_index = int((end_time_desired - start_datetime_data) / time_interval)
variables_to_read = ["T2", "TG_URB", "TA_URB", "TC_URB", "UTYPE_URB", "LU_INDEX","FRC_URB2D","HGT"]

folder_paths = [
    "NLCD/outputs_no_AH",
    "NLCD/outputs_no_AH_rtc",
    "NLCD/outputs_no_AH_ch100",
    "NLCD/outputs_no_AH_rtc_ch100"
]

def process_dataset(folder_path):
    ds = xr.open_mfdataset(os.path.join(folder_path, "wrfout_d03*"), combine='nested', concat_dim='Time')
    ds_selected = ds.sel(Time=slice(start_index, end_index))
    ds_ana = ds_selected[variables_to_read].load()
    return ds_ana

simulations = {}
for folder in folder_paths:
    ds_processed = process_dataset(folder)
    simulations[folder] = ds_processed    

#%%


def nearest_index_1d(array, value):
    """Find the nearest index in a 1D array to a given value."""
    return (np.abs(array - value)).argmin()

# Initialize dictionaries to store simulation stats for each folder
sim_stats = {folder: {'t2_avg': None, 't2_std': None, 'tc_avg': None, 'tc_std': None, 'ta_avg': None, 'ta_std': None, 'tg_avg': None, 'tg_std': None} for folder in folder_paths}

for folder, ds in simulations.items():
    # Initialize accumulators as arrays with the length of the time dimension
    num_time_steps = len(ds['Time'])
    t2_accum = np.zeros(num_time_steps)
    tc_accum = np.zeros(num_time_steps)
    ta_accum = np.zeros(num_time_steps)
    tg_accum = np.zeros(num_time_steps)

    t2_square_accum = np.zeros(num_time_steps)
    tc_square_accum = np.zeros(num_time_steps)
    ta_square_accum = np.zeros(num_time_steps)
    tg_square_accum = np.zeros(num_time_steps)

    station_count = 0

    for station_id, station_data in station_dfs.items():
        print(f"Processing station: {station_id}")
        lat_station = station_data['LATITUDE'].iloc[0]
        lon_station = station_data['LONGITUDE'].iloc[0]
        station_name = station_data['NAME'].iloc[0]

        if str(station_id) in exclude_stations:
            print(f"Excluding station: {station_id}")
            continue  # Skip this station

        station_count += 1   

        # Use the first time step to get the latitude and longitude values
        lats = ds['XLAT'][0].values
        lons = ds['XLONG'][0].values
        # Use the middle value (assuming the data is somewhat evenly spaced) to get the latitude and longitude 1D arrays
        lat_1d = lats[:, int(lons.shape[1] / 2)]
        lon_1d = lons[int(lats.shape[0] / 2), :]

        # Find the nearest latitude and longitude indices in the dataset
        lat_idx = nearest_index_1d(lat_1d, lat_station)
        lon_idx = nearest_index_1d(lon_1d, lon_station)

        # Extract the simulation data for that
        t2_sim = ds['T2'][:, lat_idx, lon_idx]
        tc_sim = ds['TC_URB'][:, lat_idx, lon_idx]
        ta_sim = ds['TA_URB'][:, lat_idx, lon_idx]
        tg_sim = ds['TG_URB'][:, lat_idx, lon_idx]
        lu_index_sim = ds['LU_INDEX'][0, lat_idx, lon_idx]
        utype_sim = ds['UTYPE_URB'][0, lat_idx, lon_idx]
        
        # Use the XTIME coordinate for t2_sim time values
        sim_times = t2_sim['XTIME'].values        
    
        # Accumulate the simulation values
        t2_accum += t2_sim.values
        tc_accum += tc_sim.values
        ta_accum += ta_sim.values
        tg_accum += tg_sim.values
    
        t2_square_accum += np.square(t2_sim.values)
        tc_square_accum += np.square(tc_sim.values)
        ta_square_accum += np.square(ta_sim.values)
        tg_square_accum += np.square(tg_sim.values)

# Calculate averages and standard deviations for each variable
    if station_count > 0:
        sim_stats[folder]['t2_avg'] = t2_accum / station_count
        sim_stats[folder]['tc_avg'] = tc_accum / station_count
        sim_stats[folder]['ta_avg'] = ta_accum / station_count
        sim_stats[folder]['tg_avg'] = tg_accum / station_count
    
        sim_stats[folder]['t2_std'] = np.sqrt(t2_square_accum / station_count - (t2_accum / station_count) ** 2)
        sim_stats[folder]['tc_std'] = np.sqrt(tc_square_accum / station_count - (tc_accum / station_count) ** 2)
        sim_stats[folder]['ta_std'] = np.sqrt(ta_square_accum / station_count - (ta_accum / station_count) ** 2)
        sim_stats[folder]['tg_std'] = np.sqrt(tg_square_accum / station_count - (tg_accum / station_count) ** 2)


#%% Code to plot the average temperature across stations


# Set font sizes
axis_label_font_size = 16  # Font size for x and y axis labels
title_font_size = 14       # Font size for the subplot titles
tick_label_font_size = 12  # Font size for tick labels
legend_font_size = 12      # Font size for legend

subplot_titles = ["(a) Case 1", "(b) Case 2", "(c) Case 3", "(d) Case 4"]

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten()  # Flatten the array to easily iterate over it

for i, folder in enumerate(sim_stats.keys()):
    ax = axs[i]

    # Plotting observation data
    ax.plot(avg_df['DATE'], avg_df['AVG_TMP'], label='OBS', color='blue')
    ax.fill_between(avg_df['DATE'], avg_df['AVG_TMP'] - avg_df['STD_TMP'], avg_df['AVG_TMP'] + avg_df['STD_TMP'], color='blue', alpha=0.2)

    # Plotting WRF data for the current folder
    t2_avg = sim_stats[folder]['t2_avg']
    t2_std = sim_stats[folder]['t2_std']
    ax.plot(sim_times, t2_avg, label='WRF', color='red')
    ax.fill_between(sim_times, t2_avg - t2_std, t2_avg + t2_std, color='red', alpha=0.2)

    # Set the locator and formatter for the x-axis
    locator = mdates.HourLocator(interval=24)  # every 24 hours
    formatter = mdates.DateFormatter('%m-%d')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Set x-axis limits
    ax.set_xlim(start_time_desired, end_time_desired)

    # Customize the tick parameters
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

    # Other plot settings
    ax.set_title(subplot_titles[i], fontsize=title_font_size)
    ax.set_ylim(290, 310)
    ax.legend(loc='upper left', fontsize=legend_font_size)
    ax.set_ylabel("Temperature (K)", fontsize=axis_label_font_size)
    ax.grid(True)

# Adjust layout
plt.tight_layout()
fig.savefig("figures/OBS_AVERAGE.png", dpi=300)
# %%

ref_ds = simulations[folder_paths[0]]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

# Plot HGT at Time = 0 in the first panel (upper left)
im1 = ref_ds['HGT'].isel(Time=0).plot(ax=axes[0, 0], cmap='terrain', vmin=0, vmax=500, cbar_kwargs={'label': ''}) #, cbar_kwargs={'label': 'Terrain height (m)'}
axes[0, 0].set_title('(a) Terrain Height (m)', fontsize=14)


# Plot LU_INDEX at Time = 0 in the third panel (lower left)
im2 = ref_ds['LU_INDEX'].isel(Time=0).plot(ax=axes[0, 1], cmap='tab20b', levels=list(range(20, 40)), cbar_kwargs={'ticks': list(range(20, 40)),'label': ''})
axes[0, 1].set_title('(b) Land Use Index', fontsize=14)

# Mask FRC_URB2D where UTYPE_URB is not 0 before plotting
masked_frc_urb2d = ref_ds['FRC_URB2D'].where(ref_ds['UTYPE_URB'] != 0)
# Plot masked FRC_URB2D at Time = 0 in the second panel (upper right)
im3 = masked_frc_urb2d.isel(Time=0).plot(ax=axes[1, 0], cmap='coolwarm', vmin=0, vmax=1, cbar_kwargs={'label': ''})
axes[1, 0].set_title('(c) Impervious Surface Fraction', fontsize=14)


# Plot UTYPE_URB at Time = 0 in the fourth panel (lower right)
filtered_data = ref_ds['UTYPE_URB'].where(ref_ds['UTYPE_URB'] != 0)
im4 = filtered_data.isel(Time=0).plot(ax=axes[1, 1], cmap='tab20', levels=[0.5, 1.5, 2.5, 3.5], extend='neither', cbar_kwargs={'label': ''})
axes[1, 1].set_title('(d) Urban Type', fontsize=14)

# Loop through each station and plot it on both panels
for station_id, station_data in station_dfs.items():
    if str(station_id) not in exclude_stations:
        print(f"Processing station: {station_id}")
        lat_station = station_data['LATITUDE'].iloc[0]
        lon_station = station_data['LONGITUDE'].iloc[0]
        station_name = station_data['NAME'].iloc[0]


        # Use the first time step to get the latitude and longitude values
        lats = ds['XLAT'][0].values
        lons = ds['XLONG'][0].values
        # Use the middle value (assuming the data is somewhat evenly spaced) to get the latitude and longitude 1D arrays
        lat_1d = lats[:, int(lons.shape[1] / 2)]
        lon_1d = lons[int(lats.shape[0] / 2), :]

        # Find the nearest latitude and longitude indices in the dataset
        lat_idx = nearest_index_1d(lat_1d, lat_station)
        lon_idx = nearest_index_1d(lon_1d, lon_station)
        
        
        # Convert lat/lon to dataset coordinate system if necessary
        # For demonstration, we directly use lat and lon assuming they match the dataset's system
        # You might need to adjust this depending on your dataset's coordinate system
        axes[0,1].plot(lon_idx, lat_idx, marker='^', color='red', markersize=5, fillstyle='none', markeredgewidth=2)
        axes[1,1].plot(lon_idx, lat_idx, marker='^', color='red', markersize=5, fillstyle='none', markeredgewidth=2)

# Adjust subplot settings as before
for ax_row in axes:
    for ax in ax_row:
        ax.set_xlim(0, 150)
        ax.set_xticks(list(range(0, 151, 50)))
        ax.set_ylim(0, 150)
        ax.set_yticks(list(range(0, 151, 50)))
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)


# Adjust colorbar settings as before
# For HGT plot
cbar1 = im1.colorbar
cbar1.ax.tick_params(labelsize=14) # Adjust font size as needed

# For FRC_URB2D plot, ensure it reflects the intended range 0 to 1
# No need to manually set ticks here as vmin and vmax are already specified
cbar2 = im2.colorbar
cbar2.ax.tick_params(labelsize=14) # Adjust font size as needed

# Similarly, adjust colorbars for the LU_INDEX and UTYPE_URB plots as needed
cbar3 = im3.colorbar
cbar3.ax.tick_params(labelsize=14) # Adjust for LU_INDEX plot

cbar4 = im4.colorbar
# For UTYPE_URB, if you need specific ticks and labels:
cbar4.set_ticks([1, 2, 3]) # Adjust ticks as needed
cbar4.set_ticklabels(['1', '2', '3'], fontsize=14) # Adjust labels and font size as needed


plt.tight_layout()
plt.show()
fig.savefig("figures/NLCD_landuse_with_stations_with_frcurb.png", dpi=300)

#%%

# Create a 2-panel figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Plot LU_INDEX at Time = 0 in the second panel
im1 = ref_ds['LU_INDEX'].isel(Time=0).plot(ax=axes[0], cmap='tab20b', levels=list(range(20, 40)), 
                                               cbar_kwargs={'label': '', 'ticks': list(range(20, 40))})  
axes[0].set_title('(a) Land use index', fontsize=title_font_size)

# Plot UTYPE_URB at Time = 0 in the first panel
filtered_data = ref_ds['UTYPE_URB'].where(ref_ds['UTYPE_URB'] != 0)
im2 = filtered_data.isel(Time=0).plot(ax=axes[1], cmap='tab20', levels=[0.5, 1.5, 2.5, 3.5], extend='neither', 
                                      cbar_kwargs={'label': ''})
axes[1].set_title('(b) Urban type', fontsize=title_font_size)

# Loop through each station and plot it on both panels
for station_id, station_data in station_dfs.items():
    if str(station_id) not in exclude_stations:
        print(f"Processing station: {station_id}")
        lat_station = station_data['LATITUDE'].iloc[0]
        lon_station = station_data['LONGITUDE'].iloc[0]
        station_name = station_data['NAME'].iloc[0]


        # Use the first time step to get the latitude and longitude values
        lats = ds['XLAT'][0].values
        lons = ds['XLONG'][0].values
        # Use the middle value (assuming the data is somewhat evenly spaced) to get the latitude and longitude 1D arrays
        lat_1d = lats[:, int(lons.shape[1] / 2)]
        lon_1d = lons[int(lats.shape[0] / 2), :]

        # Find the nearest latitude and longitude indices in the dataset
        lat_idx = nearest_index_1d(lat_1d, lat_station)
        lon_idx = nearest_index_1d(lon_1d, lon_station)
        
        
        # Convert lat/lon to dataset coordinate system if necessary
        # For demonstration, we directly use lat and lon assuming they match the dataset's system
        # You might need to adjust this depending on your dataset's coordinate system
        axes[0].plot(lon_idx, lat_idx, marker='^', color='red', markersize=5, fillstyle='none', markeredgewidth=2)
        axes[1].plot(lon_idx, lat_idx, marker='^', color='red', markersize=5, fillstyle='none', markeredgewidth=2)

# Adjust subplot settings as before
for ax in axes:
    ax.set_xlim(0, 150)
    ax.set_xticks(list(range(0, 151, 50)))
    ax.set_ylim(0, 150)
    ax.set_yticks(list(range(0, 151, 50)))
    ax.set_xlabel('x', fontsize=axis_label_font_size)
    ax.set_ylabel('y', fontsize=axis_label_font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

# Adjust colorbar settings as before
cbar1 = im1.colorbar
cbar1.ax.tick_params(labelsize=tick_label_font_size)

cbar = im2.colorbar
cbar.set_ticks([1, 2, 3])
cbar.set_ticklabels(['1', '2', '3'], fontsize=tick_label_font_size)

plt.tight_layout()
plt.show()
fig.savefig("figures/NLCD_landuse_with_stations.png", dpi=300)
