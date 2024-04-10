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

variables_to_read = ["TC_URB", "TB_URB", "TG_URB", "TA_URB", "TR_URB",  "TSK", "TS_URB", "T2",
                     "ALPHAC_URB2D", "ALPHAB_URB2D", "ALPHAG_URB2D", "ALPHAR_URB2D",
                     "UTYPE_URB", "LU_INDEX", "FRC_URB2D",
                     "HFX","SH_URB","CHS_URB2D", "CHS2_URB2D"]


AH_option_to_study = 2


if AH_option_to_study == 2 : 
    # Organize folders and AH values for cases
    folder_paths_cases = {
        "Case 1": [
            "T2_test/T2_test_no_AH",
            "T2_test/T2_test_AH_option2"
        ],
        "Case 2": [
            "T2_test/T2_test_rtc_no_AH",
            "T2_test/T2_test_AH_option2_rtc"
        ],
        "Case 3": [
            "T2_test/T2_test_ch100_no_AH",
            "T2_test/T2_test_AH_option2_ch100"
        ],
        "Case 4": [
            "T2_test/T2_test_rtc_ch100_no_AH",
            "T2_test/T2_test_AH_option2_rtc_ch100"
        ]    
    }
    
    AH_value = 100
    
    AH_values_cases = {
        "Case 1": [0, AH_value],
        "Case 2": [0, AH_value],
        "Case 3": [0, AH_value],
        "Case 4": [0, AH_value],   
    }

    AH_T2_values_cases = {
        "Case 1": [0, 0],
        "Case 2": [0, 0],
        "Case 3": [0, 0],
        "Case 4": [0, 0],   
    }     
    
elif AH_option_to_study == 5 : 
    
    # Organize folders and AH values for cases
    folder_paths_cases = {
        "Case 1": [
            "T2_test/T2_test_no_AH",
            "T2_test/T2_test_AH_option5"
        ],
        "Case 2": [
            "T2_test/T2_test_rtc_no_AH",
            "T2_test/T2_test_AH_option5_rtc"
        ],
        "Case 3": [
            "T2_test/T2_test_ch100_no_AH",
            "T2_test/T2_test_AH_option5_ch100"
        ],
        "Case 4": [
            "T2_test/T2_test_rtc_ch100_no_AH",
            "T2_test/T2_test_AH_option5_rtc_ch100"
        ]    
    }
    
    AH_value = 100
    
    AH_values_cases = {
        "Case 1": [0, 0],
        "Case 2": [0, 0],
        "Case 3": [0, 0],
        "Case 4": [0, 0],   
    }
    
    AH_T2_values_cases = {
        "Case 1": [0, AH_value],
        "Case 2": [0, AH_value],
        "Case 3": [0, AH_value],
        "Case 4": [0, AH_value],   
    }     
    
elif AH_option_to_study == 3 : 
    
    # Organize folders and AH values for cases
    folder_paths_cases = {
        "Case 1": [
            "T2_test/T2_test_no_AH",
            "T2_test/T2_test_AH_option3"
        ],
        "Case 2": [
            "T2_test/T2_test_rtc_no_AH",
            "T2_test/T2_test_AH_option3_rtc"
        ],
        "Case 3": [
            "T2_test/T2_test_ch100_no_AH",
            "T2_test/T2_test_AH_option3_ch100"
        ],
        "Case 4": [
            "T2_test/T2_test_rtc_ch100_no_AH",
            "T2_test/T2_test_AH_option3_rtc_ch100"
        ]    
    }
    
    AH_value = 100
    
    AH_values_cases = {
        "Case 1": [0, 0],
        "Case 2": [0, 0],
        "Case 3": [0, 0],
        "Case 4": [0, 0],   
    }    
    
    AH_T2_values_cases = {
        "Case 1": [0, 0],
        "Case 2": [0, 0],
        "Case 3": [0, 0],
        "Case 4": [0, 0],   
    }        
    

converstion_factor = 0.001 * 0.24 / 1004.5


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
                             
    ds_ana['dTC_URB_DIAG_dAH'] = (converstion_factor) / \
                             (ds_ana['RW_TBL']*ds_ana['ALPHAC_URB2D'] + 
                              ds_ana['RW_TBL']*ds_ana['ALPHAG_URB2D'] + 
                              ds_ana['W_TBL']*ds_ana['ALPHAB_URB2D'])                             
                             
                             
    # create time average outputs for TC_URB_DIAG and TC_URB (masked)
                            
    
    #mask = ds_ana['UTYPE_URB'] != 0
    ds_ana['TC_URB_MASK'] = ds_ana['TC_URB'].where(ds_ana['UTYPE_URB'] != 0)
    ds_ana['tc_urb_avg_time'] = ds_ana['TC_URB_MASK'].mean(dim='Time')
    ds_ana['tc_urb_diag_avg_time'] = ds_ana['TC_URB_DIAG'].mean(dim='Time') 
    
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


def process_dataset_rtc(folder_path, AH, AH_T2_value):
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
    
    ds_ana['TS_RUL'] =  (ds_ana['TSK'] -   ds_ana['TS_URB']*ds_ana['FRC_URB2D'] ) / (1 - ds_ana['FRC_URB2D'])            
    
    ds_ana['Qu'] = (ds_ana['TC_URB']-ds_ana['TA_URB'])*ds_ana['ALPHAC_URB2D']/converstion_factor 
                             
    ds_ana['Qu2'] = ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                   ds_ana['W_TBL']*(ds_ana['TB_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAB_URB2D']/converstion_factor + \
                   ds_ana['RW_TBL']*(ds_ana['TG_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAG_URB2D']/converstion_factor + \
                       + AH
                       
        
    #ds_ana['QT'] = ds_ana['Q_RUL']* (1 - ds_ana['FRC_URB2D'])  + ds_ana['Qu']*ds_ana['FRC_URB2D']
                         
    
    ds_ana['Tu'] = ds_ana['Qu']/ds_ana['CHS_URB2D'] + ds_ana['TA_URB']
    
    ds_ana['TS_URB_DIAG'] = ds_ana['TS_RUL']* (1 - ds_ana['FRC_URB2D']) + ((ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                   ds_ana['W_TBL']*(ds_ana['TB_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAB_URB2D']/converstion_factor + \
                   ds_ana['RW_TBL']*(ds_ana['TG_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAG_URB2D']/converstion_factor + \
                       + AH)/ds_ana['CHS_URB2D'] + ds_ana['TA_URB']) *ds_ana['FRC_URB2D']
    
    ds_ana['HFX_corrected'] = ds_ana['HFX']-AH_T2_value*ds_ana['FRC_URB2D']
    
    ds_ana['Q_RUL'] =  (ds_ana['HFX_corrected'] - ds_ana['SH_URB']*ds_ana['FRC_URB2D']) / (1 - ds_ana['FRC_URB2D'])
    
    
    ds_ana['T2_URB_DIAG'] = ds_ana['TS_RUL']* (1 - ds_ana['FRC_URB2D']) + ((ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                   ds_ana['W_TBL']*(ds_ana['TB_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAB_URB2D']/converstion_factor + \
                   ds_ana['RW_TBL']*(ds_ana['TG_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAG_URB2D']/converstion_factor + \
                       + AH)/ds_ana['CHS_URB2D'] + ds_ana['TA_URB']) *ds_ana['FRC_URB2D'] -  \
        (ds_ana['Q_RUL']*(1 - ds_ana['FRC_URB2D'])+(ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                                      ds_ana['W_TBL']*(ds_ana['TB_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAB_URB2D']/converstion_factor + \
                                      ds_ana['RW_TBL']*(ds_ana['TG_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAG_URB2D']/converstion_factor + \
                                          + AH)*ds_ana['FRC_URB2D'])/ds_ana['CHS2_URB2D']
    
    ds_ana['T2_URB_DIAG2'] = ds_ana['TS_RUL']* (1 - ds_ana['FRC_URB2D']) + ((ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
                   ds_ana['W_TBL']*(ds_ana['TB_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAB_URB2D']/converstion_factor + \
                   ds_ana['RW_TBL']*(ds_ana['TG_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAG_URB2D']/converstion_factor + \
                       + AH)/ds_ana['CHS_URB2D'] + ds_ana['TA_URB']) *ds_ana['FRC_URB2D'] -  \
        ((ds_ana['TS_RUL']-ds_ana['TA_URB'])*ds_ana['CHS_URB2D']*(1 - ds_ana['FRC_URB2D'])+(ds_ana['R_TBL']*(ds_ana['TR_URB']-ds_ana['TC_URB'])*ds_ana['ALPHAR_URB2D']/converstion_factor + \
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

# Process datasets for each case
datasets = {case: [] for case in folder_paths_cases}

for case, folder_paths in folder_paths_cases.items():
    for i, folder in enumerate(folder_paths):
        AH = AH_values_cases[case][i]
        AH_T2 = AH_T2_values_cases[case][i]
        # Use the original process_dataset for Case 1 and Case 2
        if case in ["Case 1", "Case 3"]:
            ds_processed = process_dataset(folder, AH, AH_T2)
        # Use the new process_dataset_v2 for Case 3 and Case 4
        else:
            ds_processed = process_dataset_rtc(folder, AH, AH_T2)
        datasets[case].append(ds_processed)
        
  
#%% 
# Set font sizes
axis_label_font_size = 14  # Font size for x and y axis labels
title_font_size = 14       # Font size for the subplot titles
tick_label_font_size = 14  # Font size for tick labels
legend_font_size = 14      # Font size for legend

# # # Create a 2x2 grid of subplots
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
# axes = axes.flatten()  # Flatten the 2x2 matrix to make indexing easier


# for i, (case, ds_list) in enumerate(datasets.items()):
#     ds = ds_list[0]  # Get the first dataset for each case

#     # Extract the variables and flatten the arrays
#     CHS_URB = ds['Qu_avg_time'].values.flatten()
#     CHS2_URB = ds['Qu2_diag_avg_time'].values.flatten()
    
#     # Use the correct axis from the grid, assuming only one column of subplots
#     if len(datasets) > 1:
#         ax = axes[i]
#     else:
#         ax = axes  # If there's only one subplot, axs is not an array

#     ax.scatter(CHS_URB, CHS2_URB, marker='o', alpha=0.5)
#     # # Adding 1:1 line
#     # Adding 1:1 line
#     # # Adding 1:1 line
#     ax.plot([50, 250], [50, 250], 'r-', label="1:1 Line")

#     # Setting the axis limits
#     ax.set_xlim([50, 250])
#     ax.set_ylim([50, 250])
    
#     ax.set_title(f'{case}')
#     ax.set_xlabel("Q$_U$", fontsize=axis_label_font_size)
#     ax.set_ylabel("Diagnosed Q$_U$", fontsize=axis_label_font_size)

# # Show the plots
# plt.show()

# # Create a 2x2 grid of subplots
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
# axes = axes.flatten()  # Flatten the 2x2 matrix to make indexing easier


# for i, (case, ds_list) in enumerate(datasets.items()):
#     ds = ds_list[1]  # Get the first dataset for each case

#     # Extract the variables and flatten the arrays
#     CHS_URB = ds['ts_urb_avg_time'].values.flatten()
#     CHS2_URB = ds['ts_urb_diag_avg_time'].values.flatten()
    
#     # Use the correct axis from the grid, assuming only one column of subplots
#     if len(datasets) > 1:
#         ax = axes[i]
#     else:
#         ax = axes  # If there's only one subplot, axs is not an array

#     ax.scatter(CHS_URB, CHS2_URB, marker='o', alpha=0.5)
#     # # Adding 1:1 line
#     # Adding 1:1 line
#     ax.plot([290, 310], [290, 310], 'r-', label="1:1 Line")

#     # Setting the axis limits
#     ax.set_xlim([290, 310])
#     ax.set_ylim([290, 310])
    
#     ax.set_title(f'{case}')
#     ax.set_xlabel("T$_S$ (K)", fontsize=axis_label_font_size)
#     ax.set_ylabel("Diagnosed T$_S$ (K)", fontsize=axis_label_font_size)

# # Show the plots
# plt.show()

# # Create a 2x2 grid of subplots
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
# axes = axes.flatten()  # Flatten the 2x2 matrix to make indexing easier


# for i, (case, ds_list) in enumerate(datasets.items()):
#     ds = ds_list[1]  # Get the first dataset for each case

#     # Extract the variables and flatten the arrays
#     CHS_URB = ds['t2_urb_avg_time'].values.flatten()
#     CHS2_URB = ds['t2_urb_diag_avg_time'].values.flatten()
    
#     # Use the correct axis from the grid, assuming only one column of subplots
#     if len(datasets) > 1:
#         ax = axes[i]
#     else:
#         ax = axes  # If there's only one subplot, axs is not an array

#     ax.scatter(CHS_URB, CHS2_URB, marker='o', alpha=0.5)
#     # # Adding 1:1 line
#     # Adding 1:1 line
#     ax.plot([290, 310], [290, 310], 'r-', label="1:1 Line")

#     # Setting the axis limits
#     ax.set_xlim([290, 310])
#     ax.set_ylim([290, 310])
    
#     ax.set_title(f'{case}')
#     ax.set_xlabel("T$_2$ (K)", fontsize=axis_label_font_size)
#     ax.set_ylabel("Diagnosed T$_2$ (K)", fontsize=axis_label_font_size)

# # Show the plots
# plt.show()
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

 #%% The first one has TC while the second one does not have TC

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

def compute_TS_URB_DIAG_TC_NOT_INCLUDED_rtc(ds, TS_RUL=None, TR_URB=None, TB_URB=None, TG_URB=None, TA_URB=None, ALPHAR_URB2D=None, ALPHAB_URB2D=None, ALPHAG_URB2D=None, ALPHAC_URB2D=None, AH=None, CHS_URB2D=None):
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
    
                  
    TC_URB = (TA_URB*ALPHAC_URB2D + \
                    ds['RW_TBL']*TG_URB*ALPHAG_URB2D + \
                    ds['R_TBL']*TR_URB*ALPHAR_URB2D + \
                    ds['W_TBL']*TB_URB*ALPHAB_URB2D + AH*converstion_factor) / \
                  (ALPHAC_URB2D + 
                    ds['RW_TBL']*ALPHAG_URB2D +
                    ds['R_TBL']*ALPHAR_URB2D +                    
                    ds['W_TBL']*ALPHAB_URB2D)
                  
    TS_URB_DIAG = TS_RUL * (1 - ds['FRC_URB2D']) + \
        ((ds['R_TBL'] * (TR_URB - TC_URB) * ALPHAR_URB2D / converstion_factor + \
          ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + \
          ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + \
          AH) / CHS_URB2D + TA_URB) * ds['FRC_URB2D']

    return TS_URB_DIAG
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
        
        baseline_tc_urb_diag_avg = baseline_filtered['TS_URB_DIAG'].mean(dim='Time')

        for i, ds in enumerate(ds_list[1:], 1):  # Starting from the second dataset in the list
            ds_filtered = ds if utype == "all" else ds.where(ds['UTYPE_URB'] == utype, drop=True)
            AH = AH_values_cases[case][i]

            results = []
            std_devs = []
            
            if case in ["Case 1", "Case 3"]:

    
                for compute_func, params in [
                    (compute_TS_URB_DIAG_TC_NOT_INCLUDED, {"AH": AH}),
                    (compute_TS_URB_DIAG_TC_NOT_INCLUDED, {"ALPHAR_URB2D": ds_filtered['ALPHAR_URB2D'], "ALPHAB_URB2D": ds_filtered['ALPHAB_URB2D'], "ALPHAG_URB2D": ds_filtered['ALPHAG_URB2D'], "ALPHAC_URB2D": ds_filtered['ALPHAC_URB2D'],  "CHS_URB2D": ds_filtered['CHS_URB2D']}),
                    (compute_TS_URB_DIAG_TC_NOT_INCLUDED, {"TS_RUL": ds_filtered['TS_RUL'], "TR_URB": ds_filtered['TR_URB'],"TB_URB": ds_filtered['TB_URB'], "TG_URB": ds_filtered['TG_URB']}),
                    (compute_TS_URB_DIAG_TC_NOT_INCLUDED, {"TA_URB": ds_filtered['TA_URB']})
                ]:
                    diff_time_avg = (compute_func(baseline_filtered, **params).mean(dim='Time') - baseline_tc_urb_diag_avg) / AH_value
                    spatial_avg_diff = diff_time_avg.mean(['south_north', 'west_east']).values
                    spatial_std_dev = diff_time_avg.std(['south_north', 'west_east']).values
    
                    results.append(spatial_avg_diff)
                    std_devs.append(spatial_std_dev)
                
            else: 
                
                for compute_func, params in [
                    (compute_TS_URB_DIAG_TC_NOT_INCLUDED_rtc, {"AH": AH}),
                    (compute_TS_URB_DIAG_TC_NOT_INCLUDED_rtc, {"ALPHAR_URB2D": ds_filtered['ALPHAR_URB2D'], "ALPHAB_URB2D": ds_filtered['ALPHAB_URB2D'], "ALPHAG_URB2D": ds_filtered['ALPHAG_URB2D'], "ALPHAC_URB2D": ds_filtered['ALPHAC_URB2D'],  "CHS_URB2D": ds_filtered['CHS_URB2D']}),
                    (compute_TS_URB_DIAG_TC_NOT_INCLUDED_rtc, {"TS_RUL": ds_filtered['TS_RUL'], "TR_URB": ds_filtered['TR_URB'],"TB_URB": ds_filtered['TB_URB'], "TG_URB": ds_filtered['TG_URB']}),
                    (compute_TS_URB_DIAG_TC_NOT_INCLUDED_rtc, {"TA_URB": ds_filtered['TA_URB']})
                ]:
                    diff_time_avg = (compute_func(baseline_filtered, **params).mean(dim='Time') - baseline_tc_urb_diag_avg) / AH_value
                    spatial_avg_diff = diff_time_avg.mean(['south_north', 'west_east']).values
                    spatial_std_dev = diff_time_avg.std(['south_north', 'west_east']).values
    
                    results.append(spatial_avg_diff)
                    std_devs.append(spatial_std_dev)       
                    

            overall_diff_time_avg = ds_filtered['TS_URB_DIAG'].mean(dim='Time') 
            overall_spatial_avg_diff = ((overall_diff_time_avg - baseline_tc_urb_diag_avg) / AH_value).mean(['south_north', 'west_east']).values
            overall_spatial_std_dev = ((overall_diff_time_avg - baseline_tc_urb_diag_avg) / AH_value).std(['south_north', 'west_east']).values

            cumulative_sum = sum(results)
            cumulative_sum_std_dev = np.sqrt(sum([std_dev**2 for std_dev in std_devs]))

            decomposed_values[(case, utype)] = [overall_spatial_avg_diff, cumulative_sum] + results
            decomposed_std_dev[(case, utype)] = [overall_spatial_std_dev, cumulative_sum_std_dev] + std_devs    
#%%
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()  # Flattening the 2D axes array into 1D for easier iteration

labels = ["Direct", "Sum", "Baseline", "r", "T", "T$_A$"]
colors = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'black']  # Modify this to your desired colors

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
    print(values)
    ax.bar(labels, values, color=colors, yerr=std_devs, capsize=5)
    full_title = f"{subplot_labels[i]} {case}"
    ax.set_title(full_title, fontsize=title_font_size)    
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_label_font_size)
    if AH_option_to_study == 2 : 
        ax.set_ylim([-0.01, 0.04])  # Adjust as needed based on your data
    elif AH_option_to_study == 5 :
        ax.set_ylim([-0.01, 0.04])  # Adjust as needed based on your data
    elif AH_option_to_study == 3 :
        ax.set_ylim([-0.01, 0.04])  # Adjust as needed based on your data
            
    ax.set_ylabel('dT$_{S}$/dQ$_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)

    # Adjusting the font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

plt.tight_layout()
plt.show()

if AH_option_to_study == 2 : 
    fig.savefig(f"T2_figures/TS_decomposition_all_4cases_AH100_AH_option_2_utype_{utype}.png", dpi=300)
elif AH_option_to_study == 5 :
    fig.savefig(f"T2_figures/TS_decomposition_all_4cases_AH100_AH_option_5_utype_{utype}.png", dpi=300)
elif AH_option_to_study == 3 :
    fig.savefig(f"T2_figures/TS_decomposition_all_4cases_AH100_AH_option_3_utype_{utype}.png", dpi=300)


#%%

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
    return T2_URB_DIAG


def compute_T2_URB_DIAG_TC_NOT_INCLUDED_rtc(ds, TS_RUL=None, TR_URB=None, TB_URB=None, TG_URB=None, TA_URB=None, ALPHAR_URB2D=None, ALPHAB_URB2D=None, ALPHAG_URB2D=None, ALPHAC_URB2D=None, AH=None, CHS_URB2D=None, CHS2_URB2D=None, Q_RUL=None):
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

                  
    TC_URB = (TA_URB*ALPHAC_URB2D + \
                    ds['RW_TBL']*TG_URB*ALPHAG_URB2D + \
                    ds['R_TBL']*TR_URB*ALPHAR_URB2D + \
                    ds['W_TBL']*TB_URB*ALPHAB_URB2D + AH*converstion_factor) / \
                  (ALPHAC_URB2D + 
                    ds['RW_TBL']*ALPHAG_URB2D +
                    ds['R_TBL']*ALPHAR_URB2D +                    
                    ds['W_TBL']*ALPHAB_URB2D)
                  
    T2_URB_DIAG = (TS_RUL * (1 - ds['FRC_URB2D']) + 
                    ((ds['R_TBL'] * (TR_URB - TC_URB) * ALPHAR_URB2D / converstion_factor + 
                      ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + 
                      ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + 
                      AH) / CHS_URB2D + TA_URB) * ds['FRC_URB2D']) - \
                    ((Q_RUL * (1 - ds['FRC_URB2D']) + 
                      (ds['R_TBL'] * (TR_URB - TC_URB) * ALPHAR_URB2D / converstion_factor + 
                      ds['W_TBL'] * (TB_URB - TC_URB) * ALPHAB_URB2D / converstion_factor + 
                      ds['RW_TBL'] * (TG_URB - TC_URB) * ALPHAG_URB2D / converstion_factor + 
                      AH) * ds['FRC_URB2D']) / CHS2_URB2D)    
    return T2_URB_DIAG
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
        
        baseline_tc_urb_diag_avg = baseline_filtered['T2_URB_DIAG'].mean(dim='Time')
        for i, ds in enumerate(ds_list[1:], 1):  # Starting from the second dataset in the list
            ds_filtered = ds if utype == "all" else ds.where(ds['UTYPE_URB'] == utype, drop=True)
            AH = AH_values_cases[case][i]

            results = []
            std_devs = []
            
            if case in ["Case 1", "Case 3"]:

    
                for compute_func, params in [
                    (compute_T2_URB_DIAG_TC_NOT_INCLUDED, {"AH": AH}),
                    (compute_T2_URB_DIAG_TC_NOT_INCLUDED, {"ALPHAR_URB2D": ds_filtered['ALPHAR_URB2D'], "ALPHAB_URB2D": ds_filtered['ALPHAB_URB2D'], "ALPHAG_URB2D": ds_filtered['ALPHAG_URB2D'], "ALPHAC_URB2D": ds_filtered['ALPHAC_URB2D'],  "CHS_URB2D": ds_filtered['CHS_URB2D']}),
                    (compute_T2_URB_DIAG_TC_NOT_INCLUDED, {"Q_RUL": ds_filtered['Q_RUL'], "TS_RUL": ds_filtered['TS_RUL'], "TR_URB": ds_filtered['TR_URB'], "TB_URB": ds_filtered['TB_URB'], "TG_URB": ds_filtered['TG_URB']}),
                    (compute_T2_URB_DIAG_TC_NOT_INCLUDED, {"TA_URB": ds_filtered['TA_URB']})
                ]:
                    diff_time_avg = (compute_func(baseline_filtered, **params).mean(dim='Time') - baseline_tc_urb_diag_avg) / AH_value
                    spatial_avg_diff = diff_time_avg.mean(['south_north', 'west_east']).values
                    spatial_std_dev = diff_time_avg.std(['south_north', 'west_east']).values
    
                    results.append(spatial_avg_diff)
                    std_devs.append(spatial_std_dev)
                
            else: 
                
                for compute_func, params in [
                    (compute_T2_URB_DIAG_TC_NOT_INCLUDED_rtc, {"AH": AH}),
                    (compute_T2_URB_DIAG_TC_NOT_INCLUDED_rtc, {"ALPHAR_URB2D": ds_filtered['ALPHAR_URB2D'], "ALPHAB_URB2D": ds_filtered['ALPHAB_URB2D'], "ALPHAG_URB2D": ds_filtered['ALPHAG_URB2D'], "ALPHAC_URB2D": ds_filtered['ALPHAC_URB2D'],  "CHS_URB2D": ds_filtered['CHS_URB2D']}),
                    (compute_T2_URB_DIAG_TC_NOT_INCLUDED_rtc, {"Q_RUL": ds_filtered['Q_RUL'], "TS_RUL": ds_filtered['TS_RUL'], "TR_URB": ds_filtered['TR_URB'], "TB_URB": ds_filtered['TB_URB'], "TG_URB": ds_filtered['TG_URB']}),
                    (compute_T2_URB_DIAG_TC_NOT_INCLUDED_rtc, {"TA_URB": ds_filtered['TA_URB']})
                ]:
                    diff_time_avg = (compute_func(baseline_filtered, **params).mean(dim='Time') - baseline_tc_urb_diag_avg) / AH_value
                    spatial_avg_diff = diff_time_avg.mean(['south_north', 'west_east']).values
                    spatial_std_dev = diff_time_avg.std(['south_north', 'west_east']).values
    
                    results.append(spatial_avg_diff)
                    std_devs.append(spatial_std_dev)       
                    

            overall_diff_time_avg = ds_filtered['T2_URB_DIAG'].mean(dim='Time') 
            overall_spatial_avg_diff = ((overall_diff_time_avg - baseline_tc_urb_diag_avg) / AH_value).mean(['south_north', 'west_east']).values
            overall_spatial_std_dev = ((overall_diff_time_avg - baseline_tc_urb_diag_avg) / AH_value).std(['south_north', 'west_east']).values
            
            cumulative_sum = sum(results)
            cumulative_sum_std_dev = np.sqrt(sum([std_dev**2 for std_dev in std_devs]))

            decomposed_values[(case, utype)] = [overall_spatial_avg_diff, cumulative_sum] + results
            decomposed_std_dev[(case, utype)] = [overall_spatial_std_dev, cumulative_sum_std_dev] + std_devs                       
#%%
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()  # Flattening the 2D axes array into 1D for easier iteration

labels = ["Direct", "Sum", "Baseline", "r", "T", "T$_A$"]
colors = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'black']  # Modify this to your desired colors

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
    print(values)
    ax.bar(labels, values, color=colors, yerr=std_devs, capsize=5)
    full_title = f"{subplot_labels[i]} {case}"
    ax.set_title(full_title, fontsize=title_font_size)    
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_label_font_size)
    if AH_option_to_study == 2 : 
        ax.set_ylim([-0.01, 0.04])  # Adjust as needed based on your data
    elif AH_option_to_study == 5 :
        ax.set_ylim([-0.01, 0.04])  # Adjust as needed based on your data
    elif AH_option_to_study == 3 :
        ax.set_ylim([-0.01, 0.04])  # Adjust as needed based on your data
            
    ax.set_ylabel('dT$_{2}$/dQ$_{AH}$ (K/(W m$^{-2}$))', fontsize=axis_label_font_size)

    # Adjusting the font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

plt.tight_layout()
plt.show()

if AH_option_to_study == 2 : 
    fig.savefig(f"T2_figures/T2_decomposition_all_4cases_AH100_AH_option_2_utype_{utype}.png", dpi=300)
elif AH_option_to_study == 5 :
    fig.savefig(f"T2_figures/T2_decomposition_all_4cases_AH100_AH_option_5_utype_{utype}.png", dpi=300)
elif AH_option_to_study == 3 :
    fig.savefig(f"T2_figures/T2_decomposition_all_4cases_AH100_AH_option_3_utype_{utype}.png", dpi=300)

            
#%% Setup the plot

# Set colormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable



def create_colormap():
    # Calculate the proportion of the colorbar that should be blue
    negative_proportion = 0.002 / (0.02 + 0.002)
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


fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Adjust the size as needed
axs = axs.flatten()  # Flatten to easily index them in a loop

case_titles = ["Case 1", "Case 2", "Case 3", "Case 4"]
colorbar_ticks = np.arange(-0.002, 0.021, 0.002)
font_size = 10  # Example font size, adjust as necessary

# Plot TA_URB for the first folder of all 4 cases, where UTYPE_URB != 0
for i, case in enumerate(case_titles):
    ds1 = datasets[case][0]  # Access the first dataset for each case
    ds2 = datasets[case][1]  # Access the second dataset for each case
    
    # Apply mask where UTYPE_URB != 0 for both datasets
    mask1 = ds1['UTYPE_URB'] != 0
    mask2 = ds2['UTYPE_URB'] != 0

    # Ensure TA_URB is only considered where UTYPE_URB != 0, then calculate the difference
    ta_urb_diff = ((ds2['TA_URB'].where(mask2)).mean(dim='Time') - (ds1['TA_URB'].where(mask1)).mean(dim='Time'))/AH_value
    
    ax = axs[i]
    # Use the custom colormap with the adjusted scale
    norm = matplotlib.colors.Normalize(vmin=-0.002, vmax=0.02)
    im = ta_urb_diff.plot(ax=ax, cmap=custom_cmap, norm=norm, add_colorbar=False)

    # Create a separate colorbar for each subplot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=colorbar_ticks)
    cbar.ax.set_yticklabels(['{:.3f}'.format(tick) for tick in colorbar_ticks], fontsize=font_size)

    ax.set_title(case)
    ax.set_xlabel("X-axis label")  # Adjust as necessary
    ax.set_ylabel("Y-axis label")  # Adjust as necessary
    # Adjust tick label sizes if needed
    ax.tick_params(axis='both', labelsize=font_size)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Adjust the size as needed
axs = axs.flatten()  # Flatten to easily index them in a loop

case_titles = ["Case 1", "Case 2", "Case 3", "Case 4"]
colorbar_ticks = np.arange(0.2, 0.91, 0.1)  # Note the 0.9 as stop value to include 0.8 in the output
font_size = 10  # Example font size, adjust as necessary

# Plot TA_URB for the first folder of all 4 cases, where UTYPE_URB != 0
for i, case in enumerate(case_titles):
    ds1 = datasets[case][0]  # Access the first dataset for each case
    ds2 = datasets[case][1]  # Access the second dataset for each case
    
    # Apply mask where UTYPE_URB != 0 for both datasets
    mask1 = ds1['UTYPE_URB'] != 0
    mask2 = ds2['UTYPE_URB'] != 0

    # Ensure TA_URB is only considered where UTYPE_URB != 0, then calculate the difference
    ta_urb_diff = ((ds2['HFX'].where(mask2)).mean(dim='Time') - (ds1['HFX'].where(mask1)).mean(dim='Time'))/AH_value
    
    ax = axs[i]
    # Use the custom colormap with the adjusted scale
    norm = matplotlib.colors.Normalize(vmin=0.2, vmax=0.9)
    im = ta_urb_diff.plot(ax=ax, cmap='Reds', norm = norm, add_colorbar=False)

    # Create a separate colorbar for each subplot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=colorbar_ticks)
    cbar.ax.set_yticklabels(['{:.1f}'.format(tick) for tick in colorbar_ticks], fontsize=font_size)

    ax.set_title(case)
    ax.set_xlabel("X-axis label")  # Adjust as necessary
    ax.set_ylabel("Y-axis label")  # Adjust as necessary
    # Adjust tick label sizes if needed
    ax.tick_params(axis='both', labelsize=font_size)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

# Calculate mean differences for each case
mean_differences = []
for case in case_titles:
    ds1 = datasets[case][0]  # First dataset for each case
    ds2 = datasets[case][1]  # Second dataset for each case
    
    # Mask where UTYPE_URB != 0
    mask1 = ds1['UTYPE_URB'] != 0
    mask2 = ds2['UTYPE_URB'] != 0
    
    # Calculate normalized difference
    normalized_diff = ((ds2['TA_URB'].where(mask2)).mean() - (ds1['TA_URB'].where(mask1)).mean())/AH_value
    mean_differences.append(normalized_diff.values)

# Create a bar plot
fig, ax = plt.subplots(figsize=(8, 6))
x_pos = np.arange(len(case_titles))
ax.bar(x_pos, mean_differences, color='skyblue')
ax.set_xticks(x_pos)
ax.set_xticklabels(case_titles)
ax.set_ylabel('Normalized Mean Difference of TA_URB')
ax.set_title('Comparison of Normalized Mean Difference of TA_URB Across Cases')
ax.tick_params(axis='x', rotation=45)  # Rotate case titles for better visibility

plt.tight_layout()
plt.show()

# Calculate mean differences for each case
mean_differences = []
for case in case_titles:
    ds1 = datasets[case][0]  # First dataset for each case
    ds2 = datasets[case][1]  # Second dataset for each case
    
    # Mask where UTYPE_URB != 0
    mask1 = ds1['UTYPE_URB'] != 0
    mask2 = ds2['UTYPE_URB'] != 0
    
    # Calculate normalized difference
    normalized_diff = ((ds2['HFX'].where(mask2)).mean() - (ds1['HFX'].where(mask1)).mean())/AH_value
    mean_differences.append(normalized_diff.values)

# Create a bar plot
fig, ax = plt.subplots(figsize=(8, 6))
x_pos = np.arange(len(case_titles))
ax.bar(x_pos, mean_differences, color='skyblue')
ax.set_xticks(x_pos)
ax.set_xticklabels(case_titles)
ax.set_ylabel('Normalized Mean Difference of HFX')
ax.set_title('Comparison of Normalized Mean Difference of HFX Across Cases')
ax.tick_params(axis='x', rotation=45)  # Rotate case titles for better visibility

plt.tight_layout()
plt.show()
