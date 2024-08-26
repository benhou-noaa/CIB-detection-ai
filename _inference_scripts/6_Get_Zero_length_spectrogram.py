# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:10:19 2022

@author: mml
"""
##%% Imports

import pandas as pd
import numpy as np
import glob
import os

import shutil


##%%
output_bad_dir = r"Z:\processing\Data\Extracted_Spectrogram_Full_Analysis"


##%% get file sizes
def get_filesize(df):
    try:
        df['file_size_kb'] = os.path.getsize(df['spectrogram_file_path'])/1000
    except:pass
    return df

def remove_broken_spectrogram():
    input_path = input("paste path to file:").replace('"','')
    
    deployment = os.path.splitext(input_path)[0].split("_")[-1]
    output_dir = os.path.join(output_bad_dir, "_".join([str(deployment),"bad_spectrograms"]))
    
    for i in [output_dir]:
        if not os.path.exists(i):
            os.makedirs(i)

    
    #get df
    input_df = pd.read_csv(input_path)
                                     
    #get file sizes            
    print('calculating file sizes')        
    input_df = input_df.apply(get_filesize, axis=1)
    
    #filter to just small files
    print('filtering to just files under 50KB')
    input_df = input_df[(input_df["file_size_kb"]<=50)]
    
    for index, row in input_df.iterrows():
        print('moving ',row['spectrogram_file_path'])
        shutil.move(row['spectrogram_file_path'], os.path.join(output_dir, os.path.basename(row['spectrogram_file_path'])))

remove_broken_spectrogram()