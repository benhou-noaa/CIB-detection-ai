#
# extract_spectrograms_for_new_dataset.py
#
# Generate spectrograms for a data set on which we want to run the models trained
# in steps 1-4.
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

#%% Imports

import glob
import os
import wave
import pylab
from matplotlib import pyplot
from joblib import Parallel, delayed  
import multiprocessing
import gc

##%% noaa imports
import datetime
import pandas as pd
import numpy as np
import zlib
import pathlib
import re

##%% Path configuration

current_dir = os.path.abspath("Z:\processing")
#current_dir = os.path.abspath("Y:\\NMML_CAEP_Acoustics\\DOS_CIBAS\\_Automated_Processing\\2022_retraining")
data_dir = os.path.join(current_dir, "Data")
spectrogram_manifest_to_score_dir = os.path.join(data_dir, "spectrogram_manifest_to_score_dir")

effort_dir = os.path.join(data_dir, "Effort")




#we need to specify the "training set name" we want to work with
#e.g. BS1KHZ-n721-x1_BDB-n29859-x1
#for replicability's sake, we will store this in a dict

b_positive_variant_dict = {1:"BS1KHZ-n721-x1_BDB-n29859-x1"}
b_false_variant_dict = {1: "FML237-n157-x1_FS1KHZ-n76-x1_FDB-n9860-x1"}
b_negative_variant_dict = {1: "N-n40673-x1"}

#here you can change which variant you are using
positive_variant = 1
false_variant = 1
negative_variant = 1
model_name = "B{b_var}-F{f_var}-N{n_var}".format(b_var = str(positive_variant), f_var = str(false_variant), n_var = str(negative_variant))
print(model_name)

##%% input job name

deployment = input('Which deployment are you processing?:\n').replace(" ","")
deployment = "".join(char for char in deployment if char.isdigit())

print("You are processing %s" % deployment)
output_spectrogram_dir = os.path.join(data_dir, "Extracted_Spectrogram_Full_Analysis", deployment)
for i in [output_spectrogram_dir, spectrogram_manifest_to_score_dir]:

    if not os.path.exists(i):
        os.makedirs(i)
        print('making directory: ',i)
    
##%% get effort table

effort_table = pd.read_csv(os.path.join(effort_dir, "CIBA_3_Effort.csv")).set_index(('Deploy ID'))
#effort_table['Effort Start file (UTC)'] = effort_table['Effort Start file (UTC)'].astype('int64')
#effort_table['Effort End file (UTC)'] = effort_table['Effort End file (UTC)'].astype('int64')
current_deployment = effort_table.loc[int(deployment)]

audio_dir = current_deployment[6]#os.path.join(data_dir, "Raw_Audio_Full_Analysis")


effort_start = int("".join(["20", str(int(current_deployment[2]))]))
effort_end = int("".join(["20", str(int(current_deployment[4]))]))

effort_start = pd.to_datetime(effort_start, format=("%Y%m%d%H%M%S"))#.to_datetime64()
effort_end = pd.to_datetime(effort_end, format=("%Y%m%d%H%M%S"))#.to_datetime64()
print(current_deployment)


##%% functions
def get_filename(df, label, output_label_name, extension):
    df[output_label_name] = "".join([os.path.basename(df[label])])
    return df

def get_wave_time(df):
    df['recording_start_time_abs'] = "".join(["20", df['wave_filename'].split(".")[1]])
    return df

##%% Get waves and filter before generation
    
audio_filenames = glob.glob(audio_dir + '/*.wav')
print("Total number of New Audio Files to Generate Spectrograms:", len(audio_filenames))

audio_filename_df  = pd.DataFrame(audio_filenames, columns=(['filepath']))
print("constructing df")


#audio_filename_df = audio_filename_df.apply(get_filepath_datetime, axis=1)
audio_filename_df = audio_filename_df.apply(get_filename, axis=1, args = ('filepath', 'wave_filename', '.wav'))
#audio_filename_df['recording_start_time_abs'] = audio_filename_df['wave_filename'].split(

audio_filename_df = audio_filename_df.apply(get_wave_time, axis=1)
audio_filename_df['recording_start_time_abs'] = pd.to_datetime(audio_filename_df['recording_start_time_abs'], format=("%Y%m%d%H%M%S"))

audio_filename_df = audio_filename_df[(audio_filename_df['recording_start_time_abs']>= effort_start) & (audio_filename_df['recording_start_time_abs']<= effort_end)]

def get_wav_info(audio_filename):
    wav = wave.open(audio_filename, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def graph_spectrogram(spectrogram_second_length, audio_filename):
    sound_info, frame_rate = get_wav_info(audio_filename)
    audio_length_second = int(len(sound_info) / frame_rate)
    file_basename = os.path.basename(audio_filename)
    
    for j in range(0, audio_length_second, spectrogram_second_length):
        
        output_file_name = os.path.join(output_spectrogram_dir, "_".join([file_basename.replace('.wav',""),str(j), str(j + spectrogram_second_length)])+'.png')
        
        if os.path.exists(output_file_name):
            pass
        else:
            pyplot.figure(num=None, figsize=(19, 12))
            pyplot.subplot(222)
            ax = pyplot.axes()
            ax.set_axis_off()
            pyplot.specgram(sound_info[frame_rate * j: frame_rate * (j + spectrogram_second_length)], Fs = frame_rate, NFFT=256)
            
            
            print(output_file_name)
    
            pyplot.savefig(output_file_name, bbox_inches='tight', transparent=True, pad_inches=0.0)
                
            pyplot.close()
            #print('generated %s' % os.path.join(output_spectrogram_dir, 
            #                            "-".join([file_basename,"_".join([file_basename.replace('.wav',""),str(j), str(j + spectrogram_second_length)])])+'.png'))
    gc.collect()

def generate_spectrograms(i):
    audio_filename = audio_filenames[i]
    try:
        return graph_spectrogram(2, audio_filename)
    except:
        pass
    
print('Ready to Generate!')
    
##%% Generate Spectrograms
print('Generating Spectrograms')
num_cores = multiprocessing.cpu_count()
num_cores_to_use = int(num_cores)
spectrograms = Parallel(n_jobs=num_cores_to_use)(delayed(generate_spectrograms)(i) for i in range(len(audio_filenames)))
print('Done Generating Spectrograms')

##%% #Get spectrogram DF

def construct_wave_filename_from_spectrogram_filename(df):
    
    file_no_extension = pathlib.Path(df['spectrogram_file_path']).stem.split("_")[:-2]


    df['wave_filename'] = "_".join(pathlib.Path(df['spectrogram_file_path']).stem.split("_")[:-2])+".wav"
                                   
    return df

def lookup_filename(df, lookup_df):
    df['wave_file_path'] = lookup_df['']

spectrogram_filenames = glob.glob(os.path.join(output_spectrogram_dir, "*.png"))

spectrogram_filename_df = pd.DataFrame(spectrogram_filenames, columns=['spectrogram_file_path'])
spectrogram_filename_df['wave_filename'] = ""
spectrogram_filename_df = spectrogram_filename_df.apply(construct_wave_filename_from_spectrogram_filename, axis=1)

spectrogram_filename_df
spectrogram_filename_df = pd.merge(left=spectrogram_filename_df, right= audio_filename_df, how='left')

##%% get absolute time and save data
def get_filepath_datetime(df):
    
    spectrogram_timing_list = pathlib.Path(df['spectrogram_file_path']).stem.split(".")[-1].replace("_+20dB","").split("_")
    
    df['time_wave_recording_start_abs'] = '20'+spectrogram_timing_list[0]
    df['time_spectrogram_start_relative'] = spectrogram_timing_list[1]
    df['time_spectrogram_end_relative'] = spectrogram_timing_list[2]
    
    return df

#audio_filename_df['recording_start_time_abs'] = audio_filename_df['recording_start_time_abs'].astype(np.uint64)

spectrogram_filename_df = spectrogram_filename_df.apply(get_filepath_datetime, axis=1)
spectrogram_filename_df['time_spectrogram_start_relative'] = spectrogram_filename_df['time_spectrogram_start_relative'].astype(np.uint64) 
spectrogram_filename_df['time_spectrogram_end_relative'] = spectrogram_filename_df['time_spectrogram_end_relative'].astype(np.uint64) 

spectrogram_filename_df['time_wave_recording_start_abs'] = pd.to_datetime(spectrogram_filename_df['time_wave_recording_start_abs'].astype(np.uint64), 
                                                                          format="%Y%m%d%H%M%S")

spectrogram_filename_df['time_spectrogram_start_relative'] = pd.to_timedelta(spectrogram_filename_df['time_spectrogram_start_relative'], unit="S")
spectrogram_filename_df['time_spectrogram_end_relative'] = pd.to_timedelta(spectrogram_filename_df['time_spectrogram_end_relative'], unit="S")

#spectrogram_filename_df['time_spectrogram_start_relative'] = pd.to_timedelta(spectrogram_filename_df['time_spectrogram_start_relative'], unit="S").dt.total_seconds()
#spectrogram_filename_df['time_spectrogram_end_relative'] = pd.to_timedelta(spectrogram_filename_df['time_spectrogram_end_relative'], unit="S").dt.total_seconds()

spectrogram_filename_df['time_spectrogram_start_absolute'] = spectrogram_filename_df['time_wave_recording_start_abs'] + spectrogram_filename_df['time_spectrogram_start_relative']
spectrogram_filename_df['time_spectrogram_end_absolute'] = spectrogram_filename_df['time_wave_recording_start_abs'] + spectrogram_filename_df['time_spectrogram_end_relative']


output_file_name = "_".join([datetime.datetime.now().strftime("%Y-%m-%d-%H%M"),str(deployment)+".csv"])
print('exporting', output_file_name)
spectrogram_filename_df.to_csv(os.path.join(spectrogram_manifest_to_score_dir,output_file_name), index=False)