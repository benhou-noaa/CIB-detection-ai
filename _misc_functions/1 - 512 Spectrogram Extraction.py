# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:27:47 2022

@author: mml
"""

#
# extract_spectrograms.py
#
# Load detection labels, extract audio for detection and non-detection regions,
# compute and save spectrograms.
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

#%% Imports

import pandas as pd
from datetime import datetime, timedelta
import glob
import os
import wave
import pylab
from matplotlib import pyplot
from joblib import Parallel, delayed
import multiprocessing
import gc
import random
import pathlib
import numpy as np
import traceback

#%% Step 1: import the labels

#working on split drives; one drive has all of the data on it, the other is internal 
#this is where all of the big stuff lives; this ipynb can live anywhere...
current_dir = os.path.abspath("Z:\processing")


data_dir = os.path.join(current_dir, "Data")
labeled_data_dir = os.path.join(data_dir, "Labeled_Data")
audio_dir = os.path.join(data_dir,"Raw_Audio")
output_spectrogram_dir = os.path.join(data_dir, "Extracted_Spectrogram_512")

for i in [data_dir, labeled_data_dir, audio_dir, output_spectrogram_dir]:
    if not os.path.exists(i):
        os.makedirs(i)
        
if not os.path.exists(output_spectrogram_dir):
    os.makedirs(output_spectrogram_dir)
    
    

#detector_labelled_data = pd.read_excel (labeled_data_dir + '_PG_WandM_Detector.xlsx')[['UTC', 'Species']].drop_duplicates()
detector_labelled_data = pd.read_csv(os.path.join(labeled_data_dir,"updated_combined_training_dataset.csv")).drop_duplicates()
detector_labelled_data.shape
detector_labelled_data.dropna(inplace=True)
detector_labelled_data['UTC'] =  detector_labelled_data['UTC'].astype('datetime64[s]')
detector_labelled_data = detector_labelled_data.drop_duplicates()
detector_labelled_data['Detection_TimeStamp'] = detector_labelled_data['UTC'].dt.strftime('%Y%m%d%H%M%S')
detector_labelled_data['Date'] = detector_labelled_data['UTC'].dt.strftime('%Y%m%d')

detector_labelled_data.Date.value_counts().sort_index()
print(detector_labelled_data.shape)

audio_filenames = pathlib.Path(os.path.join(audio_dir)).rglob("*.wav")
audio_filenames = [os.path.abspath(filename) for filename in audio_filenames]
audio_filenames_df = pd.DataFrame(audio_filenames, columns = ['audio_filepath'])

audio_filenames_df['audio_filename'] = audio_filenames_df['audio_filepath'].apply(lambda x: os.path.basename(x))
audio_filenames_df['deployment'] = audio_filenames_df['audio_filepath'].apply(lambda x: pathlib.Path(x).parts[4].replace("D",""))
audio_filenames_df.head()
#%% match wave
"""
Some wave files are named based on id.date.time instead of id.datetime. 
To address this, I calculate the number of periods in the filename.
If there are 3 periods, we combine the second and 3rd values
If only 2, we use the original MS logic
But first, we calculate the number of periods in each filename
"""
audio_filenames_df['count'] = audio_filenames_df['audio_filename'].str.count("\.")

def construct_audio_timestamp(df):
    if df['count'] == 3.0:
        df['audio_start_TimeStamp'] = '20' + df['audio_filename'].split(".")[1] + df['audio_filename'].split(".")[2]
    
    elif df['count'] == 2.0:
        df['audio_start_TimeStamp'] = '20' + df['audio_filename'].split(".")[1]
        
    else:
        print("issue!")
    return df

#then we apply the function
audio_filenames_df = audio_filenames_df.apply(construct_audio_timestamp,axis=1)

#Some files have a "_+20dB" in the name. we need to remove this from the timestamp because it is not a number.
audio_filenames_df['audio_start_TimeStamp'] = audio_filenames_df['audio_start_TimeStamp'].str.replace("_+20dB","", regex=False)

audio_filenames_df['audio_end_TimeStamp'] = ''


#
detector_labelled_data['Detection_TimeStamp'] = detector_labelled_data['Detection_TimeStamp'].astype(np.uint64)

#we also need the deployment ID e.g. 201


for index, row in audio_filenames_df.iterrows():
    audio_start_TimeStamp = row['audio_start_TimeStamp']
    audio_end_time = datetime(int(audio_start_TimeStamp[0:4]), 
                              int(audio_start_TimeStamp[4:6]), 
                              int(audio_start_TimeStamp[6:8]), 
                              int(audio_start_TimeStamp[8:10]), 
                              int(audio_start_TimeStamp[10:12]),
                              int(audio_start_TimeStamp[12:14])) + timedelta(minutes = 5) 
    audio_end_TimeStamp = audio_end_time.strftime('%Y') + audio_end_time.strftime('%m') + audio_end_time.strftime('%d') + audio_end_time.strftime('%H')  + audio_end_time.strftime('%M') + audio_end_time.strftime('%S')
    audio_filenames_df.at[index,'audio_end_TimeStamp'] = audio_end_TimeStamp
    
audio_filenames_df['Date'] = audio_filenames_df['audio_start_TimeStamp'].str[:8]
audio_filenames_df.Date.value_counts().sort_index()

# Transform to dictionary with format {audio_filename: ['audio_start_TimeStamp', 'audio_end_TimeStamp', 'audio_start_date']}
audio_filenames_dict = audio_filenames_df.set_index('audio_filename').T.to_dict('list')

#audio_filenames_dict

detector_labelled_data.reset_index(inplace=True)

detector_labelled_data_length = len(detector_labelled_data)

detector_labelled_data['audio_filename'] = ''

report_interval = 500
for index, row in detector_labelled_data.iterrows():
    Detection_TimeStamp = row['Detection_TimeStamp']
    Detection_Deployment = int(row['Deployment'])
    
    
    matched_audio_filename = [k for k, v in audio_filenames_dict.items() if int(v[3]) <= int(Detection_TimeStamp) < int(v[4]) and int(v[1])==Detection_Deployment]
    if len(matched_audio_filename) == 0:
        detector_labelled_data.at[index,'audio_filename'] = 'No Matched Audio File'
        message = "0"
    elif len(matched_audio_filename) == 1:      
        detector_labelled_data.at[index,'audio_filename'] = matched_audio_filename[0]
        message = "1"
    elif len(matched_audio_filename) >=2:      
        detector_labelled_data.at[index,'audio_filename'] = str(matched_audio_filename) + "Multiple Matched Audio Files"
        message = "1+"
    
    if index % report_interval == 0:
        print("matching detections to waves...", index, "/",detector_labelled_data_length, " ", message)

print(detector_labelled_data.audio_filename.value_counts())

detector_labelled_data['Species_Type'] = detector_labelled_data['Species']

try:
    detector_labelled_data.drop(['Unnamed: 0'], axis=1, inplace=True)
except:pass

try:
        detector_labelled_data.drop(['index'], axis=1, inplace=True)
except: pass

detector_labelled_data['Source'].unique()


source_species_dict = {'AB_ML_237_VALIDATION':"ML237", 'Sub1Khz':"S1K", 'DosDB':"DB"}
detector_labelled_data['Species_Type'] = detector_labelled_data['Source'].map(source_species_dict)

detector_labelled_data['Species_Type'] = detector_labelled_data['Species'] + "-" + detector_labelled_data['Species_Type']

#%% prep for spectrogram 
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def graph_spectrogram(wav_file, serialnumber, audio_begin_TimeStamp, start_second, Species):
    sound_info, frame_rate = get_wav_info(wav_file)
    
    file_path_name = os.path.join(output_spectrogram_dir, serialnumber + '.' + audio_begin_TimeStamp + '_' + str(start_second)  + '_' + Species + '.png')
    
    if os.path.exists(file_path_name):
        print('file_path_name exists, skipping:', file_path_name)
    else:
        pyplot.figure(num=None, figsize=(19, 12))
        pyplot.subplot(222)
        ax = pyplot.axes()
        ax.set_axis_off()
        pyplot.specgram(sound_info[frame_rate * start_second: frame_rate * (start_second+2)], Fs = frame_rate, NFFT=512)
        pyplot.savefig(file_path_name, 
                       bbox_inches='tight', 
                       transparent=True, 
                       pad_inches=0.0)
    pyplot.close()
    gc.collect()

def generate_spectrogram_B_F(i):
    """
    To address a class imbalance (principally with sub-1khz and ML false positives),
    we previously created a column called Species Type to help identify the species (beluga or not)
    as well as the source (i.e. is a sub-1khz detection).
    
    We do this so we can boost the number of these in the training set later. 
    This is addressed in the subsequent script
    
    """

    Species = matched_detector_labelled_data_B_F.loc[i, 'Species_Type']
    deployment_folder = str(matched_detector_labelled_data_B_F.loc[i, 'Deployment'])+"D"
    audio_filename = matched_detector_labelled_data_B_F.loc[i, 'audio_filename']
    
    split_audio_filename = audio_filename.split('.')
    if len(split_audio_filename) == 4:
        serialnumber = split_audio_filename[0]
        audio_begin_TimeStamp = split_audio_filename[1]+split_audio_filename[2]
    else:
        serialnumber, audio_begin_TimeStamp = split_audio_filename[0:2]
    
    
    Detection_TimeStamp = str(matched_detector_labelled_data_B_F.loc[i, 'Detection_TimeStamp'])
    detection_start_timedelta = datetime(int(Detection_TimeStamp[0:4]), 
                 int(Detection_TimeStamp[4:6]), 
                 int(Detection_TimeStamp[6:8]),
                 int(Detection_TimeStamp[8:10]),
                 int(Detection_TimeStamp[10:12]),
                 int(Detection_TimeStamp[12:14])) - datetime(int('20' + audio_begin_TimeStamp[0:2]), 
                          int(audio_begin_TimeStamp[2:4]), 
                          int(audio_begin_TimeStamp[4:6]),
                          int(audio_begin_TimeStamp[6:8]), 
                          int(audio_begin_TimeStamp[8:10]),
                          int(audio_begin_TimeStamp[10:12]))
    detection_start_second = detection_start_timedelta.seconds
    wave_path = os.path.join(audio_dir,deployment_folder,audio_filename)
    return graph_spectrogram(wave_path, serialnumber, audio_begin_TimeStamp, detection_start_second, Species)

#%% Step 3: extract spectrograms from detections

matched_detector_labelled_data = detector_labelled_data.loc[(~detector_labelled_data.audio_filename.str.contains('No Matched Audio File')) & 
                                                            (~detector_labelled_data.audio_filename.str.contains('Multiple Matched Audio Files'))].reset_index(drop=True)
print(matched_detector_labelled_data.shape)

matched_detector_labelled_data_B_F = matched_detector_labelled_data.loc[(matched_detector_labelled_data.Species == 'B') | (matched_detector_labelled_data.Species == 'F')].reset_index(drop=True)
print(matched_detector_labelled_data_B_F.shape)

spectrogram_seconds_duration = 2 


num_cores = multiprocessing.cpu_count()
print('now generating spectrograms')
spectrograms_B_F = Parallel(n_jobs=num_cores)(delayed(generate_spectrogram_B_F)(i) for i in range(len(matched_detector_labelled_data_B_F)))
print('Done generating spectrograms')


#%% Step 4: extract spectrograms from non-detection audio regions

sample_size = 40673
sound_detected_audio_filenames = detector_labelled_data.loc[~detector_labelled_data.audio_filename.str.contains('No Matched Audio File')].audio_filename.unique().tolist()
nosound_detected_audio_filenames = [filename for filename in audio_filenames if filename not in sound_detected_audio_filenames]
nosound_detected_audio_filenames_sample = random.sample(nosound_detected_audio_filenames, min(len(nosound_detected_audio_filenames), sample_size))

def generate_spectrogram_N(i):
    """ 
    This function works on a list of files that are not part of the positive beluga collection. 
    Specifically, it gives the path to the files. Thus, we need to parse the filepath to get the information we need
    There is no need to reconstruct the filepath because it is given. 
    But, we still need to provide the requisite information the grapher desires. 
    """
    audio_filepath = nosound_detected_audio_filenames_sample[i]
    audio_filename = os.path.basename(audio_filepath)
    Species = 'N'
    
    split_audio_filename = audio_filename.split('.')
    
    if len(split_audio_filename) == 4:
        serialnumber = split_audio_filename[0]
        audio_begin_TimeStamp = split_audio_filename[1]+split_audio_filename[2]
    else:
        serialnumber, audio_begin_TimeStamp = split_audio_filename[0:2]
    
    # Each audio file is five minutes; sample the starting timestamp between second 0 - 299
    start_second = random.sample(range(0, 299), 1)[0]      
    output_wave_path = os.path.join(audio_dir,audio_filename)

    return graph_spectrogram(audio_filepath, serialnumber, audio_begin_TimeStamp, start_second, Species)

num_cores = multiprocessing.cpu_count()
spectrograms_N = Parallel(n_jobs=num_cores)(delayed(generate_spectrogram_N)(i) for i in range(len(nosound_detected_audio_filenames_sample)))

