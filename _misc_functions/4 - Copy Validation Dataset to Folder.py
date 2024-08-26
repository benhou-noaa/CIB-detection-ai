# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:43:56 2022

@author: benhou
"""

import os
import csv
import shutil

current_dir = os.path.abspath(r"D:\beluga\Data")
validation_spectrogram_folder = os.path.join(current_dir, "Validation_Spectrograms")

for i in [output_test_folder]:

    if not os.path.exists(i):
        os.makedirs(i)

list_of_filenames = ['validation_filenames_BDB.csv', 
                     'validation_filenames_BS1K.csv',
                     'validation_filenames_FDB.csv',
                     'validation_filenames_FML237.csv',
                     'validation_filenames_FS1K.csv',
                     'validation_filenames_N.csv']


for i in list_of_filenames:
    print('copying list %s' % i)
    filepath = os.path.join(current_dir, "Output_Spectrogram_Vector", i)
    with open(filepath, "r") as file:
        file_reader = csv.reader(file, delimiter = ' ')
        for i in file_reader:
            file_path = (i[0])
            shutil.copy(file_path, validation_spectrogram_folder)
print('done copying files')