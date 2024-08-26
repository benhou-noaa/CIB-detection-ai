#
# full_analysis_scoring_for_new_dataset.py
#
# Run trained models on a new data set for which spectrograms have already
# been generated.
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

#%% Imports

import pandas as pd
import numpy as np
import glob
import os
import cv2
from keras.models import model_from_json
#%% noaa additions
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import os
import traceback
import pathlib
import datetime

#%% dynamic allocation
"""
Step 1 : Enable Dynamic Memory Allocation

In Jupyter Notebook, restart the kernel (Kernel -> Restart). 
The previous model remains in the memory until the Kernel is restarted, so rerunning the Notebook cells without restarting the kernel may lead to a false Out Of Memory error.

By default, Tensorflow statically allocates the memory in the GPU for the model. 
Larger models can be built when allowing Tensorflow to dynamically allocate the memory.

To enable dynamic memory allocation, run the following commands at the start of the session :
https://www.linkedin.com/pulse/solving-out-memory-oom-errors-keras-tensorflow-running-wayne-cheng
"""

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = InteractiveSession(config=config)
tf.compat.v1.disable_eager_execution()


#%% Path configuration

current_dir = os.path.abspath("Z:\\Processing")
#current_dir = os.path.abspath("Y:\\NMML_CAEP_Acoustics\\DOS_CIBAS\\_Automated_Processing\\2022_retraining")
data_dir = os.path.join(current_dir, "Data")
spectrogram_dir = os.path.join(data_dir,"Validation_Spectrograms")
model_dir = os.path.join(current_dir,"Model")
output_dir = os.path.join(current_dir, "Output")

for i in [output_dir]:

    if not os.path.exists(i):
        os.makedirs(i)

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

#%% Enumerate spectrograms to score
    
spectrogram_filenames = glob.glob(os.path.join(spectrogram_dir, '*.png'))
print("Total number of Spectrograms: ", len(spectrogram_filenames))


#%% Load models

"""
with open(os.path.join(model_dir, "-".join([model_name,'cnn_architecture_all_data.json'])), 'r') as f:
    model_cnn = model_from_json(f.read())
model_cnn.load_weights(os.path.join(model_dir,"-".join([model_name, 'cnn_weights_all_data.h5'])))

with open(os.path.join(model_dir, "-".join([model_name,'vgg16_architecture_all_data.json'])), 'r') as f:
    model_vgg16 = model_from_json(f.read())
model_vgg16.load_weights(os.path.join(model_dir, "-".join([model_name,'vgg16_weights_all_data.h5'])))

with open(os.path.join(model_dir, "-".join([model_name,'ResNet50_architecture_all_data.json'])), 'r') as f:
    model_ResNet50 = model_from_json(f.read())
model_ResNet50.load_weights(os.path.join(model_dir,"-".join([model_name,'ResNet50_weights_all_data.h5'])))
"""
with open(os.path.join(model_dir, "-".join([model_name,'DenseNet121_architecture_all_data.json'])), 'r') as f:
    model_DenseNet121 = model_from_json(f.read())
model_DenseNet121.load_weights(os.path.join(model_dir,"-".join([model_name,'DenseNet121_weights_all_data.h5'])))
print("loaded models")
#%% Run models on spectrograms

ncol, nrow = 300, 300

full_analysis_score = pd.DataFrame()
full_analysis_score['spectrogram_filename'] = spectrogram_filenames
full_analysis_score['audio_filename'] = ''

full_analysis_score['spectrogram_start_second'] = ''
full_analysis_score['predicted_probability'] = 0.0

opt_weights = pd.read_excel(os.path.join(output_dir, 
                                         "-".join([model_name, 'opt_weights.xlsx'])), 
                                         header = None)[0].values.tolist()

full_analysis_score = pd.DataFrame()
full_analysis_score['spectrogram_filename'] = spectrogram_filenames
full_analysis_score['audio_filename'] = ''
full_analysis_score['spectrogram_start_second'] = ''
full_analysis_score['predicted_probability'] = 0.0

opt_weights = pd.read_excel(os.path.join(output_dir,"-".join([model_name,'opt_weights.xlsx'])), header = None)[0].values.tolist()

#%% score data
for index, row in full_analysis_score.iterrows():
    if (index % 100 == 0):
        print("scoring {i} of {total}".format(i = index, total = len(full_analysis_score)))
    audio_filename = "_".join((pathlib.Path(row['spectrogram_filename']).stem.split("_")[:-2]))+".wav"
    spectrogram_start_second = pathlib.Path(row['spectrogram_filename']).stem.split("_")[-2]
    img = cv2.imread(row['spectrogram_filename'])
    img = cv2.resize(img, (ncol, nrow))
    img_reshaped = []
    img_reshaped.append(img)
    #predict_prob_cnn = model_cnn.predict(np.asarray(img_reshaped) / 255.0).tolist()[0][0]
    #predict_prob_vgg16 = model_vgg16.predict(np.asarray(img_reshaped) / 255.0).tolist()[0][0]
    #predict_prob_ResNet50 = model_ResNet50.predict(np.asarray(img_reshaped) / 255.0).tolist()[0][0]
    predict_prob_DenseNet121 = model_DenseNet121.predict(np.asarray(img_reshaped) / 255.0).tolist()[0][0]
    ## the opmized weight for each model was computed in previous step
    predicted_probability = predict_prob_DenseNet121 #sum([x*y for x,y in zip([predict_prob_cnn, predict_prob_vgg16, predict_prob_ResNet50, predict_prob_DenseNet121], opt_weights)])
    
    full_analysis_score.at[index, 'audio_filename'] = audio_filename
    full_analysis_score.at[index, 'spectrogram_start_second'] = spectrogram_start_second
    full_analysis_score.at[index, 'predicted_probability_densenet'] = predicted_probability
print('Done scoring!')
#%% save data
full_analysis_score.to_excel(os.path.join(output_dir,'full_analysis_ouptut_predicted_scores.xlsx'), index=False)
print('saved!')
