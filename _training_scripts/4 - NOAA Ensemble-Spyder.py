#
# model_ensemble.py
#
# Train the ensemble model
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

#%% imports
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Keras imports
from keras import models
from keras.models import model_from_json
from scipy import optimize
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

#%% noaa additions
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import os
import traceback


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

current_dir = os.path.abspath("D:\\beluga")
#current_dir = os.path.abspath("Y:\\NMML_CAEP_Acoustics\\DOS_CIBAS\\_Automated_Processing\\2022_retraining")
data_dir = os.path.join(current_dir, "Data")

model_dir = os.path.join(current_dir,"Model")
spectrogram_dir = os.path.join(data_dir,"Extracted_Spectrogram")
output_spectrogram_vector_dir = os.path.join(data_dir, "Output_Spectrogram_Vector")
output_dir = os.path.join(current_dir, "Output")

for i in [model_dir, output_spectrogram_vector_dir, output_dir]:

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


#%% Step 1: train/validation/test split

ncol, nrow = 300, 300
spectrogram_suffix = "spectrograms_sample_300_300.npy"
csv_suffix = "filenames_sample.csv"

beluga_positive_basepath = os.path.join(output_spectrogram_vector_dir, "_".join(["beluga-positive", b_positive_variant_dict[positive_variant]]))
beluga_false_basepath = os.path.join(output_spectrogram_vector_dir, "_".join(["beluga-false", b_false_variant_dict[false_variant]]))
beluga_negative_basepath = os.path.join(output_spectrogram_vector_dir, os.path.join("_".join(["beluga-negative", b_negative_variant_dict[negative_variant]])))

#print(beluga_positive_basepath, beluga_false_basepath, beluga_negative_basepath)

#get the numpy arrays
print("loading beluga")
spectrograms_B_sample = np.load("-".join([beluga_positive_basepath,spectrogram_suffix]))
print("loading false beluga")
spectrograms_F_sample = np.load("-".join([beluga_false_basepath,spectrogram_suffix]))
print("loading negative beluga")
spectrograms_N_sample = np.load("-".join([beluga_negative_basepath,spectrogram_suffix]))

#get the csv files    
filenames_B_sample = []
filenames_F_sample = []
filenames_N_sample = []


with open("-".join([beluga_positive_basepath, csv_suffix]), newline='') as f:
    for row in csv.reader(f):
        filenames_B_sample.append(row[0])

with open("-".join([beluga_false_basepath, csv_suffix]), newline='') as f:
    for row in csv.reader(f):
        filenames_F_sample.append(row[0])

with open("-".join([beluga_negative_basepath, csv_suffix]), newline='') as f:
    for row in csv.reader(f):
        filenames_N_sample.append(row[0])

spectrograms_B_train_validation, spectrograms_B_test, filenames_B_train_validation, filenames_B_test = train_test_split(spectrograms_B_sample, filenames_B_sample, test_size = 0.3, random_state = 1)
spectrograms_F_train_validation, spectrograms_F_test, filenames_F_train_validation, filenames_F_test = train_test_split(spectrograms_F_sample, filenames_F_sample, test_size = 0.3, random_state = 1)
spectrograms_N_train_validation, spectrograms_N_test, filenames_N_train_validation, filenames_N_test = train_test_split(spectrograms_N_sample, filenames_N_sample, test_size = 0.3, random_state = 1)

spectrograms_train_validation = np.concatenate((spectrograms_B_train_validation, spectrograms_F_train_validation, spectrograms_N_train_validation), axis=0)
labels_train_validation = np.array([1] * len(spectrograms_B_train_validation) + [0] * len(spectrograms_F_train_validation) + [0] * len(spectrograms_N_train_validation))

X_train, X_validation, y_train, y_validation = train_test_split(spectrograms_train_validation, labels_train_validation, test_size = 0.3, random_state = 1)

X_train = X_train / 255.0
X_validation = X_validation / 255.0

print(X_train.shape)   
print(X_validation.shape)   
print(spectrograms_B_test.shape)   
print(spectrograms_F_test.shape)
print(spectrograms_N_test.shape)


#%% Step 2: load models

with open(os.path.join(model_dir, "-".join([model_name, "cnn_architecture_all_data.json"])), 'r') as f:
    model_cnn = model_from_json(f.read())
model_cnn.load_weights(os.path.join(model_dir, "-".join([model_name,'cnn_weights_all_data.h5'])))

with open(os.path.join(model_dir, "-".join([model_name,'vgg16_architecture_all_data.json'])), 'r') as f:
    model_vgg16 = model_from_json(f.read())
model_vgg16.load_weights(os.path.join(model_dir, "-".join([model_name,'vgg16_weights_all_data.h5'])))

with open(os.path.join(model_dir, "-".join([model_name,'ResNet50_architecture_all_data.json'])), 'r') as f:
    model_ResNet50 = model_from_json(f.read())
model_ResNet50.load_weights(os.path.join(model_dir, "-".join([model_name,'ResNet50_weights_all_data.h5'])))


with open(os.path.join(model_dir, "-".join([model_name,'DenseNet121_architecture_all_data.json'])), 'r') as f:
    model_DenseNet121 = model_from_json(f.read())
model_DenseNet121.load_weights(os.path.join(model_dir, "-".join([model_name,'DenseNet121_weights_all_data.h5'])))


#%% Step 3: predict on the validation and test sets
validation_predict_cnn = model_cnn.predict(X_validation) 
validation_predict_cnn = [x for sublist in validation_predict_cnn.tolist() for x in sublist]

validation_predict_vgg16 = model_vgg16.predict(X_validation)
validation_predict_vgg16 = [x for sublist in validation_predict_vgg16.tolist() for x in sublist]

validation_predict_ResNet50 = model_ResNet50.predict(X_validation)
validation_predict_ResNet50 = [x for sublist in validation_predict_ResNet50.tolist() for x in sublist]

validation_predict_DenseNet121 = model_DenseNet121.predict(X_validation)
validation_predict_DenseNet121 = [x for sublist in validation_predict_DenseNet121.tolist() for x in sublist]

validation_predict = [validation_predict_cnn, validation_predict_vgg16, validation_predict_ResNet50, validation_predict_DenseNet121]

# Optimize weights for each model
def f(weights):
    validation_predict_ensemble = np.average(validation_predict, axis=0, weights=weights)
    validation_predict_ensemble_class = [int(validation_predict_ensemble[i] > 0.5) for i in range(len(validation_predict_ensemble))]
    return validation_predict_ensemble_class

def loss_function(weights):
    validation_predict_ensemble_class = f(weights)
    n_lost = [prediction != label for prediction, label in zip(validation_predict_ensemble_class, y_validation)]
    return np.sum(n_lost) / len(y_validation)

model_cnt = 4 # the number of models for ensembling

opt_weights = optimize.minimize(loss_function,
                                [1/ model_cnt] * model_cnt,
                                constraints=({'type': 'eq','fun': lambda w: 1-sum(w)}),
                                method= 'Nelder-Mead', #'SLSQP',
                                bounds=[(0.0, 1.0)] * model_cnt,
                                options = {'ftol':1e-3},
                            )['x']

print('Optimum weights = ', opt_weights, 'with loss', loss_function(opt_weights))

# Save the optimal weights of each individual model
opt_weights[-1] = 1.0 - sum(opt_weights[:-1])  ## to force the total weights sums up to 1 
pd.DataFrame(opt_weights).to_excel(os.path.join(output_dir, "-".join([model_name, 'opt_weights.xlsx'])), header=False, index=False)

del X_train
del X_validation
del spectrograms_train_validation
del spectrograms_B_train_validation
del spectrograms_F_train_validation
del spectrograms_N_train_validation
del spectrograms_B_sample
del spectrograms_F_sample
del spectrograms_N_sample

############### CNN: prediction ##############
spectrograms_B_test_predict_cnn = model_cnn.predict(spectrograms_B_test / 255.0)
spectrograms_B_test_predict_cnn = [x for sublist in spectrograms_B_test_predict_cnn.tolist() for x in sublist]

spectrograms_F_test_predict_cnn = model_cnn.predict(spectrograms_F_test / 255.0)
spectrograms_F_test_predict_cnn = [x for sublist in spectrograms_F_test_predict_cnn.tolist() for x in sublist]

spectrograms_N_test_predict_cnn = model_cnn.predict(spectrograms_N_test / 255.0)
spectrograms_N_test_predict_cnn = [x for sublist in spectrograms_N_test_predict_cnn.tolist() for x in sublist]

############### VGG16: prediction ##############
spectrograms_B_test_predict_vgg16 = model_vgg16.predict(spectrograms_B_test / 255.0)
spectrograms_B_test_predict_vgg16 = [x for sublist in spectrograms_B_test_predict_vgg16.tolist() for x in sublist]

spectrograms_F_test_predict_vgg16 = model_vgg16.predict(spectrograms_F_test / 255.0)
spectrograms_F_test_predict_vgg16 = [x for sublist in spectrograms_F_test_predict_vgg16.tolist() for x in sublist]

spectrograms_N_test_predict_vgg16 = model_vgg16.predict(spectrograms_N_test / 255.0)
spectrograms_N_test_predict_vgg16 = [x for sublist in spectrograms_N_test_predict_vgg16.tolist() for x in sublist]

############### ResNet50: prediction ##############
spectrograms_B_test_predict_ResNet50 = model_ResNet50.predict(spectrograms_B_test / 255.0)
spectrograms_B_test_predict_ResNet50 = [x for sublist in spectrograms_B_test_predict_ResNet50.tolist() for x in sublist]

spectrograms_F_test_predict_ResNet50 = model_ResNet50.predict(spectrograms_F_test / 255.0)
spectrograms_F_test_predict_ResNet50 = [x for sublist in spectrograms_F_test_predict_ResNet50.tolist() for x in sublist]

spectrograms_N_test_predict_ResNet50 = model_ResNet50.predict(spectrograms_N_test / 255.0)
spectrograms_N_test_predict_ResNet50 = [x for sublist in spectrograms_N_test_predict_ResNet50.tolist() for x in sublist]

############### DenseNet121: prediction ##############
spectrograms_B_test_predict_DenseNet121 = model_DenseNet121.predict(spectrograms_B_test / 255.0)
spectrograms_B_test_predict_DenseNet121 = [x for sublist in spectrograms_B_test_predict_DenseNet121.tolist() for x in sublist]

spectrograms_F_test_predict_DenseNet121 = model_DenseNet121.predict(spectrograms_F_test / 255.0)
spectrograms_F_test_predict_DenseNet121 = [x for sublist in spectrograms_F_test_predict_DenseNet121.tolist() for x in sublist]

spectrograms_N_test_predict_DenseNet121 = model_DenseNet121.predict(spectrograms_N_test / 255.0)
spectrograms_N_test_predict_DenseNet121 = [x for sublist in spectrograms_N_test_predict_DenseNet121.tolist() for x in sublist]

############### emsemble: prediction ##############
spectrograms_B_test_predict = [spectrograms_B_test_predict_cnn, spectrograms_B_test_predict_vgg16, spectrograms_B_test_predict_ResNet50, spectrograms_B_test_predict_DenseNet121]
spectrograms_B_test_predict_ensemble = np.average(spectrograms_B_test_predict, axis=0, weights = opt_weights)
spectrograms_B_test_predict_ensemble_wrong_predictions = [i for i,v in enumerate(spectrograms_B_test_predict_ensemble) if v < 0.5]
plt.hist(spectrograms_B_test_predict_ensemble)

spectrograms_F_test_predict = [spectrograms_F_test_predict_cnn, spectrograms_F_test_predict_vgg16, spectrograms_F_test_predict_ResNet50, spectrograms_F_test_predict_DenseNet121]
spectrograms_F_test_predict_ensemble = np.average(spectrograms_F_test_predict, axis=0, weights = opt_weights)
spectrograms_F_test_predict_ensemble_wrong_predictions = [i for i,v in enumerate(spectrograms_F_test_predict_ensemble) if v > 0.5]
plt.hist(spectrograms_F_test_predict_ensemble)

spectrograms_N_test_predict = [spectrograms_N_test_predict_cnn, spectrograms_N_test_predict_vgg16, spectrograms_N_test_predict_ResNet50, spectrograms_N_test_predict_DenseNet121]
spectrograms_N_test_predict_ensemble = np.average(spectrograms_N_test_predict, axis=0, weights = opt_weights)
spectrograms_N_test_predict_ensemble_wrong_predictions = [i for i,v in enumerate(spectrograms_N_test_predict_ensemble) if v > 0.5]
plt.hist(spectrograms_N_test_predict_ensemble)

print(1 - len(spectrograms_B_test_predict_ensemble_wrong_predictions) / len(spectrograms_B_test))  ## 92.26%
print(1 - len(spectrograms_F_test_predict_ensemble_wrong_predictions) / len(spectrograms_F_test))  ## 98.36%
print(1 - len(spectrograms_N_test_predict_ensemble_wrong_predictions) / len(spectrograms_N_test))  ## 99.9%

print(len(spectrograms_B_test_predict_ensemble_wrong_predictions))
print(len(spectrograms_F_test_predict_ensemble_wrong_predictions))
print(len(spectrograms_N_test_predict_ensemble_wrong_predictions))

plt.hist(spectrograms_B_test_predict_ensemble)
plt.xlabel('Predicted Probability')
plt.ylabel('Count of Spectrograms')

plt.hist(spectrograms_F_test_predict_ensemble)
plt.xlabel('Predicted Probability')
plt.ylabel('Count of Spectrograms')

plt.hist(spectrograms_B_test_predict_ensemble[spectrograms_B_test_predict_ensemble_wrong_predictions])
plt.hist(spectrograms_F_test_predict_ensemble[spectrograms_F_test_predict_ensemble_wrong_predictions])
plt.hist(spectrograms_N_test_predict_ensemble[spectrograms_N_test_predict_ensemble_wrong_predictions])

tp = len([i for i,v in enumerate(spectrograms_B_test_predict_ensemble) if v >= 0.5])
fn = len([i for i,v in enumerate(spectrograms_B_test_predict_ensemble) if v < 0.5])
tn = len([i for i,v in enumerate(spectrograms_F_test_predict_ensemble) if v < 0.5])
fp = len([i for i,v in enumerate(spectrograms_F_test_predict_ensemble) if v >= 0.5])
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + fp + tn + fn)

y_true = [1] * len(spectrograms_B_test_predict_ensemble) + [0] * len(spectrograms_F_test_predict_ensemble)
y_scores = spectrograms_B_test_predict_ensemble.tolist() + spectrograms_F_test_predict_ensemble.tolist()

# Calculate ROC and AUC
AUC = roc_auc_score(y_true, y_scores) 
print('AUC: %.4f' % AUC)

# Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision_recall_auc = auc(recall, precision)
print('Precesion Recall AUC: %.4f' % precision_recall_auc)

# Plot precision-recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()