#
# model_densenet.py
#
# Train the DenseNet model
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

#%% Imports

from sklearn.model_selection import train_test_split
import csv
import numpy as np
import matplotlib.pyplot as plt

# Keras imports
from keras import models, layers, optimizers
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

#noaa additions
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

for i in [model_dir, output_spectrogram_vector_dir]:

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


#%% Step 2: build model

# Hyper-parameters
from keras.applications import DenseNet121

# Load the DenseNet121 model
DenseNet121_conv = DenseNet121(weights='imagenet', include_top=False, input_shape=(nrow, ncol, 3))

for layer in DenseNet121_conv.layers:
    layer.trainable = True
 # Check the trainable status of the individual layers

for layer in DenseNet121_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()
model.add(DenseNet121_conv)
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


#%% Compile Model
# Compile the model

#optimizers.adam was deprecated, but is accessible as tf.keras.Adam
#same arguments
#optimizer = optimizers.adam(lr=0.0001, decay=1e-7)

optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-7)

model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# Show a summary of the model. Check the number of trainable parameters
model.summary()


#%% fit model
try:
    #time to fit 
    model_history = model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=1, validation_data=(X_validation, y_validation))
except:
     traceback.print_exc()

#%% plot results
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

%## save model
model.save_weights(os.path.join(model_dir, "_".join([model_name,'DenseNet121_weights_all_data.h5'])))
# Save the model architecture
with open(os.path.join(model_dir, "_".join([model_name,'DenseNet121_architecture_all_data.json'])), 'w') as f:
    f.write(model.to_json())
    
#%% Step 3: predict on the test set
    
del X_train
del X_validation
del spectrograms_B_train_validation
del spectrograms_F_train_validation
del spectrograms_N_train_validation

spectrograms_B_test_predict = model.predict(spectrograms_B_test / 255.0)
spectrograms_B_test_wrong_predictions = [i for i,v in enumerate(spectrograms_B_test_predict) if v < 0.5]
plt.hist(spectrograms_B_test_predict)
print(1 - len(spectrograms_B_test_wrong_predictions) / len(spectrograms_B_test_predict))

spectrograms_F_test_predict = model.predict(spectrograms_F_test / 255.0)
spectrograms_F_test_wrong_predictions = [i for i,v in enumerate(spectrograms_F_test_predict) if v > 0.5]
plt.hist(spectrograms_F_test_predict)
print(1 - len(spectrograms_F_test_wrong_predictions) / len(spectrograms_F_test_predict))

spectrograms_N_test_predict = model.predict(spectrograms_N_test / 255.0)
spectrograms_N_test_wrong_predictions = [i for i,v in enumerate(spectrograms_N_test_predict) if v > 0.5]
plt.hist(spectrograms_N_test_predict)
print(1 - len(spectrograms_N_test_wrong_predictions) / len(spectrograms_N_test_predict))

tp = len([i for i,v in enumerate(spectrograms_B_test_predict) if v >= 0.5])
fn = len([i for i,v in enumerate(spectrograms_B_test_predict) if v < 0.5])
tn = len([i for i,v in enumerate(spectrograms_F_test_predict) if v < 0.5])
fp = len([i for i,v in enumerate(spectrograms_F_test_predict) if v >= 0.5])
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + fp + tn + fn) 

y_true = [1] * len(spectrograms_B_test_predict) + [0] * len(spectrograms_F_test_predict)
y_scores = spectrograms_B_test_predict.tolist() + spectrograms_F_test_predict.tolist()

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


#%% save weights
# Save the weights
model.save_weights(os.path.join(model_dir, "-".join([model_name, 'DenseNet121_weights_all_data.h5'])))

# Save the model architecture
with open(os.path.join(model_dir, "-".join([model_name, 'DenseNet121_architecture_all_data.json'])), 'w') as f:
    f.write(model.to_json())
    
 #%% Step 3: predict on the test set
    
del X_train
del X_validation
del spectrograms_B_train_validation
del spectrograms_F_train_validation
del spectrograms_N_train_validation

spectrograms_B_test_predict = model.predict(spectrograms_B_test / 255.0)
spectrograms_B_test_wrong_predictions = [i for i,v in enumerate(spectrograms_B_test_predict) if v < 0.5]
plt.hist(spectrograms_B_test_predict)
print(1 - len(spectrograms_B_test_wrong_predictions) / len(spectrograms_B_test_predict))

spectrograms_F_test_predict = model.predict(spectrograms_F_test / 255.0)
spectrograms_F_test_wrong_predictions = [i for i,v in enumerate(spectrograms_F_test_predict) if v > 0.5]
plt.hist(spectrograms_F_test_predict)
print(1 - len(spectrograms_F_test_wrong_predictions) / len(spectrograms_F_test_predict))

spectrograms_N_test_predict = model.predict(spectrograms_N_test / 255.0)
spectrograms_N_test_wrong_predictions = [i for i,v in enumerate(spectrograms_N_test_predict) if v > 0.5]
plt.hist(spectrograms_N_test_predict)
print(1 - len(spectrograms_N_test_wrong_predictions) / len(spectrograms_N_test_predict))

tp = len([i for i,v in enumerate(spectrograms_B_test_predict) if v >= 0.5])
fn = len([i for i,v in enumerate(spectrograms_B_test_predict) if v < 0.5])
tn = len([i for i,v in enumerate(spectrograms_F_test_predict) if v < 0.5])
fp = len([i for i,v in enumerate(spectrograms_F_test_predict) if v >= 0.5])
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + fp + tn + fn) 

y_true = [1] * len(spectrograms_B_test_predict) + [0] * len(spectrograms_F_test_predict)
y_scores = spectrograms_B_test_predict.tolist() + spectrograms_F_test_predict.tolist()

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