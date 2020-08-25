# -*- coding: utf-8 -*-
'''
instanciating the desired model and doing predictions for some test patients
test set:
	HCPA patients
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import os

from keras import layers
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import models
from keras import optimizers
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras import applications
from keras.applications import VGG16
from keras.applications import VGG19
import tensorflow as tf

import glob
from datetime import datetime

EXAM_SLICE = 100
width = 512
height = 512

def get_file_path(folder, search_filter=''):
    paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            if search_filter in path:
                paths.append(path)
    return paths

def get_data_generator(dataframe, x_col, y_col, subset=None, shuffle=True, batch_size=16, class_mode="binary"):
    datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1)
    
    data_generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col,
        subset=subset,
        target_size=(width, height),
        class_mode=class_mode,
        # color_mode="rgb",
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return data_generator			

def get_model():
	    
    conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(width,height,3))
    conv_base.trainable = True

	    
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = 'sigmoid'))
	    
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
		          loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predictions_by_patient(model, patients):
#this function will print the count of slices classified for each patient in a list of patients,
#it supposes that the label legend of the model in question is saved
#the image generators are generated on demand, which might be slow, consider changing it to 
#reciving a loaded generator if applying the function on validation set's patients
#the method used to get the prediction for each slice might not work for classifiers with more
#than 2 classes.

	results = []
	for p in patients:
		if p[:3] == 'TYP':
			p = p[4:]
			curr_dir = 'HCPA-Processadas/Typical/' + p + '/'
		elif p[:3] == 'NEG':
			p = p[4:]
			curr_dir = 'HCPA-Processadas/Negative/' + p + '/'
		else:
			curr_dir = 'ImagensProcessadas/' + p + '/'
		imgs_filename = sorted(os.listdir(curr_dir))
		test_filenames = imgs_filename[(len(imgs_filename)-EXAM_SLICE)//2:(len(imgs_filename)+EXAM_SLICE)//2]
		test_df = pd.DataFrame({
                	'filename': test_filenames
		})
		nb_samples = test_df.shape[0]

		test_gen = ImageDataGenerator(rescale=1./255)
		test_generator = test_gen.flow_from_dataframe(
                	test_df, 
                	curr_dir, 
               		x_col='filename',
                	y_col=None,
                	class_mode=None,
                	target_size=(width, height),
                	batch_size=16,
                	shuffle=False
		)

		predict = model.predict(test_generator, steps=np.ceil(nb_samples/16))
		test_df['category'] = [int(round(p[0])) for p in predict]
		results.append(test_df)

	for i,test_df in enumerate(results):
		print('Patient number: ', patients[i])
		if os.path.isfile('saved_legends/binary_balanced_legend.npy'):
			print('loading label legend file')
			class_indices = np.load('saved_legends/binary_balanced_legend.npy', allow_pickle=True).item()
			class_indices = dict((v,k) for k,v in class_indices.items())
			test_df['category'] = test_df['category'].replace(class_indices)
		print(test_df['category'].value_counts())
		print('')


#here starts the main funciton

#creating test set including hcpa patient's
test_negative_folders = []
test_typical_folders = []
test_img_src_folder = 'HCPA-Processadas'

for patient in os.listdir(test_img_src_folder + '/' + 'Negative'):
	patient_name = 'NEG-' + patient
	test_negative_folders.append(patient_name)
	
for patient in os.listdir(test_img_src_folder + '/' + 'Typical'):
	patient_name = 'TYP-' + patient
	test_typical_folders.append(patient_name)
	
print("Found", len(test_negative_folders), "HCPA Negative patients")
print("Found", len(test_typical_folders), "HCPA Typical patients")

print('')
print('Typical Patients in test set: ', test_typical_folders)
print('Negative Patients in test set: ', test_negative_folders)
print('')

#instanciating the model
model = get_model()

#loading the previously trained models
checkpoint_filepath = './saved_weights/binary_balanced_weights/binary_balanced_weights'
model.load_weights(checkpoint_filepath)

#getting the predictions of test set by patient

print('')
print('Predicting Test Patients')
print('')
print('')
print('Predicting HCPA Typical Patients')
print('')
predictions_by_patient(model, test_typical_folders)
print('')
print('Predicting HCPA Negative Patients')
print('')
predictions_by_patient(model, test_negative_folders)
