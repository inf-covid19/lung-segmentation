# -*- coding: utf-8 -*-
'''
instanciating the desired model and doing predictions for SPGC validation set.
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import os

from keras import layers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import models
from keras import optimizers
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras import applications
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import ResNet101V2
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

def get_data_generator(dataframe, x_col, y_col, subset=None, shuffle=True, batch_size=16, class_mode="categorical"):
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
      	color_mode="rgb",
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return data_generator			

def get_model():

	conv_base = ResNet101V2(weights='imagenet', include_top=False, input_shape=(width,height,3))
	conv_base.trainable = True

	x = conv_base.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation = 'relu')(x)
	x = Dropout(0.4)(x)
	x = Dense(1024, activation = 'relu')(x)
	x = Dropout(0.4)(x)
	x = BatchNormalization()(x)
	preds = Dense(3, activation = 'softmax')(x)

	model = Model(inputs=conv_base.input, outputs=preds) 

	model.compile(optimizer=optimizers.Adam(learning_rate=2e-5),
                          loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def predictions_by_patient(model, patients):

	results = []
	for p in patients:
		p_id = p[-3:]
		if p[0] == 'P':
			curr_dir = "{}/{}/z".format('SPGC/Covid', str(int(p_id[-3:]))) #in case of a SPGC covid patient
		elif p[:3] == 'cap':
			curr_dir = "{}/{}/z".format('SPGC/Cap', str(int(p_id[-3:]))) #in case of a SPGC cap patient
		else:
			curr_dir = "{}/{}/z".format('SPGC/Normal', str(int(p_id[-3:]))) #in case of a SPGC normal patient

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
		test_df['category'] = [np.where(pr == np.max(pr))[0][0] for pr in predict]
		results.append(test_df)

	for i,test_df in enumerate(results):
		print('Patient number: ', patients[i])
		if os.path.isfile('saved_legends/ternary_spgc.npy'):
			print('loading label legend file')
			class_indices = np.load('saved_legends/ternary_spgc.npy', allow_pickle=True).item()
			class_indices = dict((v,k) for k,v in class_indices.items())
			test_df['category'] = test_df['category'].replace(class_indices)
		print(test_df['category'].value_counts())
		print('')

def main():

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	annotations_filepth = './SPGC/Stats.csv'
	annotations_csv = pd.read_csv(annotations_filepth)
	
	validation_normal_folders = [annotations_csv["Folder"][i] for i in range(len(annotations_csv)) if 
		annotations_csv["Diagnosis"][i] == "Normal" and annotations_csv["Category"][i] == "Valid"]
	
	validation_covid_folders = [annotations_csv["Folder"][i] for i in range(len(annotations_csv)) if 
		annotations_csv["Diagnosis"][i] == "COVID-19" and annotations_csv["Category"][i] == "Valid"]
	
	validation_cap_folders = [annotations_csv["Folder"][i] for i in range(len(annotations_csv)) if 
		annotations_csv["Diagnosis"][i] == "CAP" and annotations_csv["Category"][i] == "Valid"]
	
	print('')
	print('Typ Patients in validation set: ', len(validation_covid_folders), validation_covid_folders)
	print('Cap Patients in validation set: ', len(validation_cap_folders), validation_cap_folders)
	print('Normal Patients in validation set: ', len(validation_normal_folders), validation_normal_folders)
	print('')

	#instanciating the model
	model = get_model()

	#loading the previously trained models
	checkpoint_filepath = 'SavedModels/Resnet/ternary_spgc_train/weights_'
	model.load_weights(checkpoint_filepath)

	#getting the predictions of test set by patient

	print('')
	print('Predicting Test Patients')
	print('')
	print('Predicting Normal Patients')
	print('')
	predictions_by_patient(model, validation_normal_folders)
	print('')
	print('Predicting Covid Patients')
	print('')
	predictions_by_patient(model, validation_covid_folders)
	print('')
	print('')
	print('Predicting CAP Patients')
	print('')
	predictions_by_patient(model, validation_cap_folders)
	print('')

if __name__ == "__main__":
	main()
