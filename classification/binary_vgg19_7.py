# -*- coding: utf-8 -*-
'''
binary classifier implemented with transfer learning, using vgg16
the program shoud train a model using HMV patient's (having negative and typical classifications) and test using patients from HCPA
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
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = 'sigmoid'))
	    
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
		          loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predictions_by_patient(model, patients, img_src_folder):
#this function will print the count of slices classified for each patient in a list of patients,
#it supposes that the label legend of the model in question is saved
#the image generators are generated on demand, which might be slow, consider changing it to 
#reciving a loaded generator if applying the function on validation set's patients
#the method used to get the prediction for each slice might not work for classifiers with more
#than 2 classes.

	results = []
	for p in patients:
		if p[:3] == 'TYP' or p[:3] == 'NEG':
			p = p[4:]
		curr_dir = img_src_folder + '/' + p + '/'
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
		if os.path.isfile('vgg_legend.npy'):
			print('loading label legend file')
			class_indices = np.load('vgg_legend.npy', allow_pickle=True).item()
			class_indices = dict((v,k) for k,v in class_indices.items())
			test_df['category'] = test_df['category'].replace(class_indices)
		print(test_df['category'].value_counts())
		print('')

def train_model(model, train_df, validation_df, epochs, callbacks=[]):
	batch_size = 10
	train_generator = get_data_generator(train_df, "id", "label", batch_size=batch_size)
	validation_generator = get_data_generator(validation_df, "id", "label",batch_size=batch_size)

	step_size_train = train_generator.n // train_generator.batch_size
	step_size_validation = validation_generator.n // validation_generator.batch_size

	if step_size_train == 0:
		step_size_train = train_generator.n // 2
		step_size_validation = validation_generator.n // 2

	history = model.fit(
		train_generator,
		steps_per_epoch=step_size_train,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=step_size_validation,
		callbacks=callbacks
	)
	    
	#needed when a model is loaded
	np.save('vgg_legend', train_generator.class_indices)
	    
	return history.history

def get_dicts(train_folders_1, train_folders_2, validation_folders_1, validation_folders_2):
	train_set = dict(zip(train_folders_1, ['other' for i in range(len(train_folders_1))]))
	train_set.update(dict(zip(train_folders_2, ['covid' for i in range(len(train_folders_2))])))
	validation_set = dict(zip(validation_folders_1, ['other' for i in range(len(validation_folders_1))]))
	validation_set.update(dict(zip(validation_folders_2, ['covid' for i in range(len(validation_folders_2))])))
	
	return train_set, validation_set

def get_data(train_img_src_folder, train_img_folders, validation_img_src_folder, validation_img_folders):   
    dfs = []
    train_images = {"id": [], "label": []}
    validation_images = {"id": [], "label": []}

    df_config = [
        	(train_img_folders, train_images),
        	(validation_img_folders, validation_images)
    ]
    
    for (folder, dic) in df_config:
        for img_folder, img_label in folder.items():
            if img_folder[:3] == 'TYP':
            	search_folder = "{}/{}".format('HCPA-Processadas/Typical', img_folder[4:]) #in case of a HCPA Typical patient
            elif img_folder[:3] == 'NEG':
            	search_folder = "{}/{}".format('HCPA-Processadas/Negative', img_folder[4:]) #in case of a HCPA Negative patient
            else:
            	search_folder = "{}/{}".format('ImagensProcessadas', img_folder) #in case of a HMV patient
            imgs_filename = sorted(get_file_path(search_folder, search_filter = ''))
            imgs_filename = imgs_filename[(len(imgs_filename)-EXAM_SLICE)//2:(len(imgs_filename)+EXAM_SLICE)//2]
            dic["id"].extend(imgs_filename)
            dic["label"].extend([img_label] * len(imgs_filename))

        dfs.append(pd.DataFrame(data=dic))

    train_df, validation_df = dfs[0], dfs[1]
    train_df.to_csv('train_df.csv', index=False)
    validation_df.to_csv('validation_df.csv', index=False)
	
    print("Train fold with {} images".format(len(train_df)))
    print(train_df.groupby("label").label.count())
    print()
    print("Validation fold with {} images".format(len(validation_df)))
    print(validation_df.groupby("label").label.count())
    print("-" * 30)
    
    return train_df, validation_df

def balance_classes(list_of_classes):
#recieves a list of classes, each class being a list of patients
#return the same list but balanced, based on the number of patients from minoritary class and leftover patients

#searching for the class with less patients
	MAX = 1000
	min_elements = MAX
	for c in list_of_classes:
		size = len(c)
		if size < min_elements:
			min_elements = size

#excluding random elements until all classes have the same number of elements
	leftover = [] #list of lists, leftover patients by class
	for index in range(len(list_of_classes)):
		leftover.append([])
		while(len(list_of_classes[index]) > min_elements):
			random_int = random.randint(0, len(list_of_classes[index]) - 1)
			leftover[index].append(list_of_classes[index][random_int])
			list_of_classes[index].pop(random_int)

	return list_of_classes, leftover


#here starts the main funciton

train_img_src_folder = 'ImagensProcessadas'
#reading hmv patient's annotation files and including these patients in train set
train_other_folders = []
train_covid_folders = []

data_description = 'annotations.xlsx' #file with the data annotations
annotations = pd.read_excel(data_description)
    
#in this analysis, patients described below were not included:
#  classification 4 - atipico
#  classification 3 - indeterminado
c=0
for i in range(len(annotations)):
	if str(annotations["nome"][i]) != "nan":
		c+=1
		p_num = int(annotations["nome"][i][1:])
		p_id = 'C' + str(p_num)
		if annotations["Classificação"][i] == "2 - típico" and annotations["PCR_FINAL"][i] == 1 and p_num != 53 and p_num != 95 and p_num != 153:
			train_covid_folders.append(p_id)
		elif annotations["Classificação"][i] == "1 - negativo" and annotations["PCR_FINAL"][i] == 2:
			train_other_folders.append(p_id)
		else: c-=1 #in case of outliers patients that were not included in this analysis
print('included patients from annotated excel file: ', c)
print('')
print('Class other patients has size: ', len(train_other_folders))
other_p = sorted([int(p[1:]) for p in train_other_folders])
print(other_p)
print('')
print('Class covid patients has size', len(train_covid_folders))
covid_p = sorted([int(p[1:]) for p in train_covid_folders])
print(covid_p)
print('')

print('Total number of patients in one of the two classes: ', 
	      len(train_other_folders) + len(train_covid_folders))

#creating validation set including hcpa patient's
validation_other_folders = []
validation_covid_folders = []
validation_img_src_folder = 'HCPA-Processadas'

for patient in os.listdir(validation_img_src_folder + '/' + 'Negative'):
	patient_name = 'NEG-' + patient
	validation_other_folders.append(patient_name)
	
for patient in os.listdir(validation_img_src_folder + '/' + 'Typical'):
	patient_name = 'TYP-' + patient
	validation_covid_folders.append(patient_name)
	
print("Found", len(validation_other_folders), "HCPA Negative patients")
print("Found", len(validation_covid_folders), "HCPA Typical patients")

#balancing training classes
classes = []
classes.append(train_other_folders)
classes.append(train_covid_folders)

classes, leftover = balance_classes(classes)
train_other_folders = classes[0]
train_covid_folders = classes[1]

covid_leftover = leftover[1]
validation_covid_folders = validation_covid_folders + covid_leftover

print('Typical Patients in training set: ', train_covid_folders)
print('Negative Patients in training set: ', train_other_folders)
print('')
print('Typical Patients in validation set: ', validation_covid_folders)
print('Negative Patients in validation set: ', validation_other_folders)
print('')

#creating a mirrored strategy
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#fitting and training the model
with strategy.scope():
	model = get_model()

#fitting a model
train_set, validation_set = get_dicts(train_other_folders, train_covid_folders, 
		                              validation_other_folders, validation_covid_folders)

train_df, validation_df = get_data(train_img_src_folder, train_set, 
		                          validation_img_src_folder, validation_set)

#training the model and saving the label legends
checkpoint_filepath = './checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath = checkpoint_filepath,
	save_weights_only = True,
	monitor = 'val_accuracy',
	mode = 'max',
	save_best_only = True)

train_model(model, train_df, validation_df, 10, callbacks = [model_checkpoint_callback])
#loading the best weights
model.load_weights(checkpoint_filepath)

#getting the predictions of test set by patient
negative_src = './HCPA-Processadas/Negative'
typical_src = './HCPA-Processadas/Typical'
negative_patients = os.listdir(negative_src)
typical_patients = os.listdir(typical_src)
print('')
print('Predicting Validation Patients')
print('')
print('')
print('Predicting Typical Patients')
print('')
predictions_by_patient(model, validation_covid_folders)
print('')
print('Predicting Negative Patients')
print('')
predictions_by_patient(model, validation_other_folders)
