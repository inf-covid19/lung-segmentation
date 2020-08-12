# -*- coding: utf-8 -*-
'''
binary classifier implemented with transfer learning, using vgg16
this analysis includes patients classified as 'indeterminado', having PCR negativo
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
#    set_trainable = False
#    for layer in conv_base.layers:
#        if layer.name == 'block3_conv1':
#            set_trainable = True
#        if set_trainable:
#            layer.trainable = True
#        else:
#            layer.trainable = False
	    
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
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
        	(train_img_src_folder, train_img_folders, train_images),
        	(validation_img_src_folder, validation_img_folders, validation_images)
    ]
    
    for (base, folder, dic) in df_config:
        for img_folder, img_label in folder.items():
            search_folder = "{}/{}".format(base, img_folder)
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



#here starts the main funciton

img_src_folder = 'ImagensProcessadas'
#1-5: normal patients, 11-20: covid patients.
other_patients = []
covid_patients = []

data_description = 'annotations.xlsx' #file with the data annotations
annotations = pd.read_excel(data_description)
    
#in this analysis, patients described below were not included:
#  classification 4 - atipico
#  classification 3 - indeterminado
validation_other_folders = []
c=0
normal_lungs_added = 0
infected_lungs_added = 0
for i in range(len(annotations)):
	if str(annotations["nome"][i]) != "nan":
		c+=1
		p_num = int(annotations["nome"][i][1:])
		p_id = 'C' + str(p_num)
		if annotations["Classificação"][i] == "2 - típico" and annotations["PCR_FINAL"][i] == 1:
			covid_patients.append(p_id)
		elif annotations["Classificação"][i] == "1 - negativo" and annotations["PCR_FINAL"][i] == 2:
			other_patients.append(p_id)
			if normal_lungs_added < 5:
				validation_other_folders.append(p_id)
				normal_lungs_added += 1
		elif annotations["Classificação"][i] == "3 - indeterminado" and annotations["PCR_FINAL"][i] == 2:
			other_patients.append(p_id)
			if infected_lungs_added < 5:
				validation_other_folders.append(p_id)
				infected_lungs_added += 1
		else: c-=1 #in case of outliers patients that were not included in this analysis
print('included patients from annotated excel file: ', c)
print('')
print('Class other patients has size: ', len(other_patients))
other_p = sorted([int(p[1:]) for p in other_patients])
print(other_p)
print('')
print('Class covid patients has size', len(covid_patients))
covid_p = sorted([int(p[1:]) for p in covid_patients])
print(covid_p)
print('')

print('Total number of patients in one of the two classes: ', 
	      len(other_patients) + len(covid_patients))

#creating a mirrored strategy
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#fitting and training the model
with strategy.scope():
	model = get_model()

#fitting a model

#validation_other_folders = ['C'+str(p) for p in other_p[:7]]
validation_covid_folders = ['C'+str(p) for p in covid_p[:10]]
#validation_indefinite_folders = []

train_other_folders = list(set(other_patients) - set(validation_other_folders))
train_covid_folders = list(set(covid_patients) - set(validation_covid_folders))
#train_indefinite_folders = list(set(indefinite_patients) - set(validation_indefinite_folders))

print("Validating Folders: ", validation_other_folders, 
	      validation_covid_folders) #, sorted(validation_indefinite_folders))
#print("Training Folders: ", sorted(train_other_folders), 
#	      sorted(train_covid_folders)) , sorted(train_indefinite_folders))

train_img_src_folder = img_src_folder
validation_img_src_folder = img_src_folder

train_set, validation_set = get_dicts(train_other_folders, train_covid_folders, 
		                              validation_other_folders, validation_covid_folders)

train_df, validation_df = get_data(train_img_src_folder, train_set, 
		                          validation_img_src_folder, validation_set)

#training the model asd saving the label legends
train_model(model, train_df, validation_df, 20)
#getting the predictions of validation set by patient
predictions_by_patient(model, validation_other_folders)
predictions_by_patient(model, validation_covid_folders) 
   

