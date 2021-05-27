# -*- coding: utf-8 -*-

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
from keras.applications import ResNet50
from keras.applications import ResNet101V2
from keras.optimizers import schedules
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
	np.save('saved_legends/ternary_spgc.npy', train_generator.class_indices)
	    
	return history.history

def get_dicts(train_folders_1, train_folders_2, train_folders_3, validation_folders_1, validation_folders_2, validation_folders_3):
	train_set = dict(zip(train_folders_1, ['normal' for i in range(len(train_folders_1))]))
	train_set.update(dict(zip(train_folders_2, ['covid' for i in range(len(train_folders_2))])))
	train_set.update(dict(zip(train_folders_3, ['cap' for i in range(len(train_folders_3))])))

	validation_set = dict(zip(validation_folders_1, ['normal' for i in range(len(validation_folders_1))]))
	validation_set.update(dict(zip(validation_folders_2, ['covid' for i in range(len(validation_folders_2))])))
	validation_set.update(dict(zip(validation_folders_3, ['cap' for i in range(len(validation_folders_3))])))
	
	return train_set, validation_set

def get_data(train_img_folders, validation_img_folders):   
	dfs = []
	train_images = {"id": [], "label": []}
	validation_images = {"id": [], "label": []}

	df_config = [
		(train_img_folders, train_images),
		(validation_img_folders, validation_images)
	]

	for (folder, dic) in df_config:
		for img_folder, img_label in folder.items():
			if img_folder[0] == 'P':
                		search_folder = "{}/{}/z".format('SPGC/Covid', str(int(img_folder[-3:]))) #in case of a SPGC covid patient
			elif img_folder[:3] == 'cap':
				search_folder = "{}/{}/z".format('SPGC/Cap', str(int(img_folder[-3:]))) #in case of a SPGC cap patient
			else:
				search_folder = "{}/{}/z".format('SPGC/Normal', str(int(img_folder[-3:]))) #in case of a SPGC normal patient

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


def main():

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	annotations_filepth = './SPGC/Stats.csv'
	annotations_csv = pd.read_csv(annotations_filepth)
	
	validation_normal_folders = [annotations_csv["Folder"][i] for i in range(len(annotations_csv)) if 
		annotations_csv["Diagnosis"][i] == "Normal" and annotations_csv["Category"][i] == "Valid"]
	train_normal_folders = [annotations_csv["Folder"][i] for i in range(len(annotations_csv)) if 
		annotations_csv["Diagnosis"][i] == "Normal" and annotations_csv["Category"][i] == "Train"]
	
	validation_covid_folders = [annotations_csv["Folder"][i] for i in range(len(annotations_csv)) if 
		annotations_csv["Diagnosis"][i] == "COVID-19" and annotations_csv["Category"][i] == "Valid"]
	train_covid_folders = [annotations_csv["Folder"][i] for i in range(len(annotations_csv)) if 
		annotations_csv["Diagnosis"][i] == "COVID-19" and annotations_csv["Category"][i] == "Train"]
	
	validation_cap_folders = [annotations_csv["Folder"][i] for i in range(len(annotations_csv)) if 
		annotations_csv["Diagnosis"][i] == "CAP" and annotations_csv["Category"][i] == "Valid"]
	train_cap_folders = [annotations_csv["Folder"][i] for i in range(len(annotations_csv)) if 
		annotations_csv["Diagnosis"][i] == "CAP" and annotations_csv["Category"][i] == "Train"]
	
	print('')
	print('Typ Patients in training set: ', len(train_covid_folders), train_covid_folders)
	print('Cap Patients in training set: ', len(train_cap_folders), train_cap_folders)
	print('Normal Patients in training set: ', len(train_normal_folders), train_normal_folders)
	print('')
	print('Typ Patients in validation set: ', len(validation_covid_folders), validation_covid_folders)
	print('Cap Patients in validation set: ', len(validation_cap_folders), validation_cap_folders)
	print('Normal Patients in training set: ', len(validation_normal_folders), validation_normal_folders)
	print('')

	model = get_model()

	#fitting a model
	train_set, validation_set = get_dicts(train_normal_folders, train_covid_folders, train_cap_folders, validation_normal_folders,
		 validation_covid_folders, validation_cap_folders)

	train_df, validation_df = get_data(train_set, validation_set)

	#training the model and saving the label legends
	checkpoint_filepath = 'SavedModels/Resnet/ternary_spgc_train/weights_'
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath = checkpoint_filepath,
		save_weights_only = True,
		monitor = 'val_accuracy',
		mode = 'max',
		save_best_only = True)

	train_model(model, train_df, validation_df, 20, callbacks = [model_checkpoint_callback])
	#loading the best weights
	model.load_weights(checkpoint_filepath)
	model.save('SavedModels/Resnet/ternary_spgc_train/model')

	#getting the predictions of validation set by patient
	print('')
	print('Predicting Validation Patients')
	print('')
	print('')
	print('Predicting Covid Patients')
	print('')
	predictions_by_patient(model, validation_covid_folders)
	print('')
	print('Predicting Normal Patients')
	print('')
	predictions_by_patient(model, validation_normal_folders)
	print('')
	print('Predicting Cap Patients')
	print('')
	predictions_by_patient(model, validation_cap_folders)
	
if __name__ == "__main__":
	main()
