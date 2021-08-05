# -*- coding: utf-8 -*-

'''
	First step classifier training
	
	Binary classification, with the following classes:
	
	Class Covid: typ
	Class Others: atyp, neg, ind
	
	Model backbone: Densenet121
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
from keras.applications import DenseNet121
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
      	color_mode="rgb",
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return data_generator			

def get_model():

	conv_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(width,height,3))
	conv_base.trainable = True

	x = conv_base.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(512, activation = 'relu')(x)
	x = Dropout(0.3)(x)
	x = Dense(512, activation = 'relu')(x)
	x = Dropout(0.3)(x)
	x = BatchNormalization()(x)
	preds = Dense(1, activation = 'sigmoid')(x)

	model = Model(inputs=conv_base.input, outputs=preds) 

	model.compile(optimizer=optimizers.Adam(learning_rate=2e-5),
                          loss='binary_crossentropy', metrics=['accuracy'])
	return model

def predictions_by_patient(model, patients):

	results = []
	for p in patients:
		if p[:3] == 'TYP':
			p = p[4:]
			curr_dir = '../HCPA_raw/Typical/' + p + '/z/'

		elif p[:3] == 'NEG':
			p = p[4:]
			curr_dir = '../HCPA_raw/Negative/' + p + '/z/'

		elif p[:3] == 'ATY':
			p = p[4:]
			curr_dir = '../HCPA_raw/Atypical/' + p + '/z/'

		elif p[:3] == 'IND':
			p = p[4:]
			curr_dir = '../HCPA_raw/Indeterminate/' + p + '/z/'

		elif p[0] == 'C':
			curr_dir = '../HMV_raw/' + p + '/z/'

		else:
			curr_dir = '../HMV_raw/' + p + '/z/'

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
		if os.path.isfile('SavedLegends/binary1.npy'):
			print('loading label legend file')
			class_indices = np.load('SavedLegends/binary1.npy', allow_pickle=True).item()
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
	np.save('SavedLegends/binary1.npy', train_generator.class_indices)
	    
	return history.history

def get_dicts(train_folders_1, train_folders_2, validation_folders_1, validation_folders_2):
	train_set = dict(zip(train_folders_1, ['others' for i in range(len(train_folders_1))]))
	train_set.update(dict(zip(train_folders_2, ['typical' for i in range(len(train_folders_2))])))

	validation_set = dict(zip(validation_folders_1, ['others' for i in range(len(validation_folders_1))]))
	validation_set.update(dict(zip(validation_folders_2, ['typical' for i in range(len(validation_folders_2))])))
	
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
			if img_folder[:3] == 'TYP':
                		search_folder = "{}/{}/z".format('../HCPA_raw/Typical', img_folder[4:]) #in case of a HCPA Typical patient
			elif img_folder[:3] == 'NEG':
				search_folder = "{}/{}/z".format('../HCPA_raw/Negative', img_folder[4:]) #in case of a HCPA Negative patient
			elif img_folder[:3] == 'IND':
                                search_folder = "{}/{}/z".format('../HCPA_raw/Indeterminate', img_folder[4:]) #in case of a HCPA IND patient
			elif img_folder[:3] == 'ATY':
                                search_folder = "{}/{}/z".format('../HCPA_raw/Atypical', img_folder[4:]) #in case of a HCPA ATY patient
			elif img_folder[0] == 'C':
				search_folder = "{}/{}/z".format('../HMV_raw', img_folder) 
			else:
				search_folder = "{}/{}/z".format('../HMV_raw', img_folder)

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

	train_typical_exams = ['TYP-028', 'TYP-007', 'C138', 'C13', 'C51', 'C12', 'C79', 'TYP-011', 'TYP-021', 'C114', 'C71', 'TYP-002', 'C112', 'TYP-022', 'TYP-031', 'N266', 'C111', 'C60', 'TYP-025', 'C146', 'C15', 'TYP-006', 'C150', 'C135', 'TYP-023', 'C133', 'C75', 'C91', 'C125', 'C80', 'C117', 'C16', 'TYP-014', 'C155', 'C151', 'TYP-026']
	train_negative_exams = ['ATY-014', 'N276', 'N165', 'N232', 'N261', 'C24', 'C86', 'N191', 'N181', 'ATY-021', 'N171', 'ATY-008', 'N288', 'N273', 'N173', 'ATY-030', 'N281', 'C58', 'IND-028', 'N275', 'N289', 'C107', 'C28', 'N198', 'IND-001', 'IND-018', 'C63', 'ATY-010', 'IND-021', 'N291', 'C5', 'C89', 'NEG-004', 'C100', 'IND-011', 'C102', 'N176', 'N264', 'N278', 'N294', 'N256', 'N242', 'N283', 'N259', 'N267', 'IND-025', 'C64', 'N187', 'N236', 'ATY-002', 'N293', 'C134', 'C139', 'NEG-009', 'ATY-011', 'N200', 'IND-024', 'C42', 'IND-007', 'C62', 'ATY-003', 'N255', 'N240', 'C46', 'N260', 'N244', 'N219', 'N210', 'C8', 'ATY-023', 'N189', 'IND-002', 'C99', 'N257', 'C57', 'C67', 'C84', 'C97']

	validation_typical_exams = ['C85', 'C162', 'C161', 'C143', 'TYP-027', 'N285', 'N190', 'N167', 'C158', 'C20', 'C124', 'C83']
	validation_negative_exams = ['NEG-007', 'C126', 'N185', 'C72', 'N197', 'NEG-012', 'C30', 'C66', 'N237', 'C37', 'C78', 'ATY-006', 'N243', 'N234', 'N254', 'C70', 'N252', 'IND-022', 'NEG-015', 'NEG-008', 'C65', 'C47', 'C127', 'N166', 'C3', 'IND-027', 'IND-013'] 

	
	print('')
	print('Typ Patients in training set: ', len(train_typical_exams), train_typical_exams)
	print('Others Patients in training set: ', len(train_negative_exams), train_negative_exams)
	print('')
	print('Typ Patients in validation set: ', len(validation_typical_exams), validation_typical_exams)
	print('Others Patients in validation set: ', len(validation_negative_exams), validation_negative_exams)
	print('')

	model = get_model()

	#fitting a model
	train_set, validation_set = get_dicts(train_negative_exams, train_typical_exams, 
				                      validation_negative_exams, validation_typical_exams)

	train_df, validation_df = get_data(train_set, validation_set)

	#training the model and saving the label legends
	checkpoint_filepath = 'SavedModels/1Etapa/SavedModels/Densenet121/binary1/{epoch:02d}-{val_loss:.2f}.hdf5'
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath = checkpoint_filepath,
		save_weights_only = True,
		monitor = 'val_accuracy',
		mode = 'max',
		save_best_only = False,
		save_freq="epoch")

	train_model(model, train_df, validation_df, 30, callbacks = [model_checkpoint_callback])

	#getting the predictions of validation set by patient
	print('')
	print('Predicting Validation Patients')
	print('')
	print('')
	print('Predicting Typ Patients')
	print('')
	predictions_by_patient(model, validation_typical_exams)
	print('')
	print('Predicting Others Patients')
	print('')
	predictions_by_patient(model, validation_negative_exams)
	print('')
	
if __name__  == "__main__":
	main()
