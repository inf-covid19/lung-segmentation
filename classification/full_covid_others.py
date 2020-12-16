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

	conv_base = ResNet101V2(weights='imagenet', include_top=False, input_shape=(width,height,3))
	conv_base.trainable = True

	x = conv_base.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation = 'relu')(x)
	x = Dropout(0.4)(x)
	x = Dense(1024, activation = 'relu')(x)
	x = Dropout(0.4)(x)
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
			curr_dir = 'HCPA-Processadas/Typical/' + p + '/'

		elif p[:3] == 'NEG':
			p = p[4:]
			curr_dir = 'HCPA-Processadas/Negative/' + p + '/'

		elif p[:3] == 'ATY':
			p = p[4:]
			curr_dir = 'HCPA-Processadas/Atypical/' + p + '/'

		elif p[:3] == 'IND':
			p = p[4:]
			curr_dir = 'HCPA-Processadas/Indeterminate/' + p + '/'

		elif p[0] == 'C':
			curr_dir = 'ImagensProcessadas/' + p + '/'

		else:
			curr_dir = 'HMV-Parte2/' + p + '/'

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
		if os.path.isfile('saved_legends/full_covid_others.npy'):
			print('loading label legend file')
			class_indices = np.load('saved_legends/full_covid_others.npy', allow_pickle=True).item()
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
	np.save('saved_legends/full_covid_others.npy', train_generator.class_indices)
	    
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
                		search_folder = "{}/{}".format('HCPA-Processadas/Typical', img_folder[4:]) #in case of a HCPA Typical patient
			elif img_folder[:3] == 'NEG':
				search_folder = "{}/{}".format('HCPA-Processadas/Negative', img_folder[4:]) #in case of a HCPA Negative patient
			elif img_folder[:3] == 'IND':
                                search_folder = "{}/{}".format('HCPA-Processadas/Indeterminate', img_folder[4:]) #in case of a HCPA IND patient
			elif img_folder[:3] == 'ATY':
                                search_folder = "{}/{}".format('HCPA-Processadas/Atypical', img_folder[4:]) #in case of a HCPA ATY patient
			elif img_folder[0] == 'C':
				search_folder = "{}/{}".format('ImagensProcessadas', img_folder) 
			else:
				search_folder = "{}/{}".format('HMV-Parte2', img_folder)

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

def train_validation_split(class_patients):
#recieves a list of patients belonging to a classification, returns randomized train and validation sets.
	validation_set = []
	training_set = class_patients
	while(len(validation_set) < len(class_patients)//5):
		random_index = random.randint(0, len(training_set) - 1)
		validation_set.append(training_set[random_index])
		training_set.pop(random_index)
		
	return training_set, validation_set
	
	
	
	
#here starts the main funciton
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_typical_fold1 = ['C116', 'C77', 'C51', 'TYP-031', 'N285', 'C27', 'C13', 'C49', 'N167', 'C96', 'TYP-003', 'TYP-009', 'TYP-007', 'C146', 'C157', 'C60', 'C145', 'N203', 'C111', 'TYP-012', 'C144', 'C115', 'C18', 'TYP-017', 'TYP-005', 'C71', 'N213', 'C135', 'TYP-029', 'C25', 'C158', 'C162', 'C124', 'C142', 'TYP-002', 'TYP-021', 'C112', 'N300', 'C11', 'C121', 'C150', 'TYP-020', 'C117', 'C32', 'N201', 'C44', 'C143', 'C75', 'C125', 'TYP-015', 'C21', 'N170', 'C39', 'C69', 'TYP-014', 'C83', 'C93', 'C131', 'C90', 'N238', 'C12', 'TYP-013', 'N286', 'C74', 'TYP-027', 'C136', 'TYP-010', 'TYP-028', 'TYP-016', 'TYP-025', 'N190', 'C133', 'N266', 'C160', 'C163', 'C130', 'C17', 'C151', 'C138', 'C23', 'C35', 'C132', 'C113', 'TYP-019', 'TYP-006', 'C88', 'C33', 'C15', 'C110', 'C16', 'C19', 'TYP-004', 'C91', 'TYP-030', 'TYP-022', 'TYP-011']
train_negative_fold1 = ['N239', 'C42', 'N233', 'IND-029', 'N197', 'NEG-007', 'N271', 'N229', 'N279', 'N196', 'NEG-005', 'C24', 'C57', 'ATY-002', 'IND-009', 'C126', 'ATY-005', 'ATY-029', 'C128', 'C119', 'C152', 'C54', 'IND-022', 'C122', 'C46', 'IND-027', 'C127', 'ATY-016', 'N187', 'N191', 'N255', 'N245', 'N273', 'N276', 'N261', 'ATY-012', 'N195', 'C64', 'N275', 'C40', 'C9', 'N256', 'C156', 'NEG-001', 'N251', 'NEG-002', 'C89', 'C59', 'N236', 'N237', 'IND-011', 'N289', 'N232', 'C73', 'IND-013', 'N207', 'NEG-011', 'N254', 'N290', 'N220', 'C58', 'C159', 'ATY-024', 'N243', 'C66', 'N180', 'N260', 'N282', 'N222', 'N200', 'N226', 'ATY-015', 'N221', 'N228', 'N204', 'ATY-028', 'N274', 'N294', 'N163', 'N291', 'N241', 'ATY-013', 'N172', 'C98', 'C5', 'C38', 'C61', 'C120', 'N189', 'C104', 'IND-002', 'C106', 'N188', 'N168', 'N193', 'NEG-003', 'N171', 'N198', 'N253', 'IND-007', 'IND-028', 'C107', 'C148', 'C37', 'C47', 'N214', 'N264', 'N175', 'C84', 'C67', 'C108', 'C34', 'C62', 'C48', 'ATY-008', 'N161', 'C100', 'NEG-006', 'N248', 'IND-018', 'C30', 'ATY-006', 'ATY-007', 'ATY-023', 'N202', 'C134', 'N297', 'N257', 'C164', 'N186', 'N259', 'IND-008', 'NEG-012', 'C55', 'N181', 'N216', 'C28', 'C72', 'N263', 'C137', 'C105', 'N192', 'N210', 'ATY-026', 'N296', 'NEG-014', 'ATY-010', 'N205', 'N206', 'ATY-009', 'C78', 'C43', 'N244', 'IND-001', 'C56', 'ATY-003', 'C50', 'N258', 'C118', 'C45', 'ATY-027', 'N242', 'N280', 'IND-023', 'ATY-021', 'N281', 'NEG-009', 'C52', 'NEG-008', 'N184', 'N169', 'C31', 'ATY-030', 'IND-005', 'IND-026', 'C3', 'N227', 'N299', 'C63', 'N164', 'N247', 'IND-020', 'IND-012', 'IND-019', 'N284', 'N246', 'C99', 'C87', 'ATY-004', 'N208', 'C141', 'IND-004', 'N283', 'C81', 'ATY-022', 'N173', 'N178', 'N293', 'ATY-014', 'N267', 'ATY-011', 'C68', 'N278', 'N185', 'C22', 'NEG-010', 'N194', 'C140', 'ATY-025', 'IND-016', 'N166', 'ATY-017']

validation_typical_fold1 = ['C154', 'C85', 'C114', 'C103', 'N160', 'C80', 'C14', 'TYP-023', 'N230', 'C36', 'C94', 'C161', 'C79', 'TYP-024', 'TYP-008', 'C155', 'C149', 'C82', 'C20', 'TYP-018', 'C26', 'C101', 'TYP-026', 'C41']
validation_negative_fold1 = ['N182', 'C123', 'N272', 'C92', 'IND-015', 'NEG-004', 'C97', 'N265', 'N269', 'N223', 'IND-017', 'C129', 'N249', 'N176', 'C109', 'C29', 'C147', 'N177', 'N295', 'N209', 'N219', 'IND-030', 'ATY-019', 'C76', 'IND-003', 'C102', 'N165', 'C86', 'NEG-013', 'N174', 'C8', 'N235', 'NEG-015', 'N212', 'IND-006', 'IND-024', 'N252', 'C70', 'IND-025', 'IND-014', 'IND-021', 'C65', 'N234', 'N270', 'N288', 'IND-010', 'ATY-018', 'N240', 'N231', 'ATY-020', 'ATY-001', 'N183', 'C139']

train_typical_fold2 = ['C116', 'N160', 'C51', 'TYP-031', 'N285', 'C85', 'C27', 'C13', 'C49', 'TYP-009', 'TYP-007', 'C146', 'C157', 'C60', 'N203', 'C111', 'TYP-023', 'C144', 'N230', 'C115', 'C20', 'C41', 'C18', 'TYP-017', 'TYP-018', 'C80', 'C103', 'C71', 'N213', 'C135', 'TYP-029', 'C158', 'C162', 'C124', 'TYP-021', 'N300', 'C121', 'C79', 'C150', 'C149', 'TYP-020', 'N201', 'C161', 'C44', 'TYP-008', 'C143', 'C75', 'C125', 'C21', 'N170', 'C39', 'C69', 'TYP-014', 'C93', 'C131', 'TYP-013', 'N286', 'C94', 'C74', 'TYP-027', 'C136', 'TYP-010', 'TYP-028', 'TYP-016', 'C14', 'C155', 'C82', 'N190', 'C133', 'C101', 'N266', 'C160', 'C163', 'C36', 'TYP-024', 'C130', 'C17', 'C138', 'C23', 'C35', 'C113', 'TYP-019', 'TYP-006', 'C88', 'C33', 'C15', 'C26', 'C114', 'C154', 'C19', 'TYP-004', 'TYP-026', 'C91', 'TYP-030', 'TYP-022', 'TYP-011']
train_negative_fold2 = ['C42', 'N233', 'N231', 'N197', 'NEG-007', 'N174', 'N270', 'N271', 'N229', 'N279', 'N196', 'NEG-005', 'IND-024', 'C57', 'ATY-002', 'IND-009', 'C126', 'ATY-029', 'C128', 'C119', 'C152', 'N234', 'C122', 'C46', 'IND-027', 'ATY-016', 'N295', 'N187', 'NEG-015', 'IND-003', 'N191', 'N255', 'N245', 'N273', 'N276', 'N261', 'ATY-012', 'N195', 'C64', 'N275', 'C40', 'C9', 'N256', 'N251', 'N165', 'C59', 'N236', 'N237', 'IND-011', 'N289', 'N232', 'N252', 'IND-013', 'N207', 'NEG-011', 'N254', 'N290', 'N220', 'C58', 'ATY-024', 'N243', 'C66', 'N180', 'N282', 'N240', 'N222', 'N200', 'ATY-015', 'N228', 'ATY-028', 'N274', 'N294', 'N241', 'IND-014', 'ATY-020', 'ATY-013', 'N172', 'N176', 'C98', 'N272', 'C38', 'C61', 'C120', 'N189', 'C104', 'N209', 'NEG-013', 'IND-002', 'N188', 'N168', 'N193', 'NEG-003', 'N171', 'C102', 'N198', 'N253', 'IND-007', 'IND-025', 'N219', 'IND-028', 'C107', 'C148', 'IND-021', 'NEG-004', 'C147', 'C37', 'IND-010', 'N214', 'N264', 'N175', 'C84', 'N223', 'C67', 'C108', 'C62', 'C48', 'ATY-008', 'C97', 'NEG-006', 'IND-018', 'C30', 'ATY-006', 'ATY-007', 'C134', 'N269', 'N257', 'N183', 'N186', 'N259', 'N265', 'C55', 'N181', 'N216', 'ATY-018', 'C72', 'N263', 'C105', 'N192', 'N210', 'ATY-026', 'N296', 'NEG-014', 'C123', 'ATY-010', 'IND-006', 'N235', 'C65', 'N205', 'C139', 'N206', 'ATY-009', 'N177', 'N249', 'C78', 'N244', 'IND-001', 'C56', 'ATY-003', 'C50', 'C76', 'ATY-001', 'ATY-027', 'N242', 'C70', 'C92', 'N280', 'IND-023', 'ATY-021', 'IND-015', 'N281', 'C52', 'NEG-008', 'N184', 'C31', 'IND-005', 'C3', 'N227', 'IND-017', 'C129', 'N247', 'IND-020', 'C86', 'IND-019', 'C109', 'C8', 'N284', 'N246', 'C87', 'ATY-004', 'N208', 'C141', 'ATY-019', 'C81', 'N293', 'N212', 'ATY-014', 'N267', 'N182', 'ATY-011', 'C68', 'N278', 'N185', 'C22', 'C29', 'NEG-010', 'C140', 'ATY-025', 'IND-016', 'N288', 'N166', 'ATY-017', 'IND-030']


validation_typical_fold2 = ['TYP-002', 'TYP-005', 'C96', 'C145', 'TYP-025', 'C132', 'TYP-015', 'C25', 'C151', 'C142', 'TYP-003', 'C83', 'C112', 'C32', 'C117', 'C11', 'C90', 'N167', 'C110', 'C77', 'N238', 'C12', 'C16', 'TYP-012']
validation_negative_fold2 = ['N164', 'ATY-030', 'C28', 'N221', 'C24', 'C106', 'NEG-001', 'N239', 'IND-029', 'C63', 'IND-026', 'ATY-023', 'C100', 'C164', 'NEG-009', 'IND-008', 'N258', 'C159', 'C118', 'C127', 'N161', 'C137', 'N173', 'N260', 'N297', 'ATY-005', 'N169', 'C54', 'N194', 'NEG-002', 'N299', 'IND-004', 'IND-012', 'C47', 'C34', 'C73', 'N163', 'C89', 'N226', 'N291', 'N178', 'N204', 'N283', 'C43', 'N248', 'C99', 'IND-022', 'NEG-012', 'ATY-022', 'N202', 'C45', 'C156', 'C5']
 
train_typical_fold3 = ['N160', 'C77', 'C51', 'TYP-031', 'N285', 'C85', 'C27', 'C49', 'N167', 'C96', 'TYP-003', 'TYP-009', 'C146', 'C60', 'C145', 'N203', 'C111', 'TYP-012', 'TYP-023', 'C144', 'N230', 'C115', 'C20', 'C41', 'C18', 'TYP-018', 'TYP-005', 'C80', 'C103', 'C71', 'N213', 'C135', 'TYP-029', 'C25', 'C158', 'C124', 'C142', 'TYP-002', 'TYP-021', 'C112', 'N300', 'C11', 'C121', 'C79', 'C150', 'C149', 'TYP-020', 'C117', 'C32', 'C161', 'TYP-008', 'C143', 'C125', 'TYP-015', 'C21', 'N170', 'C69', 'C83', 'C93', 'C131', 'C90', 'N238', 'C12', 'TYP-013', 'N286', 'C94', 'C74', 'TYP-010', 'TYP-028', 'TYP-016', 'TYP-025', 'C14', 'C155', 'C82', 'C133', 'C101', 'N266', 'C160', 'C163', 'C36', 'TYP-024', 'C130', 'C17', 'C151', 'C35', 'C132', 'TYP-019', 'C33', 'C26', 'C114', 'C110', 'C16', 'C154', 'C19', 'TYP-004', 'TYP-026']
train_negative_fold3 = ['N239', 'C42', 'N233', 'N231', 'IND-029', 'NEG-007', 'N174', 'N270', 'N279', 'N196', 'C24', 'IND-024', 'ATY-002', 'IND-009', 'ATY-005', 'ATY-029', 'C54', 'N234', 'IND-022', 'C46', 'IND-027', 'C127', 'N295', 'NEG-015', 'IND-003', 'N191', 'N255', 'N245', 'N273', 'N276', 'N261', 'ATY-012', 'N195', 'N275', 'C40', 'C9', 'N256', 'C156', 'NEG-001', 'N251', 'NEG-002', 'C89', 'N165', 'C59', 'N236', 'N237', 'N289', 'N232', 'C73', 'N252', 'IND-013', 'NEG-011', 'N254', 'N290', 'C58', 'C159', 'ATY-024', 'N243', 'C66', 'N260', 'N282', 'N240', 'N222', 'N200', 'N226', 'ATY-015', 'N221', 'N228', 'N204', 'ATY-028', 'N274', 'N163', 'N291', 'IND-014', 'ATY-020', 'N176', 'C98', 'N272', 'C5', 'C61', 'C104', 'N209', 'NEG-013', 'IND-002', 'C106', 'N168', 'NEG-003', 'C102', 'IND-007', 'IND-025', 'N219', 'IND-028', 'C107', 'IND-021', 'NEG-004', 'C147', 'IND-010', 'C47', 'N214', 'N175', 'C84', 'N223', 'C67', 'C108', 'C34', 'C62', 'C97', 'N161', 'C100', 'N248', 'C30', 'ATY-006', 'ATY-007', 'ATY-023', 'N202', 'N297', 'N269', 'N257', 'N183', 'C164', 'N186', 'N259', 'N265', 'IND-008', 'NEG-012', 'C55', 'N181', 'ATY-018', 'C28', 'C72', 'N263', 'C137', 'C105', 'N192', 'N210', 'ATY-026', 'NEG-014', 'C123', 'ATY-010', 'IND-006', 'N235', 'C65', 'N205', 'C139', 'N206', 'ATY-009', 'N177', 'N249', 'C78', 'C43', 'N244', 'ATY-003', 'C50', 'C76', 'N258', 'C118', 'ATY-001', 'C45', 'N242', 'C70', 'C92', 'N280', 'IND-023', 'IND-015', 'N281', 'NEG-009', 'C52', 'N169', 'C31', 'ATY-030', 'IND-005', 'IND-026', 'C3', 'N227', 'IND-017', 'N299', 'C63', 'C129', 'N164', 'N247', 'IND-020', 'IND-012', 'C86', 'C109', 'C8', 'N246', 'C99', 'C87', 'ATY-004', 'N208', 'ATY-019', 'IND-004', 'N283', 'C81', 'ATY-022', 'N173', 'N178', 'N293', 'N212', 'N267', 'N182', 'ATY-011', 'N185', 'C22', 'C29', 'N194', 'C140', 'IND-016', 'N288', 'N166', 'ATY-017', 'IND-030']

validation_typical_fold3 = ['C23', 'TYP-011', 'N201', 'N190', 'TYP-022', 'C113', 'C75', 'C39', 'C162', 'C15', 'TYP-006', 'C138', 'TYP-030', 'C116', 'TYP-007', 'C44', 'TYP-017', 'C136', 'C91', 'C88', 'C157', 'TYP-014', 'C13', 'TYP-027']
validation_negative_fold3 = ['N220', 'C141', 'ATY-025', 'N193', 'IND-001', 'N253', 'N189', 'C134', 'ATY-016', 'N264', 'N207', 'N171', 'N284', 'N187', 'IND-019', 'C119', 'N271', 'ATY-014', 'NEG-006', 'N184', 'N296', 'C148', 'C37', 'N180', 'N198', 'C152', 'N229', 'N241', 'C68', 'N216', 'ATY-008', 'C38', 'ATY-021', 'C48', 'N188', 'C122', 'N197', 'NEG-008', 'N294', 'NEG-010', 'NEG-005', 'C126', 'ATY-013', 'C57', 'C128', 'N172', 'C64', 'IND-011', 'C56', 'C120', 'N278', 'ATY-027', 'IND-018']

train_typical_fold4 = ['C116', 'N160', 'C77', 'C51', 'TYP-031', 'C85', 'C13', 'N167', 'C96', 'TYP-003', 'TYP-009', 'TYP-007', 'C146', 'C157', 'C145', 'TYP-012', 'TYP-023', 'N230', 'C20', 'C41', 'C18', 'TYP-017', 'TYP-018', 'TYP-005', 'C80', 'C103', 'N213', 'C135', 'TYP-029', 'C25', 'C162', 'C142', 'TYP-002', 'TYP-021', 'C112', 'N300', 'C11', 'C79', 'C150', 'C149', 'C117', 'C32', 'N201', 'C161', 'C44', 'TYP-008', 'C75', 'TYP-015', 'C39', 'C69', 'TYP-014', 'C83', 'C93', 'C90', 'N238', 'C12', 'TYP-013', 'C94', 'TYP-027', 'C136', 'TYP-016', 'TYP-025', 'C14', 'C155', 'C82', 'N190', 'C133', 'C101', 'N266', 'C160', 'C163', 'C36', 'TYP-024', 'C130', 'C17', 'C151', 'C138', 'C23', 'C35', 'C132', 'C113', 'TYP-006', 'C88', 'C33', 'C15', 'C26', 'C114', 'C110', 'C16', 'C154', 'C19', 'TYP-026', 'C91', 'TYP-030', 'TYP-022', 'TYP-011']
train_negative_fold4 = ['N239', 'N231', 'IND-029', 'N197', 'NEG-007', 'N174', 'N270', 'N271', 'N229', 'N196', 'NEG-005', 'C24', 'IND-024', 'C57', 'C126', 'ATY-005', 'C128', 'C119', 'C152', 'C54', 'N234', 'IND-022', 'C122', 'C46', 'C127', 'ATY-016', 'N295', 'N187', 'NEG-015', 'IND-003', 'N191', 'N273', 'N261', 'C64', 'C40', 'N256', 'C156', 'NEG-001', 'NEG-002', 'C89', 'N165', 'N237', 'IND-011', 'N289', 'C73', 'N252', 'N207', 'N254', 'N290', 'N220', 'C58', 'C159', 'C66', 'N180', 'N260', 'N282', 'N240', 'N222', 'N226', 'ATY-015', 'N221', 'N228', 'N204', 'N294', 'N163', 'N291', 'N241', 'IND-014', 'ATY-020', 'ATY-013', 'N172', 'N176', 'N272', 'C5', 'C38', 'C120', 'N189', 'C104', 'N209', 'NEG-013', 'IND-002', 'C106', 'N188', 'N193', 'N171', 'C102', 'N198', 'N253', 'IND-007', 'IND-025', 'N219', 'IND-028', 'C148', 'IND-021', 'NEG-004', 'C147', 'C37', 'IND-010', 'C47', 'N214', 'N264', 'N175', 'N223', 'C34', 'C48', 'ATY-008', 'C97', 'N161', 'C100', 'NEG-006', 'N248', 'IND-018', 'C30', 'ATY-006', 'ATY-007', 'ATY-023', 'N202', 'C134', 'N297', 'N269', 'N257', 'N183', 'C164', 'N265', 'IND-008', 'NEG-012', 'N216', 'ATY-018', 'C28', 'N263', 'C137', 'C105', 'N192', 'N210', 'N296', 'NEG-014', 'C123', 'ATY-010', 'IND-006', 'N235', 'C65', 'N205', 'C139', 'N206', 'N177', 'N249', 'C78', 'C43', 'N244', 'IND-001', 'C56', 'ATY-003', 'C50', 'C76', 'N258', 'C118', 'ATY-001', 'C45', 'ATY-027', 'C70', 'C92', 'ATY-021', 'IND-015', 'N281', 'NEG-009', 'NEG-008', 'N184', 'N169', 'ATY-030', 'IND-005', 'IND-026', 'N227', 'IND-017', 'N299', 'C63', 'C129', 'N164', 'N247', 'IND-012', 'C86', 'IND-019', 'C109', 'C8', 'N284', 'N246', 'C99', 'C87', 'ATY-004', 'N208', 'C141', 'ATY-019', 'IND-004', 'N283', 'ATY-022', 'N173', 'N178', 'N293', 'N212', 'ATY-014', 'N182', 'ATY-011', 'C68', 'N278', 'N185', 'C29', 'NEG-010', 'N194', 'C140', 'ATY-025', 'N288', 'ATY-017', 'IND-030']


validation_typical_fold4 = ['C124', 'C27', 'N203', 'C49', 'TYP-004', 'C71', 'N170', 'C60', 'C121', 'C158', 'TYP-019', 'C131', 'TYP-010', 'C125', 'N285', 'TYP-028', 'C143', 'N286', 'C21', 'C144', 'TYP-020', 'C115', 'C111', 'C74']
validation_negative_fold4 = ['N168', 'C42', 'IND-023', 'N279', 'N232', 'ATY-009', 'NEG-003', 'N242', 'N195', 'N267', 'ATY-029', 'N259', 'C59', 'C31', 'N275', 'N274', 'N200', 'C62', 'C108', 'ATY-026', 'C107', 'IND-016', 'C61', 'N181', 'N233', 'IND-027', 'N276', 'ATY-012', 'N186', 'C55', 'IND-013', 'C9', 'N255', 'C84', 'N245', 'N166', 'C98', 'N243', 'C67', 'C72', 'N236', 'C3', 'NEG-011', 'C52', 'IND-009', 'IND-020', 'ATY-024', 'ATY-028', 'N251', 'C81', 'N280', 'C22', 'ATY-002']


train_typical_fold5 = ['C116', 'N160', 'C77', 'N285', 'C85', 'C27', 'C13', 'C49', 'N167', 'C96', 'TYP-003', 'TYP-007', 'C157', 'C60', 'C145', 'N203', 'C111', 'TYP-012', 'TYP-023', 'C144', 'N230', 'C115', 'C20', 'C41', 'TYP-017', 'TYP-018', 'TYP-005', 'C80', 'C103', 'C71', 'C25', 'C158', 'C162', 'C124', 'C142', 'TYP-002', 'C112', 'C11', 'C121', 'C79', 'C149', 'TYP-020', 'C117', 'C32', 'N201', 'C161', 'C44', 'TYP-008', 'C143', 'C75', 'C125', 'TYP-015', 'C21', 'N170', 'C39', 'TYP-014', 'C83', 'C131', 'C90', 'N238', 'C12', 'N286', 'C94', 'C74', 'TYP-027', 'C136', 'TYP-010', 'TYP-028', 'TYP-025', 'C14', 'C155', 'C82', 'N190', 'C101', 'C36', 'TYP-024', 'C151', 'C138', 'C23', 'C132', 'C113', 'TYP-019', 'TYP-006', 'C88', 'C15', 'C26', 'C114', 'C110', 'C16', 'C154', 'TYP-004', 'TYP-026', 'C91', 'TYP-030', 'TYP-022', 'TYP-011']
train_negative_fold5 = ['N239', 'C42', 'N233', 'N231', 'IND-029', 'N197', 'N174', 'N270', 'N271', 'N229', 'N279', 'NEG-005', 'C24', 'IND-024', 'C57', 'ATY-002', 'IND-009', 'C126', 'ATY-005', 'ATY-029', 'C128', 'C119', 'C152', 'C54', 'N234', 'IND-022', 'C122', 'IND-027', 'C127', 'ATY-016', 'N295', 'N187', 'NEG-015', 'IND-003', 'N255', 'N245', 'N276', 'ATY-012', 'N195', 'C64', 'N275', 'C9', 'C156', 'NEG-001', 'N251', 'NEG-002', 'C89', 'N165', 'C59', 'N236', 'IND-011', 'N232', 'C73', 'N252', 'IND-013', 'N207', 'NEG-011', 'N220', 'C159', 'ATY-024', 'N243', 'N180', 'N260', 'N240', 'N200', 'N226', 'N221', 'N204', 'ATY-028', 'N274', 'N294', 'N163', 'N291', 'N241', 'IND-014', 'ATY-020', 'ATY-013', 'N172', 'N176', 'C98', 'N272', 'C5', 'C38', 'C61', 'C120', 'N189', 'N209', 'NEG-013', 'C106', 'N188', 'N168', 'N193', 'NEG-003', 'N171', 'C102', 'N198', 'N253', 'IND-025', 'N219', 'C107', 'C148', 'IND-021', 'NEG-004', 'C147', 'C37', 'IND-010', 'C47', 'N264', 'C84', 'N223', 'C67', 'C108', 'C34', 'C62', 'C48', 'ATY-008', 'C97', 'N161', 'C100', 'NEG-006', 'N248', 'IND-018', 'ATY-023', 'N202', 'C134', 'N297', 'N269', 'N183', 'C164', 'N186', 'N259', 'N265', 'IND-008', 'NEG-012', 'C55', 'N181', 'N216', 'ATY-018', 'C28', 'C72', 'C137', 'ATY-026', 'N296', 'C123', 'IND-006', 'N235', 'C65', 'C139', 'ATY-009', 'N177', 'N249', 'C43', 'IND-001', 'C56', 'C76', 'N258', 'C118', 'ATY-001', 'C45', 'ATY-027', 'N242', 'C70', 'C92', 'N280', 'IND-023', 'ATY-021', 'IND-015', 'NEG-009', 'C52', 'NEG-008', 'N184', 'N169', 'C31', 'ATY-030', 'IND-026', 'C3', 'IND-017', 'N299', 'C63', 'C129', 'N164', 'IND-020', 'IND-012', 'C86', 'IND-019', 'C109', 'C8', 'N284', 'C99', 'C141', 'ATY-019', 'IND-004', 'N283', 'C81', 'ATY-022', 'N173', 'N178', 'N212', 'ATY-014', 'N267', 'N182', 'C68', 'N278', 'C22', 'C29', 'NEG-010', 'N194', 'ATY-025', 'IND-016', 'N288', 'N166', 'IND-030']


validation_typical_fold5 = ['N266', 'N213', 'C17', 'TYP-009', 'TYP-031', 'C35', 'TYP-013', 'C146', 'C133', 'C150', 'C33', 'C160', 'C130', 'TYP-029', 'C69', 'N300', 'C19', 'TYP-016', 'C93', 'C163', 'TYP-021', 'C18', 'C51', 'C135']
validation_negative_fold5 = ['N281', 'N237', 'C140', 'N246', 'N263', 'C46', 'N293', 'C50', 'ATY-010', 'C40', 'ATY-011', 'C78', 'N244', 'ATY-017', 'C87', 'N222', 'N289', 'C105', 'N175', 'N257', 'C66', 'IND-002', 'N273', 'NEG-014', 'ATY-015', 'N205', 'N247', 'N290', 'IND-007', 'N228', 'N196', 'N185', 'C30', 'N210', 'C58', 'NEG-007', 'ATY-007', 'ATY-004', 'C104', 'N206', 'N192', 'ATY-003', 'N191', 'N256', 'N261', 'N214', 'N208', 'N227', 'N282', 'N254', 'ATY-006', 'IND-005', 'IND-028']

train_negative_folds = []
train_typical_folds = []

validation_negative_folds = []
validation_typical_folds = []

train_negative_folds.append(train_negative_fold1)
train_negative_folds.append(train_negative_fold2)
train_negative_folds.append(train_negative_fold3)
train_negative_folds.append(train_negative_fold4)
train_negative_folds.append(train_negative_fold5)

validation_negative_folds.append(validation_negative_fold1)
validation_negative_folds.append(validation_negative_fold2)
validation_negative_folds.append(validation_negative_fold3)
validation_negative_folds.append(validation_negative_fold4)
validation_negative_folds.append(validation_negative_fold5)


train_typical_folds.append(train_typical_fold1)
train_typical_folds.append(train_typical_fold2)
train_typical_folds.append(train_typical_fold3)
train_typical_folds.append(train_typical_fold4)
train_typical_folds.append(train_typical_fold5)

validation_typical_folds.append(validation_typical_fold1)
validation_typical_folds.append(validation_typical_fold2)
validation_typical_folds.append(validation_typical_fold3)
validation_typical_folds.append(validation_typical_fold4)
validation_typical_folds.append(validation_typical_fold5)


for i in range(5):
	print('')
	print('Fold Number:', str(i+1))
	print('')
	
	validation_typical_folders = validation_typical_folds[i]
	validation_negative_folders = validation_negative_folds[i]
	
	train_typical_folders = train_typical_folds[i]
	train_negative_folders = train_negative_folds[i]
	
	print('')
	print('Typ Patients in training set: ', len(train_typical_folders), train_typical_folders)
	print('Others Patients in training set: ', len(train_negative_folders), train_negative_folders)
	print('')
	print('Typ Patients in validation set: ', len(validation_typical_folders), validation_typical_folders)
	print('Others Patients in validation set: ', len(validation_negative_folders), validation_negative_folders)
	print('')

	model = get_model()

	#fitting a model
	train_set, validation_set = get_dicts(train_negative_folders, train_typical_folders, 
				                      validation_negative_folders, validation_typical_folders)

	train_df, validation_df = get_data(train_set, validation_set)

	#training the model and saving the label legends
	checkpoint_filepath = 'SavedModels/Resnet/full_covid_others/weights_' + str(i+1)
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath = checkpoint_filepath,
		save_weights_only = True,
		monitor = 'val_accuracy',
		mode = 'max',
		save_best_only = True)

	train_model(model, train_df, validation_df, 15, callbacks = [model_checkpoint_callback])
	#loading the best weights
	model.load_weights(checkpoint_filepath)
	model.save('SavedModels/Resnet/full_covid_others/model_' + str(i+1))

	#getting the predictions of validation set by patient
	print('')
	print('Predicting Validation Patients')
	print('')
	print('')
	print('Predicting Typ Patients')
	print('')
	predictions_by_patient(model, validation_typical_folders)
	print('')
	print('Predicting Others Patients')
	print('')
	predictions_by_patient(model, validation_negative_folders)
	print('')
