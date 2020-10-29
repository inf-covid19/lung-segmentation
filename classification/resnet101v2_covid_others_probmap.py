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
			curr_dir = 'HCPA-ProbMap/Typical/' + p + '/'
		elif p[:3] == 'NEG':
			p = p[4:]
			curr_dir = 'HCPA-ProbMap/Negative/' + p + '/'
		elif p[:3] == 'IND':
			p = p[4:]
			curr_dir = 'HCPA-ProbMap/Indeterminate/' + p + '/'
		elif p[:3] == 'ATY':
			p = p[4:]
			curr_dir = 'HCPA-ProbMap/Atypical/' + p + '/'
		else:
			curr_dir = 'HMV-ProbMap/' + p + '/'
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
		if os.path.isfile('../saved_legends/covid_others_cv.npy'):
			print('loading label legend file')
			class_indices = np.load('../saved_legends/covid_others_cv.npy', allow_pickle=True).item()
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
	np.save('saved_legends/covid_others_cv.npy', train_generator.class_indices)
	    
	return history.history

def get_dicts(train_folders_1, train_folders_2, validation_folders_1, validation_folders_2):
	train_set = dict(zip(train_folders_1, ['negative' for i in range(len(train_folders_1))]))
	train_set.update(dict(zip(train_folders_2, ['typical' for i in range(len(train_folders_2))])))

	validation_set = dict(zip(validation_folders_1, ['negative' for i in range(len(validation_folders_1))]))
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
            	search_folder = "{}/{}".format('HCPA-ProbMap/Typical', img_folder[4:]) #in case of a HCPA Typical patient
            elif img_folder[:3] == 'NEG':
            	search_folder = "{}/{}".format('HCPA-ProbMap/Negative', img_folder[4:]) #in case of a HCPA Negative patient
            elif img_folder[:3] == 'IND':
            	search_folder = "{}/{}".format('HCPA-ProbMap/Indeterminate', img_folder[4:]) #in case of a HCPA Negative patient
            elif img_folder[:3] == 'ATY':
            	search_folder = "{}/{}".format('HCPA-ProbMap/Atypical', img_folder[4:]) #in case of a HCPA Negative patient
            else:
            	search_folder = "{}/{}".format('HMV-ProbMap', img_folder) #in case of a HMV patient
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_typical_fold1 = ['TYP-010', 'C149', 'C125', 'TYP-027', 'C101', 'C44', 'TYP-024', 'TYP-016', 'C17', 'C23', 'TYP-006', 'C21', 'C79', 'C49', 'C85', 'C117', 'TYP-008', 'TYP-025', 'C75', 'C130', 'TYP-023', 'TYP-021', 'C41', 'TYP-030', 'TYP-011', 'C114', 'C135', 'TYP-015', 'C83', 'C143', 'C13', 'C11', 'TYP-018', 'C25', 'C71', 'C142', 'C160', 'C91', 'C32', 'C51', 'C19', 'C124', 'C74', 'C96', 'TYP-009', 'C121', 'C157', 'C116', 'C80', 'TYP-026', 'C88', 'TYP-004', 'C145', 'C158', 'TYP-012', 'C77', 'TYP-013', 'C150', 'TYP-022', 'C69', 'C16', 'TYP-017', 'C93', 'C136', 'TYP-019', 'C163', 'TYP-003', 'C132', 'C39', 'C111', 'C155', 'TYP-005', 'C144', 'TYP-002', 'TYP-020', 'C103', 'C26', 'C94', 'C115', 'C162', 'C133', 'C27', 'C131', 'C138', 'TYP-031', 'C151']
train_negative_fold1 = ['ATY-029', 'IND-020', 'ATY-014', 'ATY-019', 'C29', 'IND-011', 'C127', 'C100', 'C139', 'C8', 'C126', 'C38', 'NEG-008', 'IND-007', 'C52', 'NEG-009', 'C45', 'IND-002', 'C73', 'C122', 'C102', 'C128', 'C57', 'ATY-003', 'IND-018', 'ATY-016', 'C76', 'C47', 'IND-015', 'ATY-023', 'ATY-020', 'C61', 'NEG-015', 'ATY-001', 'ATY-009', 'IND-009', 'C30', 'C9', 'C148', 'ATY-030', 'C134', 'IND-014', 'ATY-027', 'C64', 'ATY-008', 'C137', 'C54', 'C31', 'C59', 'IND-005', 'ATY-028', 'C48', 'C43', 'C159', 'ATY-021', 'C98', 'C67', 'C119', 'C99', 'C104', 'C3', 'C50', 'ATY-026', 'C89', 'C56', 'C152', 'C105', 'IND-019', 'IND-030', 'C156', 'C107', 'C68', 'C37', 'IND-025', 'C164', 'IND-016', 'IND-024', 'IND-004', 'C42', 'NEG-002', 'ATY-022', 'C72', 'C46', 'C129', 'NEG-006', 'IND-006', 'IND-021', 'C22', 'ATY-024', 'IND-003', 'IND-010', 'NEG-013', 'C87', 'C108', 'C28', 'ATY-025', 'NEG-005', 'C141', 'NEG-012', 'ATY-018', 'C78', 'NEG-004', 'C63', 'IND-001', 'C97', 'C62', 'C66', 'ATY-006', 'C65', 'ATY-002', 'ATY-012', 'C109', 'C123', 'IND-013', 'ATY-013', 'ATY-007', 'IND-012', 'IND-017', 'C34', 'NEG-014', 'IND-027', 'ATY-017', 'ATY-011']

validation_typical_fold1 = ['C33', 'C60', 'C112', 'C82', 'C12', 'C161', 'C90', 'C154', 'TYP-014', 'TYP-007', 'C20', 'C36', 'TYP-029', 'C110', 'C15', 'C35', 'C113', 'C14', 'C18', 'TYP-028', 'C146']
validation_negative_fold1 = ['NEG-007', 'C40', 'ATY-015', 'IND-029', 'IND-028', 'NEG-010', 'C106', 'IND-026', 'ATY-010', 'C81', 'C120', 'C70', 'C86', 'C84', 'NEG-003', 'C5', 'IND-022', 'ATY-005', 'NEG-011', 'C58', 'IND-023', 'C55', 'C92', 'NEG-001', 'IND-008', 'C24', 'C118', 'C140', 'ATY-004', 'C147']

train_typical_fold2 = ['C110', 'C149', 'C125', 'TYP-027', 'C18', 'C44', 'TYP-024', 'TYP-016', 'C35', 'C154', 'C23', 'TYP-006', 'C21', 'C79', 'C49', 'C85', 'C117', 'TYP-028', 'TYP-008', 'C20', 'TYP-025', 'C130', 'TYP-023', 'TYP-021', 'C41', 'TYP-030', 'TYP-007', 'TYP-011', 'C135', 'TYP-015', 'C83', 'C15', 'C13', 'C82', 'C90', 'C11', 'C60', 'C25', 'C71', 'C142', 'C36', 'C91', 'C51', 'TYP-014', 'C19', 'C74', 'C96', 'C121', 'C116', 'C80', 'TYP-029', 'C88', 'C145', 'C158', 'TYP-012', 'C77', 'TYP-013', 'C14', 'C150', 'TYP-022', 'C69', 'C112', 'C12', 'C146', 'C93', 'C136', 'C163', 'TYP-003', 'C161', 'C132', 'C39', 'C111', 'TYP-005', 'TYP-002', 'C113', 'C103', 'C26', 'C33', 'C94', 'C115', 'C162', 'C133', 'C27', 'C138', 'TYP-031', 'C151']
train_negative_fold2 = ['ATY-029', 'IND-020', 'ATY-014', 'ATY-019', 'C29', 'IND-011', 'C127', 'C100', 'C139', 'C8', 'C126', 'C38', 'IND-007', 'C52', 'C45', 'IND-026', 'C73', 'C122', 'C102', 'C128', 'C57', 'IND-018', 'ATY-016', 'C76', 'C47', 'C24', 'IND-015', 'C147', 'ATY-004', 'C55', 'C140', 'C61', 'NEG-003', 'ATY-001', 'ATY-009', 'C118', 'C30', 'C84', 'C9', 'IND-014', 'ATY-027', 'C64', 'ATY-008', 'C40', 'IND-022', 'C54', 'C31', 'C59', 'IND-005', 'C48', 'C43', 'NEG-007', 'C159', 'C81', 'ATY-021', 'C98', 'C67', 'C119', 'C99', 'C104', 'C50', 'ATY-026', 'C89', 'C56', 'C152', 'NEG-001', 'C105', 'IND-019', 'IND-008', 'C156', 'C107', 'C68', 'C37', 'IND-025', 'C164', 'IND-016', 'IND-024', 'IND-004', 'C5', 'NEG-011', 'ATY-005', 'NEG-002', 'IND-029', 'C72', 'C46', 'C106', 'IND-023', 'C58', 'C86', 'C129', 'IND-028', 'NEG-006', 'IND-006', 'ATY-015', 'IND-021', 'ATY-024', 'IND-003', 'NEG-013', 'C87', 'C28', 'ATY-010', 'C141', 'NEG-012', 'IND-001', 'C97', 'NEG-010', 'ATY-006', 'C65', 'ATY-002', 'ATY-012', 'C120', 'C109', 'C70', 'C123', 'IND-013', 'C92', 'ATY-013', 'IND-012', 'IND-017', 'C34', 'NEG-014', 'IND-027', 'ATY-017']


validation_typical_fold2 = ['C17', 'C144', 'TYP-019', 'C16', 'C114', 'C131', 'C75', 'C32', 'C157', 'TYP-004', 'C101', 'C160', 'TYP-026', 'C155', 'TYP-020', 'C143', 'TYP-009', 'TYP-017', 'C124', 'TYP-018', 'TYP-010']
validation_negative_fold2 = ['C137', 'ATY-011', 'ATY-028', 'NEG-009', 'ATY-023', 'ATY-020', 'IND-002', 'ATY-022', 'NEG-008', 'C134', 'C3', 'IND-030', 'ATY-030', 'C63', 'C42', 'ATY-025', 'IND-010', 'C62', 'ATY-003', 'ATY-018', 'C148', 'C78', 'C22', 'ATY-007', 'C108', 'IND-009', 'NEG-004', 'C66', 'NEG-005', 'NEG-015']

train_typical_fold3 = ['TYP-010', 'C110', 'C149', 'TYP-027', 'C101', 'C18', 'C44', 'TYP-024', 'TYP-016', 'C35', 'C154', 'C17', 'TYP-006', 'C21', 'C49', 'C85', 'TYP-028', 'TYP-008', 'C20', 'C75', 'C130', 'TYP-021', 'C41', 'TYP-007', 'TYP-011', 'C114', 'C135', 'TYP-015', 'C83', 'C143', 'C15', 'C13', 'C82', 'C90', 'C11', 'C60', 'TYP-018', 'C25', 'C71', 'C142', 'C36', 'C160', 'C32', 'C51', 'TYP-014', 'C19', 'C124', 'C96', 'TYP-009', 'C121', 'C157', 'C116', 'TYP-029', 'TYP-026', 'C88', 'TYP-004', 'C145', 'TYP-012', 'C77', 'TYP-013', 'C14', 'C69', 'C16', 'C112', 'TYP-017', 'C12', 'C146', 'TYP-019', 'TYP-003', 'C161', 'C132', 'C111', 'C155', 'C144', 'C113', 'TYP-020', 'C103', 'C26', 'C33', 'C94', 'C115', 'C162', 'C27', 'C131', 'C138', 'C151']
train_negative_fold3 = ['ATY-029', 'IND-020', 'ATY-014', 'ATY-019', 'C29', 'IND-011', 'C100', 'C8', 'C38', 'NEG-008', 'C52', 'NEG-009', 'C45', 'IND-026', 'IND-002', 'C73', 'C102', 'C128', 'ATY-003', 'ATY-016', 'C76', 'C47', 'C24', 'C147', 'ATY-004', 'C55', 'C140', 'ATY-023', 'ATY-020', 'C61', 'NEG-003', 'NEG-015', 'ATY-009', 'C118', 'IND-009', 'C30', 'C84', 'C9', 'C148', 'ATY-030', 'C134', 'IND-014', 'ATY-027', 'C40', 'C137', 'IND-022', 'C54', 'C31', 'IND-005', 'ATY-028', 'C48', 'C43', 'NEG-007', 'C159', 'C81', 'ATY-021', 'C67', 'C3', 'ATY-026', 'C56', 'C152', 'NEG-001', 'IND-019', 'IND-030', 'IND-008', 'C156', 'C68', 'C37', 'IND-025', 'C164', 'IND-024', 'IND-004', 'C42', 'C5', 'NEG-011', 'ATY-005', 'NEG-002', 'IND-029', 'ATY-022', 'C46', 'C106', 'IND-023', 'C58', 'C86', 'C129', 'IND-028', 'IND-006', 'ATY-015', 'IND-021', 'C22', 'IND-003', 'IND-010', 'NEG-013', 'C87', 'C108', 'ATY-025', 'ATY-010', 'NEG-005', 'C141', 'NEG-012', 'ATY-018', 'C78', 'NEG-004', 'C63', 'IND-001', 'C97', 'NEG-010', 'C62', 'C66', 'ATY-006', 'C65', 'ATY-002', 'C120', 'C70', 'IND-013', 'C92', 'ATY-013', 'ATY-007', 'IND-012', 'IND-017', 'C34', 'NEG-014', 'ATY-011']

validation_typical_fold3 = ['C39', 'TYP-022', 'C117', 'C23', 'TYP-031', 'TYP-030', 'TYP-023', 'C93', 'TYP-005', 'C150', 'TYP-025', 'C74', 'C158', 'C79', 'C163', 'C125', 'C133', 'C91', 'TYP-002', 'C80', 'C136']
validation_negative_fold3 = ['IND-015', 'C72', 'C139', 'IND-018', 'ATY-001', 'NEG-006', 'C98', 'ATY-024', 'C126', 'ATY-008', 'IND-027', 'C28', 'C104', 'C59', 'C127', 'IND-016', 'IND-007', 'C107', 'C105', 'ATY-017', 'C123', 'C64', 'C99', 'C50', 'ATY-012', 'C57', 'C119', 'C109', 'C122', 'C89']

train_typical_fold4 = ['TYP-010', 'C110', 'C149', 'C125', 'C101', 'C18', 'C44', 'TYP-016', 'C35', 'C154', 'C17', 'C23', 'C21', 'C79', 'C49', 'C117', 'TYP-028', 'C20', 'TYP-025', 'C75', 'C130', 'TYP-023', 'TYP-021', 'TYP-030', 'TYP-007', 'TYP-011', 'C114', 'C135', 'C83', 'C143', 'C15', 'C82', 'C90', 'C60', 'TYP-018', 'C71', 'C36', 'C160', 'C91', 'C32', 'C51', 'TYP-014', 'C19', 'C124', 'C74', 'C96', 'TYP-009', 'C121', 'C157', 'C80', 'TYP-029', 'TYP-026', 'C88', 'TYP-004', 'C158', 'TYP-013', 'C14', 'C150', 'TYP-022', 'C16', 'C112', 'TYP-017', 'C12', 'C146', 'C93', 'C136', 'TYP-019', 'C163', 'TYP-003', 'C161', 'C132', 'C39', 'C155', 'TYP-005', 'C144', 'TYP-002', 'C113', 'TYP-020', 'C26', 'C33', 'C94', 'C115', 'C133', 'C27', 'C131', 'TYP-031']
train_negative_fold4 = ['ATY-014', 'ATY-019', 'C127', 'C100', 'C139', 'C8', 'C126', 'C38', 'NEG-008', 'IND-007', 'NEG-009', 'IND-026', 'IND-002', 'C73', 'C122', 'C57', 'ATY-003', 'IND-018', 'ATY-016', 'C76', 'C47', 'C24', 'IND-015', 'C147', 'ATY-004', 'C55', 'C140', 'ATY-023', 'ATY-020', 'C61', 'NEG-003', 'NEG-015', 'ATY-001', 'ATY-009', 'C118', 'IND-009', 'C30', 'C84', 'C9', 'C148', 'ATY-030', 'C134', 'C64', 'ATY-008', 'C40', 'C137', 'IND-022', 'C31', 'C59', 'ATY-028', 'NEG-007', 'C159', 'C81', 'C98', 'C67', 'C119', 'C99', 'C104', 'C3', 'C50', 'C89', 'NEG-001', 'C105', 'IND-030', 'IND-008', 'C107', 'C37', 'C164', 'IND-016', 'IND-004', 'C42', 'C5', 'NEG-011', 'ATY-005', 'NEG-002', 'IND-029', 'ATY-022', 'C72', 'C46', 'C106', 'IND-023', 'C58', 'C86', 'C129', 'IND-028', 'NEG-006', 'ATY-015', 'IND-021', 'C22', 'ATY-024', 'IND-003', 'IND-010', 'C108', 'C28', 'ATY-025', 'ATY-010', 'NEG-005', 'NEG-012', 'ATY-018', 'C78', 'NEG-004', 'C63', 'C97', 'NEG-010', 'C62', 'C66', 'C65', 'ATY-012', 'C120', 'C109', 'C70', 'C123', 'IND-013', 'C92', 'ATY-013', 'ATY-007', 'IND-012', 'IND-017', 'C34', 'NEG-014', 'IND-027', 'ATY-017', 'ATY-011']


validation_typical_fold4 = ['TYP-006', 'C77', 'C116', 'C111', 'TYP-008', 'C162', 'C69', 'C138', 'TYP-027', 'C103', 'TYP-012', 'C13', 'C41', 'TYP-015', 'C85', 'C142', 'C25', 'C11', 'C151', 'C145', 'TYP-024']
validation_negative_fold4 = ['NEG-013', 'C52', 'C152', 'C56', 'C48', 'ATY-027', 'C54', 'ATY-021', 'IND-005', 'IND-020', 'ATY-002', 'IND-006', 'C45', 'IND-025', 'C87', 'C102', 'ATY-026', 'C43', 'IND-019', 'IND-024', 'C128', 'ATY-029', 'C68', 'ATY-006', 'IND-014', 'C156', 'IND-001', 'C29', 'C141', 'IND-011']


train_typical_fold5 = ['TYP-010', 'C110', 'C125', 'TYP-027', 'C101', 'C18', 'TYP-024', 'C35', 'C154', 'C17', 'C23', 'TYP-006', 'C79', 'C85', 'C117', 'TYP-028', 'TYP-008', 'C20', 'TYP-025', 'C75', 'TYP-023', 'C41', 'TYP-030', 'TYP-007', 'C114', 'TYP-015', 'C143', 'C15', 'C13', 'C82', 'C90', 'C11', 'C60', 'TYP-018', 'C25', 'C142', 'C36', 'C160', 'C91', 'C32', 'TYP-014', 'C124', 'C74', 'TYP-009', 'C157', 'C116', 'C80', 'TYP-029', 'TYP-026', 'TYP-004', 'C145', 'C158', 'TYP-012', 'C77', 'C14', 'C150', 'TYP-022', 'C69', 'C16', 'C112', 'TYP-017', 'C12', 'C146', 'C93', 'C136', 'TYP-019', 'C163', 'C161', 'C39', 'C111', 'C155', 'TYP-005', 'C144', 'TYP-002', 'C113', 'TYP-020', 'C103', 'C33', 'C162', 'C133', 'C131', 'C138', 'TYP-031', 'C151']
train_negative_fold5 = ['ATY-029', 'IND-020', 'C29', 'IND-011', 'C127', 'C139', 'C126', 'NEG-008', 'IND-007', 'C52', 'NEG-009', 'C45', 'IND-026', 'IND-002', 'C122', 'C102', 'C128', 'C57', 'ATY-003', 'IND-018', 'C24', 'IND-015', 'C147', 'ATY-004', 'C55', 'C140', 'ATY-023', 'ATY-020', 'NEG-003', 'NEG-015', 'ATY-001', 'C118', 'IND-009', 'C84', 'C148', 'ATY-030', 'C134', 'IND-014', 'ATY-027', 'C64', 'ATY-008', 'C40', 'C137', 'IND-022', 'C54', 'C59', 'IND-005', 'ATY-028', 'C48', 'C43', 'NEG-007', 'C81', 'ATY-021', 'C98', 'C119', 'C99', 'C104', 'C3', 'C50', 'ATY-026', 'C89', 'C56', 'C152', 'NEG-001', 'C105', 'IND-019', 'IND-030', 'IND-008', 'C156', 'C107', 'C68', 'IND-025', 'IND-016', 'IND-024', 'C42', 'C5', 'NEG-011', 'ATY-005', 'IND-029', 'ATY-022', 'C72', 'C106', 'IND-023', 'C58', 'C86', 'IND-028', 'NEG-006', 'IND-006', 'ATY-015', 'C22', 'ATY-024', 'IND-010', 'NEG-013', 'C87', 'C108', 'C28', 'ATY-025', 'ATY-010', 'NEG-005', 'C141', 'ATY-018', 'C78', 'NEG-004', 'C63', 'IND-001', 'NEG-010', 'C62', 'C66', 'ATY-006', 'ATY-002', 'ATY-012', 'C120', 'C109', 'C70', 'C123', 'C92', 'ATY-007', 'IND-027', 'ATY-017', 'ATY-011']


validation_typical_fold5 = ['C132', 'C149', 'TYP-016', 'TYP-011', 'C135', 'C21', 'C71', 'TYP-003', 'C121', 'C130', 'C115', 'TYP-021', 'C27', 'C51', 'TYP-013', 'C44', 'C94', 'C26', 'C96', 'C88', 'C83', 'C19', 'C49']
validation_negative_fold5 = ['ATY-016', 'IND-013', 'C67', 'NEG-002', 'C30', 'C97', 'C73', 'C9', 'IND-017', 'C159', 'IND-021', 'C61', 'ATY-009', 'C37', 'IND-003', 'NEG-012', 'C65', 'ATY-019', 'ATY-014', 'C34', 'C129', 'ATY-013', 'C31', 'C76', 'NEG-014', 'IND-004', 'C47', 'IND-012', 'C46', 'C100', 'C8', 'C164', 'C38']

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
	print('Typical Patients in training set: ', len(train_typical_folders), train_typical_folders)
	print('Negative Patients in training set: ', len(train_negative_folders), train_negative_folders)
	print('')
	print('Typical Patients in validation set: ', len(validation_typical_folders), validation_typical_folders)
	print('Negative Patients in validation set: ', len(validation_negative_folders), validation_negative_folders)
	print('')

	model = get_model()

	#fitting a model
	train_set, validation_set = get_dicts(train_negative_folders, train_typical_folders, 
				                      validation_negative_folders, validation_typical_folders)

	train_df, validation_df = get_data(train_set, validation_set)

	#training the model and saving the label legends
	checkpoint_filepath = 'SavedModels/ProbMap/Resnet/Resnet101V2Covid_OthersProbMap/weights_' + str(i+1)
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath = checkpoint_filepath,
		save_weights_only = True,
		monitor = 'val_accuracy',
		mode = 'max',
		save_best_only = True)

	train_model(model, train_df, validation_df, 15, callbacks = [model_checkpoint_callback])
	#loading the best weights
	model.load_weights(checkpoint_filepath)
	model.save('SavedModels/Resnet/Resnet101V2Covid_OthersProbMap/model_' + str(i+1))
	
	#model = models.load_model('SavedModels/Resnet/Resnet50Covid_OthersProbMap/model_' + str(i+1))

	#getting the predictions of validation set by patient
	print('')
	print('Predicting Validation Patients')
	print('')
	print('')
	print('Predicting Typical Patients')
	print('')
	predictions_by_patient(model, validation_typical_folders)
	print('')
	print('Predicting Negative Patients')
	print('')
	predictions_by_patient(model, validation_negative_folders)
	print('')
