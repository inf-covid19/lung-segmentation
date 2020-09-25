# -*- coding: utf-8 -*-
#resnet152v2 using adam optimizer

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
from keras.applications import ResNet152V2
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

	conv_base = ResNet152V2(weights='imagenet', include_top=False, input_shape=(width,height,3))
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
                if os.path.isfile('saved_legends/resnet152v2_2.npy'):
                        print('loading label legend file')
                        class_indices = np.load('saved_legends/resnet152v2_2.npy', allow_pickle=True).item()
                        class_indices = dict((v,k) for k,v in class_indices.items())
                        test_df['category'] = test_df['category'].replace(class_indices)
                print(test_df['category'].value_counts())
                print('')
                test_df.to_csv('teste.csv')


def train_model(model, train_df, validation_df, epochs, callbacks=[]):
	batch_size = 5
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
	np.save('saved_legends/resnet152v2_2.npy', train_generator.class_indices)
	    
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

train_typical_fold1 = ['C74', 'C133', 'C23', 'C93', 'TYP-013', 'C85', 'C69', 'C158', 'C111', 'C19', 'TYP-029', 'C125', 'C17', 'C60', 'C71', 'C150', 'C25', 
'TYP-009', 'TYP-017', 'C136', 'TYP-019', 'TYP-031', 'TYP-003', 'C83', 'C130', 'C41', 'TYP-006', 'C20', 'C124', 'C103', 'C145', 'C160', 'C44', 'C116', 
'TYP-011', 'C112', 'C138', 'C12', 'C113', 'C51', 'C135', 'C16', 'C143', 'C35', 'C14', 'TYP-016', 'C18', 'C144', 'TYP-030', 'C115', 'TYP-020', 'C157', 
'C101', 'C132', 'C88', 'C15', 'C151', 'C121', 'C21', 'TYP-015', 'TYP-018', 'TYP-022', 'C13', 'TYP-026', 'TYP-014', 'C27', 'C149', 'C79', 'C90', 'C163', 
'TYP-025', 'C146', 'TYP-010', 'TYP-023', 'C49', 'C82', 'C110', 'C80', 'C114', 'C142', 'TYP-012', 'C32', 'TYP-028', 'C155', 'C96', 'C117']
train_negative_fold1 = ['NEG-006', 'NEG-008', 'C137', 'C42', 'C3', 'C100', 'C29', 'NEG-010', 'C78', 'C87', 'C24', 'C50', 'C1', 'NEG-012', 'NEG-009', 'C22', 
'C104', 'C126', 'C46', 'C63', 'C102', 'C152', 'C159', 'NEG-013', 'C5', 'C66', 'C92', 'NEG-005', 'C147', 'NEG-003', 'C62', 'NEG-015', 'C76', 'C108', 
'NEG-014', 'NEG-004', 'NEG-001', 'C105']

validation_typical_fold1 = ['C75', 'C91', 'TYP-004', 'TYP-021', 'C39', 'TYP-002', 'C162', 'TYP-008', 'C26', 'C36', 'C33', 'C161', 'C131', 'C77', 'TYP-024', 
'C94', 'C154', 'TYP-027', 'TYP-007', 'TYP-005', 'C11']
validation_negative_fold1 = ['C52', 'NEG-007', 'NEG-011', 'C120', 'C89', 'C86', 'C106', 'NEG-002', 'C61']

train_typical_fold2 = ['C74', 'C133', 'C23', 'TYP-013', 'C85', 'C69', 'C158', 'TYP-029', 'C19', 'C125', 'TYP-008', 'C17', 'C60', 'C75', 'TYP-027', 'C150', 
'C25', 'TYP-009', 'TYP-017', 'C136', 'TYP-019', 'C83', 'C130', 'C41', 'C77', 'TYP-006', 'C20', 'TYP-004', 'C124', 'C103', 'C145', 'C33', 'C44', 'C112', 
'C138', 'C161', 'C12', 'C131', 'C113', 'C51', 'C135', 'C16', 'C143', 'C26', 'C35', 'C39', 'C91', 'TYP-016', 'C36', 'C18', 'C11', 'TYP-002', 'C101', 'C88', 
'C15', 'C151', 'TYP-024', 'C21', 'TYP-015', 'C154', 'TYP-022', 'C13', 'TYP-026', 'TYP-014', 'C149', 'C79', 'C90', 'C163', 'C162', 'C94', 'TYP-010', 
'TYP-023', 'C49', 'TYP-005', 'C82', 'C110', 'TYP-007', 'C80', 'C114', 'C142', 'C32', 'TYP-028', 'C155', 'C96', 'TYP-021', 'C117']
train_negative_fold2 = ['NEG-006', 'NEG-008', 'C137', 'C42', 'C86', 'C3', 'NEG-011', 'C52', 'C29', 'NEG-010', 'C78', 'C50', 'C1', 'NEG-009', 'C120', 'C22', 
'NEG-002', 'NEG-007', 'C104', 'C126', 'C106', 'C46', 'C63', 'C102', 'C159', 'NEG-013', 'C5', 'C66', 'C92', 'NEG-005', 'NEG-003', 'C89', 'C62', 'C76', 
'NEG-014', 'NEG-001', 'C61', 'C105']

validation_typical_fold2 = ['C111', 'TYP-003', 'C115', 'C14', 'C71', 'C27', 'C157', 'C160', 'TYP-018', 'C116', 'C121', 'TYP-031', 'TYP-020', 'C146', 
'TYP-011', 'C144', 'TYP-025', 'TYP-012', 'C132', 'TYP-030', 'C93']
validation_negative_fold2 = ['NEG-004', 'C147', 'C24', 'C108', 'NEG-015', 'C100', 'C152', 'C87', 'NEG-012']

train_typical_fold3 = ['C74', 'C133', 'C23', 'C93', 'TYP-013', 'C85', 'C69', 'C111', 'C19', 'TYP-029', 'C125', 'TYP-008', 'C17', 'C60', 'C75', 'TYP-027', 
'C71', 'C150', 'C25', 'TYP-009', 'C136', 'TYP-031', 'TYP-003', 'C130', 'C77', 'TYP-006', 'TYP-004', 'C124', 'C145', 'C160', 'C33', 'C116', 'TYP-011', 
'C138', 'C161', 'C131', 'C113', 'C51', 'C135', 'C16', 'C26', 'C39', 'C91', 'C14', 'C36', 'C11', 'C144', 'TYP-030', 'C115', 'TYP-002', 'TYP-020', 'C157', 
'C132', 'C15', 'C151', 'TYP-024', 'C121', 'C21', 'TYP-015', 'C154', 'TYP-018', 'TYP-022', 'C13', 'TYP-026', 'TYP-014', 'C27', 'C149', 'C79', 'C163', 
'TYP-025', 'C146', 'C162', 'C94', 'C49', 'TYP-005', 'C110', 'TYP-007', 'C80', 'C142', 'TYP-012', 'C32', 'TYP-028', 'C155', 'C96', 'TYP-021', 'C117']
train_negative_fold3 = ['NEG-006', 'NEG-008', 'C137', 'C42', 'C86', 'C3', 'C100', 'NEG-011', 'C52', 'C78', 'C87', 'C24', 'NEG-012', 'NEG-009', 'C120', 
'C22', 'NEG-002', 'NEG-007', 'C104', 'C126', 'C106', 'C63', 'C152', 'NEG-013', 'C5', 'C66', 'C92', 'NEG-005', 'C147', 'NEG-003', 'C89', 'C62', 'NEG-015', 
'C108', 'NEG-004', 'NEG-001', 'C61', 'C105']

validation_typical_fold3 = ['C143', 'TYP-019', 'TYP-023', 'TYP-010', 'C83', 'TYP-016', 'C103', 'C88', 'C90', 'C112', 'C35', 'C101', 'C158', 'C20', 'C114', 
'C12', 'C82', 'C41', 'C44', 'C18', 'TYP-017']
validation_negative_fold3 = ['C76', 'C159', 'C29', 'NEG-010', 'C102', 'C50', 'NEG-014', 'C46']

train_typical_fold4 = ['C93', 'TYP-013', 'C158', 'C111', 'C19', 'TYP-029', 'C125', 'TYP-008', 'C75', 'TYP-027', 'C71', 'C25', 'TYP-009', 'TYP-017', 
'TYP-019', 'TYP-031', 'TYP-003', 'C83', 'C130', 'C41', 'C77', 'C20', 'TYP-004', 'C124', 'C103', 'C160', 'C33', 'C44', 'C116', 'TYP-011', 'C112', 'C138', 
'C161', 'C12', 'C131', 'C135', 'C16', 'C143', 'C26', 'C35', 'C39', 'C91', 'C14', 'TYP-016', 'C36', 'C18', 'C11', 'C144', 'TYP-030', 'C115', 'TYP-002', 
'TYP-020', 'C157', 'C101', 'C132', 'C88', 'C15', 'C151', 'TYP-024', 'C121', 'C21', 'TYP-015', 'C154', 'TYP-018', 'TYP-022', 'TYP-026', 'C27', 'C149', 
'C90', 'C163', 'TYP-025', 'C146', 'C162', 'C94', 'TYP-010', 'TYP-023', 'TYP-005', 'C82', 'C110', 'TYP-007', 'C114', 'C142', 'TYP-012', 'C155', 'TYP-021', 
'C117']
train_negative_fold4 = ['C137', 'C42', 'C86', 'C100', 'NEG-011', 'C52', 'C29', 'NEG-010', 'C78', 'C87', 'C24', 'C50', 'C1', 'NEG-012', 'NEG-009', 'C120', 
'C22', 'NEG-002', 'NEG-007', 'C126', 'C106', 'C46', 'C63', 'C102', 'C152', 'C159', 'C5', 'C66', 'NEG-005', 'C147', 'C89', 'NEG-015', 'C76', 'C108', 
'NEG-014', 'NEG-004', 'NEG-001', 'C61']

validation_typical_fold4 = ['C136', 'TYP-014', 'C74', 'C60', 'C79', 'C96', 'C32', 'C13', 'TYP-006', 'C51', 'TYP-028', 'C113', 'C133', 'C49', 'C145', 'C17', 
'C150', 'C85', 'C80', 'C23', 'C69']
validation_negative_fold4 = ['NEG-006', 'C104', 'NEG-008', 'NEG-003', 'C3', 'C92', 'NEG-013', 'C105', 'C62']

train_typical_fold5 = ['C74', 'C133', 'C23', 'C93', 'C85', 'C69', 'C158', 'C111', 'TYP-008', 'C17', 'C60', 'C75', 'TYP-027', 'C71', 'C150', 'TYP-017', 
'C136', 'TYP-019', 'TYP-031', 'TYP-003', 'C83', 'C41', 'C77', 'TYP-006', 'C20', 'TYP-004', 'C103', 'C145', 'C160', 'C33', 'C44', 'C116', 'TYP-011', 'C112', 
'C161', 'C12', 'C131', 'C113', 'C51', 'C143', 'C26', 'C35', 'C39', 'C91', 'C14', 'TYP-016', 'C36', 'C18', 'C11', 'C144', 'TYP-030', 'C115', 'TYP-002', 
'TYP-020', 'C157', 'C101', 'C132', 'C88', 'TYP-024', 'C121', 'C154', 'TYP-018', 'C13', 'TYP-014', 'C27', 'C79', 'C90', 'TYP-025', 'C146', 'C162', 'C94', 
'TYP-010', 'TYP-023', 'C49', 'TYP-005', 'C82', 'TYP-007', 'C80', 'C114', 'TYP-012', 'C32', 'TYP-028', 'C96', 'TYP-021']
train_negative_fold5 = ['NEG-006', 'NEG-008', 'C86', 'C3', 'C100', 'NEG-011', 'C52', 'C29', 'NEG-010', 'C87', 'C24', 'C50', 'C1', 'NEG-012', 'C120', 
'NEG-002', 'NEG-007', 'C104', 'C106', 'C46', 'C102', 'C152', 'C159', 'NEG-013', 'C92', 'C147', 'NEG-003', 'C89', 'C62', 'NEG-015', 'C76', 'C108', 
'NEG-014', 'NEG-004', 'C61', 'C105']

validation_typical_fold5 = ['TYP-022', 'C130', 'C110', 'C163', 'C16', 'TYP-009', 'TYP-029', 'C15', 'C142', 'C19', 'C138', 'C124', 'C151', 'C25', 'TYP-026', 
'C21', 'TYP-015', 'C135', 'C117', 'C149', 'C125', 'TYP-013', 'C155']
validation_negative_fold5 = ['C126', 'NEG-009', 'C42', 'C63', 'C5', 'C137', 'C66', 'NEG-001', 'C78', 'C22', 'NEG-005']


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
	checkpoint_filepath = 'Resnet152V2_2/weights_' + str(i+1)
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath = checkpoint_filepath,
		save_weights_only = True,
		monitor = 'val_accuracy',
		mode = 'max',
		save_best_only = True)

	train_model(model, train_df, validation_df, 15, callbacks = [model_checkpoint_callback])
	#loading the best weights
	model.load_weights(checkpoint_filepath)
	model.save('Resnet152V2_2/model_' + str(i+1))

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
