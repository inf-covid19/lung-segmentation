# Removing tensorflow warnings
import matplotlib.cm as cm
from IPython.display import Image
import warnings
try:
    warnings.simplefilter('ignore')
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
finally:
    pass

# -*- coding: utf-8 -*-
'''
instanciating the desired model and doing predictions for some test patients form HMV
'''
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import os

import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras import models
from keras.models import load_model, Model
from keras import optimizers
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras import applications
from keras.applications import VGG16
from keras.applications import VGG19
import tensorflow as tf

import glob
from datetime import datetime

import time
import argparse
import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from PIL import Image
from keras import backend as K
from keras.utils import plot_model

# tf.compat.v1.disable_eager_execution()
# tf.enable_eager_execution()

EXAM_SLICE = 100
width = 512
height = 512

# model_filename = '/home/chicobentojr/Desktop/cidia/model/binary_model'
# model_filename = '/home/chicobentojr/Desktop/cidia/model/ternary_model'
# model_filename = '/home/chicobentojr/Desktop/cidia/model/model_1'
# model_filename = '/home/chicobentojr/Desktop/cidia/model/balanced_weights'
model_filename = 'cidia-lung-model_1.h5'
# slice_filename = '/home/chicobentojr/Desktop/cidia/data/C143/-slice255..png'
slice_filename = '/home/chicobentojr/Desktop/cidia/data/C11/-slice260..png'
# slice_filename = '/home/chicobentojr/Desktop/cidia/data/C33/-slice276..png'

# model_filename = '/home/chicobentojr/Workspace/UFRGS/image-classifier/keras/scripts/models/cat-dog.model'
# slice_filename = '/home/chicobentojr/Workspace/UFRGS/image-classifier/keras/scripts/images/example/cat/cat.jpg'
# slice_filename = '/home/chicobentojr/Workspace/UFRGS/image-classifier/keras/scripts/images/example/cat/_111434467_gettyimages-1143489763.jpg'

# model = VGG19()

# CUSTOM ATTEMPT

# base_model = load_model(model_filename)

# print('base_model summary')
# base_model.summary()
# print()
# print('base model')
# print(base_model)
# print()

# print('base model layers')
# print(base_model.layers)
# print()

# vgg19_model = base_model.get_layer('vgg19')

# n_model = vgg19_model.input

# print('vgg19 summary')
# vgg19_model.summary()
# print()

# for layer in vgg19_model.layers[1:]:
#     print('base model layer', layer.name)
#     n_model = vgg19_model.get_layer(layer.name)(n_model)

# for layer in base_model.layers[1:]:
#     print('after base model layer', layer.name)
#     n_model = base_model.get_layer(layer.name)(n_model)


# model = Model(inputs=base_model.get_layer('vgg19').input, outputs=n_model)
# model = Model(inputs=vgg19_model.input, outputs=n_model)

# mergedModel = Model(inputs=[vgg19_model.input], outputs=n_model)
# model = mergedModel


def get_model():
    base_model = VGG19(weights='imagenet', include_top=False,
                       input_shape=(width, height, 3))

    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    # x = Dropout(0.4)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.4)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.4)(x)
    # preds = Dense(3, activation='softmax')(x)

    # model = Model(inputs=base_model.input, outputs=preds)

    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation="relu")(x)
    # x = Dropout(0.5)(x)
    # x = Dense(512, activation="relu")(x)
    # x = Dropout(0.5)(x)

    # # 2 new Dense Layers
    # # x = Dense(1024,activation='relu')(x)
    # # x = Dense(1024,activation='relu')(x)

    # # 2 new Dense Layers
    # # x = Dense(1024,activation='relu')(x)
    # # x = Dense(1024,activation='relu')(x)

    # # 4 new Dense Layers
    # # x = Dense(1024,activation='relu')(x)
    # # x = Dense(1024,activation='relu')(x)
    # # x = Dense(1024,activation='relu')(x)
    # # x = Dense(1024,activation='relu')(x)

    # x = Dense(256, activation="relu")(x)
    # x = Dropout(0.5)(x)
    # preds = Dense(3, activation="softmax")(x)

    # model = Model(inputs=base_model.input, outputs=preds)

    # for layer in base_model.layers:
    #     layer.trainable = False

    # return model
    # conv_base = VGG19(weights='imagenet', include_top=False,
    #                   input_shape=(width, height, 3))
    # conv_base.trainable = True

    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    # x = Dropout(0.4)(x)
    preds = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=preds)

    return model


    # model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    #               loss='binary_crossentropy', metrics=['accuracy'])
    # return model
# model = get_model()
# END CUSTOM ATTEMPT
model = load_model(model_filename)
print('model summary')
model.summary()
print()

# plot_model(load_model(model_filename),
#            to_file='model-topology.png', show_shapes=True)

# print('model')
# print(model)

# exit()


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        print('last_conv_layer_output', last_conv_layer_output.shape)
        print(last_conv_layer_output)
        print()
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        # top_pred_index = 0
        print('top pred index', top_pred_index)

        # top_pred_index = 0
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    print('grads', grads.shape)
    print(grads)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    print('pooled grads', pooled_grads.shape)
    print(pooled_grads)
    print()

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    print('last_conv_layer_output', last_conv_layer_output.shape)
    print('last_conv_layer_output max', last_conv_layer_output.max())
    print(last_conv_layer_output)
    print()

    heatmap = np.mean(last_conv_layer_output, axis=-1)

    print('mean heatmap')
    print(heatmap)
    print()

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


# model = load_model(model_filename)
# Prepare image
img_array = preprocess_input(get_img_array(
    slice_filename, size=(width, height)))

img_raw_prep = keras.preprocessing.image.array_to_img(img_array[0])
img_raw_prep.save('raw_prep.png')

img_raw = keras.preprocessing.image.array_to_img(
    get_img_array(slice_filename, size=(width, height))[0])
img_raw.save('raw.png')
# Make model
# model = model_builder(weights="imagenet")
# last_conv_layer_name = "vgg19"
# last_conv_layer_name = "block5_conv4"
last_conv_layer_name = "block5_pool"
# last_conv_layer_name = "flatten"
# last_conv_layer_name = "block5_conv1"
# last_conv_layer_name = "conv_pw_13_relu"
# last_conv_layer_name = "global_average_pooling2d"


# Print what the top predicted class is
preds = model.predict(img_array)
print('preds', preds)

# raw_preds = base_model.predict(img_array)
# print('raw model preds', raw_preds)
# print("Predicted:", decode_predictions(preds, top=1)[0])


classifier_layer_names = [
    # "avg_pool",
    # "block5_conv2",
    # "block5_conv3",
    # "block5_conv4",
    # "block5_pool",
    # "flatten",
    # "fc1",
    # "fc2",
    # "predictions",

    # "block5_conv2",
    # "block5_conv3",
    # "block5_conv4",
    # "block5_pool",
    # "flatten",
    # "dense",
    # "dropout",
    # "dense_1",
    # "dropout_1",
    # "dense_2",

    # get_model() local (lung seg binary)
    # "block5_conv2",
    # "block5_conv3",
    # "block5_conv4",
    # "block5_pool",
    # "global_average_pooling2d",
    "flatten",
    "dense",
    "dropout",
    "dense_1",
    "dropout_1",
    # "dropout_2",
    "dense_2",


    # Cat x Dog  model
    # "global_average_pooling2d_1",
    # "dense_1",
    # "dropout_1",
    # "dense_2",
    # "dropout_2",
    # "dense_3",
    # "dropout_3",
    # "dense_4",
]

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
)

print()
print("heatmap", heatmap.shape)
print(heatmap)

# Display heatmap
plt.matshow(heatmap)
plt.savefig('heat.jpg')
# plt.show()


# Showing superimposed
img = keras.preprocessing.image.load_img(slice_filename)
img = keras.preprocessing.image.img_to_array(img)

# We rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)

# We use jet colormap to colorize heatmap
jet = cm.get_cmap("jet")

# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# We create an image with RGB colorized heatmap
jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.7 + img * 0.3
# superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

# Save the superimposed image
save_path = "slice_cam.jpg"
superimposed_img.save(save_path)

# Display Grad CAM
# display(Image(save_path))

exit(0)

img = image.load_img(slice_filename, target_size=(224, 224))

# `x` is a float32 Numpy array of shape (224, 224, 3)
vgg19_model = image.img_to_array(img)

# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
vgg19_model = np.expand_dims(vgg19_model, axis=0)

# Finally we preprocess the batch
# (this does channel-wise color normalization)
vgg19_model = preprocess_input(vgg19_model)

preds = model.predict(vgg19_model)

print('preds')
print(preds)

# conv_model = model.get_layer('vgg19')

# print('conv model')
# print(conv_model)

# print("conv model summary")
# conv_model.summary()
# print()

# for layer in model.layers:
#     conv_model.add(layer)

# print("conv model summary AFTER")
# conv_model.summary()
# print()

conv_layer = model.get_layer('block5_conv1')

heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

# Get gradient of the winner class w.r.t. the output of the (last) conv. layer
with tf.GradientTape() as gtape:
    conv_output, predictions = heatmap_model(vgg19_model)
    loss = predictions[:, np.argmax(predictions[0])]
    grads = gtape.gradient(loss, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)

print('heatmap')
print(heatmap)

exit(0)

max_pred = np.argmax(preds[0])
label_index = max_pred

pred_output = model.output[:, label_index]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = model.get_layer('vgg19')
# last_conv_layer = model.get_layer('block5_conv3')
print('last conv layer')
print(last_conv_layer)

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(pred_output, last_conv_layer.output)[0]

print('grads')
print(grads)

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

print('pooled_grads')
print(pooled_grads)


# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function(
    [model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([vgg19_model])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
# for i in range(512):

# print('layer shape', conv_layer_output_value.shape)

for i in range(conv_layer_output_value.shape[-1]):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation

# print('conv shape', conv_layer_output_value.shape)

heatmap = np.mean(conv_layer_output_value, axis=-1)
print('heatmap', heatmap.max())
print(heatmap)

exit(0)


def get_model():

    conv_base = VGG19(weights='imagenet', include_top=False,
                      input_shape=(width, height, 3))
    conv_base.trainable = True

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


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


def predictions_by_patient(model, patients):
    # this function will print the count of slices classified for each patient in a list of patients,
    # it supposes that the label legend of the model in question is saved
    # the image generators are generated on demand, which might be slow, consider changing it to
    # reciving a loaded generator if applying the function on validation set's patients
    # the method used to get the prediction for each slice might not work for classifiers with more
    # than 2 classes.

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
        test_filenames = imgs_filename[(
            len(imgs_filename)-EXAM_SLICE)//2:(len(imgs_filename)+EXAM_SLICE)//2]
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

    for i, test_df in enumerate(results):
        print('Patient number: ', patients[i])
        if os.path.isfile('saved_legends/binary_balanced_legend.npy'):
            print('loading label legend file')
            class_indices = np.load(
                'saved_legends/binary_balanced_legend.npy', allow_pickle=True).item()
            class_indices = dict((v, k) for k, v in class_indices.items())
            test_df['category'] = test_df['category'].replace(class_indices)
        print(test_df['category'].value_counts())
        print('')


# here starts the main funciton

# creating test set including hcpa patient's
test_negative_folders = []
test_typical_folders = []
test_atypical_folders = []
test_indetermined_1_folders = []
test_indetermined_2_folders = []
test_img_src_folder = 'ImagensProcessadas'

data_description = 'annotations.xlsx'  # file with the data annotations
annotations = pd.read_excel(data_description)

c = 0
for i in range(len(annotations)):
    classification = annotations["Classificação"][i]
    pcr = annotations["PCR_FINAL"][i]

    if str(annotations["nome"][i]) != "nan":
        c += 1
        p_num = int(annotations["nome"][i][1:])
        p_id = 'C' + str(p_num)
        if classification == "2 - típico" and pcr == 2:  # typical patients, negative pcr
            test_typical_folders.append(p_id)

        elif classification == "1 - negativo" and pcr == 1 and p_num != 1:  # negative patients, positive pcr
            test_negative_folders.append(p_id)

        elif classification == "3 - indeterminado" and pcr == 1:
            test_indetermined_1_folders.append(p_id)

        elif classification == "3 - indeterminado" and pcr == 2:
            test_indetermined_2_folders.append(p_id)

        elif classification == "4 - atípico":
            test_atypical_folders.append(p_id)
        else:
            c -= 1
print('included patients from annotated excel file: ', c)
print('')

print("Found", len(test_typical_folders),
      "HMV Typical (negative pcr) patients")
print("Found", len(test_negative_folders),
      "HMV Negative (positive pcr) patients")
print("Found", len(test_indetermined_1_folders),
      "HMV Indetermined (positive pcr) patients")
print("Found", len(test_indetermined_2_folders),
      "HMV Indetermined (negative pcr) patients")
print("Found", len(test_atypical_folders), "HMV Atypical patients")

# making a correction imported from annotations file
for index in range(len(test_indetermined_1_folders) - 1):
    if test_indetermined_1_folders[index] == 'C8':
        test_indetermined_1_folders[index] = 'C48'


# getting the predictions of test set by patient

print('')
print('Predicting Test Patients')
print('')
print('')
print('Predicting Negative (positive pcr) Patients')
print('')
predictions_by_patient(model, test_negative_folders)
print('')
print('Predicting Typical (negative pcr) Patients')
print('')
predictions_by_patient(model, test_typical_folders)
print('')
print('')
print('Predicting Indetermined (positive pcr) Patients')
print('')
predictions_by_patient(model, test_indetermined_1_folders)
print('')
print('Predicting Indetermined (negative pcr) Patients')
print('')
predictions_by_patient(model, test_indetermined_2_folders)
print('Predicting Atypical Patients')
print('')
predictions_by_patient(model, test_atypical_folders)
print('')
