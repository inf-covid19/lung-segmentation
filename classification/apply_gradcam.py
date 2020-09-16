import time
import os
import argparse
import json
import imutils
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model as kModel
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                print('found layer', layer.name)
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = kModel(
            inputs=[self.model.inputs],
            # inputs=[self.model.input],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)


def main(args):
    # import the necessary packages
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    #                 help="path to the input image")
    # ap.add_argument("-m", "--model", type=str, default="vgg",
    #                 # choices=("vgg", "resnet"),
    #                 help="model to be used")
    # args = vars(ap.parse_args())

    files = args.input_files
    output_folder = args.output_folder.rstrip('/')
    img_size = args.size
    target_size = (img_size, img_size)

    model_filename = args.model

    model = load_model(model_filename)

    os.makedirs(output_folder, exist_ok=True)

    if args.linear_activation:
        model.layers[-1].activation = tf.keras.activations.linear

    for i, filepath in enumerate(files):
        print(f"Processing {i+1}/{len(files)} => {filepath}")
        filename = filepath.split('/')[-1]
        # args['image'] = slice_filename

        # # initialize the model to be VGG16
        # Model = VGG16
        # # check to see if we are using ResNet
        # if args["model"] == "resnet":
        #     Model = ResNet50
        # load the pre-trained CNN from disk
        # print("[INFO] loading model...")
        # model = Model(weights="imagenet")

        # model_filename = args['model']

        # print('model summary')
        # model.summary()
        # print()

        # sss = 300

        # load the original image from disk (in OpenCV format) and then
        # resize the image to its target dimensions
        # orig = cv2.imread(args["image"])
        orig = cv2.imread(filepath)
        resized = cv2.resize(orig, target_size)
        # load the input image from disk (in Keras/TensorFlow format) and
        # preprocess it
        image = load_img(filepath, target_size=target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # use the network to make predictions on the input image and find
        # the class label index with the largest corresponding probability
        preds = model.predict(image)

        # print('Preds', preds)

        # print("preds", preds)
        i = np.argmax(preds[0])
        # decode the ImageNet predictions to obtain the human-readable label
        # decoded = imagenet_utils.decode_predictions(preds)
        # (imagenetID, label, prob) = decoded[0][0]
        # label = "{}: {:.2f}%".format(label, prob * 100)
        # print("[INFO] {}".format(label))
        label = filename

        # initialize our gradient class activation map and build the heatmap
        cam = GradCAM(model, i)
        heatmap = cam.compute_heatmap(image)

        # print('heatmap')
        # print(heatmap)

        # resize the resulting heatmap to the original input image dimensions
        # and then overlay heatmap on top of the imageprint('heatmap')
        # print(heatmap)rig, alpha=0.5)
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

        # draw the predicted label on the output image
        cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        # display the original image and resulting heatmap and output image
        # to our screen
        output = np.vstack([orig, heatmap, output])
        output = imutils.resize(output, height=700)
        # cv2.imshow("Output", output)

        out_file = f"{output_folder}/{filename}"

        cv2.imwrite(out_file, output)
        # cv2.waitKey(0)


# model_filename = 'cidia-lung-model_1.h5'
# model_filename = 'local-modelhaha.h5'
# slice_filename = '/home/chicobentojr/Desktop/cidia/data/C11/-slice260..png'

# model_filename = '/home/chicobentojr/Desktop/cidia/model/test_chico/fold1/my_checkpoint/'
# slice_filename = '/home/chicobentojr/Desktop/cidia/model/test_chico/TYP-009/axis1/3D_View1.png'

# slice_filename = '/home/chicobentojr/Desktop/cidia/data/C11/-slice260..png'
# model_filename = '/home/chicobentojr/Desktop/cidia/model/model_1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_files', metavar='INPUT_FILES', nargs='+')
    parser.add_argument('output_folder', metavar='OUTPUT_FOLDER')
    parser.add_argument('-m', '--model', metavar='model')
    parser.add_argument(
        '--linear', dest='linear_activation', action='store_true')
    parser.add_argument('-s', '--size', dest='size', type=int)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--with-raw', dest='with_raw', action='store_true')
    parser.add_argument('--preds', dest='preds', action='store_true')
    parser.add_argument('--verbose', dest='verbose', action='store_true')

    parser.set_defaults(debug=False, with_raw=False, linear_activation=False,
                        plot='confusion-matrix', verbose=False, size=300,
                        fixed_prefix='results/MobileNet/')

    args = parser.parse_args()

    print('Input')
    print(json.dumps(args.__dict__, indent=2))
    print()

    start = time.ctime()
    main(args)
    print('Process started at ', start)
    print('Process finished at ', time.ctime())
