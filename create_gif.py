import imageio
import time
import PIL
import SimpleITK as sitk
import numpy as np
from lungmask import mask
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import numpy as np
import png
from skimage.segmentation import clear_border, mark_boundaries
import glob
import os
import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('input', metavar='input',
                        help='filepath for input folder')
    parser.add_argument('output', metavar='output',
                        help='filepath for output folder')

    args = parser.parse_args()
    input_folder = args.input.rstrip('/')
    output_folder = output_folder = args.output.rstrip('/')

    try:
        os.makedirs(output_folder)
    except:
        pass

    files = sorted(glob.glob('{}/*'.format(input_folder.rstrip('/'))))

    if not files:
        print('{} does not have any file'.format(input_folder))
        return

    time_start = time.ctime()

    files.sort()

    images = list(map(lambda filename: imageio.imread(filename), files))

    output_file = '-'.join(files[0].split('/')[:-2]) + '.gif'

    imageio.mimsave(output_folder + '/' + output_file, images, duration=0.04)

    print('Process started at:', time_start)
    print('Process finished at:', time.ctime())


if __name__ == "__main__":
    main()
