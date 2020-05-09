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

FOLDER = 'lung-data/LE4LKLM2/*'
RESULT_FOLDER = 'lung-data/LE4LKLM2-segmented/'

try:
    os.makedirs(RESULT_FOLDER)
except Exception as e:
    pass


def get_result_filename(filename):
    image_name = filename.split('/')[-1]
    return RESULT_FOLDER + image_name


files = glob.glob(FOLDER)

time_start = time.ctime()

for index, filename in enumerate(files):
    print('Processing {}/{} {}'.format(index + 1, len(files), filename))
    # Saving pydicom png file
    dataset = pydicom.dcmread(filename)

    ds_shape = dataset.pixel_array.shape
    ds_2d = dataset.pixel_array.astype(float)
    ds_2d_scaled = np.uint8((np.maximum(ds_2d, 0) / ds_2d.max()) * 255.0)

    # with open('{}-pydicom.png'.format(get_result_filename(filename)), 'wb') as png_file:
    #     w = png.Writer(ds_shape[1], ds_shape[0], greyscale=True)
    #     w.write(png_file, ds_2d_scaled)

    # Saving segmentation mask
    input_image = sitk.ReadImage(filename)
    segmentation = mask.apply(input_image)[0]

    shape = segmentation.shape
    mask_scaled = np.uint8(np.maximum(segmentation, 0) /
                           segmentation.max() * 255.0)
    mask_scaled = np.uint8(np.where(mask_scaled > 0, 255, 0))

    # with open('{}-mask.png'.format(get_result_filename(filename)), 'wb') as png_file:
    #     w = png.Writer(shape[1], shape[0], greyscale=True)
    #     w.write(png_file, mask_scaled)

    image_superimposed = ds_2d_scaled
    image_superimposed[mask_scaled == 0] = 0

    # with open('{}-superimposed.png'.format(get_result_filename(filename)), 'wb') as png_file:
    #     w = png.Writer(shape[1], shape[0], greyscale=True)
    #     w.write(png_file, image_superimposed)
    def make_mb_image(i_img, i_gt, ds_op=lambda x: x[::1, ::1]):
        n_img = (i_img-i_img.mean())/(2*i_img.std())+0.5
        c_img = plt.cm.bone(n_img)[:, :, :3]
        c_img = mark_boundaries(c_img, label_img=ds_op(
            i_gt), color=(0, 1, 0), mode='thick')
        return c_img

    plt.imshow(make_mb_image(ds_2d, mask_scaled))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('{}-segmented.png'.format(get_result_filename(filename)),
                transparent=True, bbox_inches='tight', pad_inches=0)
    plt.cla()


print('Process started at:', time_start)
print('Process endend at:', time.ctime())
