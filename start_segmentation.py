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

S_FOLDER = 'segmented'
O_FOLDER = 'overlapped'
M_FOLDER = 'masks'
I_FOLDER = 'images'


def get_result_filename(dest_folder, filename):
    image_name = filename.split('/')[-1]
    return '{}/{}'.format(dest_folder.rstrip('/'), image_name)


def create_output_folders(args):
    output_folder = args.output.rstrip('/')

    for child_folder, can_create in [('', True),
                                     (S_FOLDER, True),
                                     (O_FOLDER, args.overlap),
                                     (I_FOLDER, args.images),
                                     (M_FOLDER, args.mask)]:
        if can_create:
            try:
                os.makedirs('{}/{}'.format(output_folder, child_folder))
            except Exception as e:
                pass
    return output_folder


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('input', metavar='input',
                        help='filepath for input folder')
    parser.add_argument('output', metavar='output',
                        help='filepath for output folder')
    parser.add_argument('--save-overlap', dest='overlap',
                        action='store_true', help='save folder with overlapped images')
    parser.add_argument('--save-mask', dest='mask',
                        action='store_true', help="save folder with mask images")
    parser.add_argument('--save-images', dest='images',
                        action='store_true', help="save folder with converted png images")
    parser.add_argument('--color', metavar='color',
                        help="result segmented color map (gray, bone, cool, copper, flag, hot, jet, pink, prism, spring, summer, winter)")
    parser.add_argument('--model', metavar='model',
                        help="model used to segment image (R231, LTRCLobes, LTRCLobes_R231, R231CovidWeb)")
    parser.set_defaults(overlap=False, color='bone')

    args = parser.parse_args()

    input_folder = args.input.rstrip('/')
    output_folder = create_output_folders(args)

    files = sorted(glob.glob('{}/*'.format(input_folder.rstrip('/'))))

    time_start = time.ctime()

    for index, file_path in enumerate(files):
        print('Processing {}/{} {}'.format(index + 1, len(files), file_path))

        try:
            dataset = pydicom.dcmread(file_path)

            acquisition_time = dataset.get('AcquisitionTime', index + 1)
            new_filename = '{}_{}'.format(
                acquisition_time,  file_path.split('/')[-1])

            ds_shape = dataset.pixel_array.shape
            ds_2d = dataset.pixel_array.astype(float)
            ds_2d_scaled = np.uint8(
                (np.maximum(ds_2d, 0) / ds_2d.max()) * 255.0)

            if args.images:
                with open('{}/{}/{}.png'.format(output_folder, I_FOLDER, new_filename), 'wb') as png_file:
                    w = png.Writer(ds_shape[1], ds_shape[0], greyscale=True)
                    w.write(png_file, ds_2d_scaled)

            input_image = sitk.ReadImage(file_path)

            if args.model:
                if args.model == 'LTRCLobes_R231':
                    segmentation = mask.apply_fused(input_image)[0]
                else:
                    model = mask.get_model('unet', args.model)
                    segmentation = mask.apply(input_image, model)[0]
            else:
                segmentation = mask.apply(input_image)[0]

            shape = segmentation.shape

            mask_scaled = np.uint8(np.maximum(segmentation, 0) /
                                   segmentation.max() * 255.0)
            # mask_scaled = np.uint8(np.where(mask_scaled > 0, 255, 0))

            if args.mask:
                with open('{}/{}/{}.png'.format(output_folder, M_FOLDER, new_filename), 'wb') as png_file:
                    w = png.Writer(shape[1], shape[0], greyscale=True)
                    w.write(png_file, mask_scaled)

            if args.overlap:
                image_superimposed = ds_2d_scaled
                image_superimposed[mask_scaled == 0] = 0

                with open('{}/{}/{}.png'.format(output_folder, O_FOLDER, new_filename), 'wb') as png_file:
                    w = png.Writer(shape[1], shape[0], greyscale=True)
                    w.write(png_file, image_superimposed)

            def make_mb_image(i_img, i_gt, ds_op=lambda x: x[::1, ::1]):
                n_img = (i_img-i_img.mean())/(2*i_img.std())+0.5

                if args.color == 'gray':
                    c_img = plt.cm.gray(n_img)[:, :, :3]
                elif args.color == 'cool':
                    c_img = plt.cm.cool(n_img)[:, :, :3]
                elif args.color == 'copper':
                    c_img = plt.cm.copper(n_img)[:, :, :3]
                elif args.color == 'flag':
                    c_img = plt.cm.flag(n_img)[:, :, :3]
                elif args.color == 'hot':
                    c_img = plt.cm.hot(n_img)[:, :, :3]
                elif args.color == 'jet':
                    c_img = plt.cm.jet(n_img)[:, :, :3]
                elif args.color == 'pink':
                    c_img = plt.cm.pink(n_img)[:, :, :3]
                elif args.color == 'prism':
                    c_img = plt.cm.prism(n_img)[:, :, :3]
                elif args.color == 'spring':
                    c_img = plt.cm.spring(n_img)[:, :, :3]
                elif args.color == 'summer':
                    c_img = plt.cm.summer(n_img)[:, :, :3]
                elif args.color == 'winter':
                    c_img = plt.cm.winter(n_img)[:, :, :3]
                else:
                    c_img = plt.cm.bone(n_img)[:, :, :3]

                c_img = mark_boundaries(c_img, label_img=ds_op(
                    i_gt), color=(0, 1, 0), mode='thick')
                return c_img

            plt.imshow(make_mb_image(ds_2d, mask_scaled))
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig('{}/{}/{}.png'.format(output_folder, S_FOLDER, new_filename),
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.cla()
        except Exception as e:
            print('Error while processing "{}":'.format(file_path))
            print(e)

    print('Process started at:', time_start)
    print('Process finished at:', time.ctime())


if __name__ == "__main__":
    main()
