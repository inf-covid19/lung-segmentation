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
import torch
import scipy.misc
import l3net
from l3net.src import models, datasets
from l3net import exp_configs

S_FOLDER = 'segmented'
O_FOLDER = 'overlapped'
M_FOLDER = 'masks'
I_FOLDER = 'images'

COLORS = [
    (0, 1, 0),
    (1, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (1, 0, 1),
]


def get_result_filename(dest_folder, filename):
    image_name = filename.split('/')[-1]
    return '{}/{}'.format(dest_folder.rstrip('/'), image_name)


def create_output_folders(args):
    output_folder = args.output.rstrip('/')

    if args.model == "L3Net":
        for i in range(5):
            for child_folder, can_create in [('', True),
                                             (S_FOLDER, True),
                                             (O_FOLDER, args.overlap),
                                             (I_FOLDER, args.images),
                                             (M_FOLDER, args.mask)]:
                if can_create:
                    os.makedirs('{}/{}/{}'.format(output_folder,
                                                  i+1, child_folder), exist_ok=True)
            os.makedirs('{}/all/segmented/'.format(output_folder),
                        exist_ok=True)
    for child_folder, can_create in [('', True),
                                     (S_FOLDER, True),
                                     (O_FOLDER, args.overlap),
                                     (I_FOLDER, args.images),
                                     (M_FOLDER, args.mask)]:
        if can_create:
            os.makedirs('{}/{}'.format(output_folder,
                                       child_folder), exist_ok=True)
    return output_folder


def get_model_segmentation(args, file_path):
    if args.model == 'L3Net':
        model_dict = torch.load(
            'l3net/unet2d-output/d676301aefbdd7ac0de10b39089173da/model_best.pth')
        exp_dict = exp_configs.EXP_GROUPS['open_source_unet2d'][0]

        test_set = l3net.src.datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                                  split="val",
                                                  datadir='l3net/data/L3netDemoData/',
                                                  exp_dict=exp_dict,
                                                  dataset_size=exp_dict['dataset_size'])

        model = l3net.src.models.get_model(model_dict=exp_dict['model'],
                                           exp_dict=exp_dict,
                                           train_set=test_set).cuda()

        img_dcm = pydicom.dcmread(file_path)

        image = img_dcm.pixel_array.astype(float)

        shape = img_dcm.pixel_array.shape

        ds_2d_scaled = np.uint8(
            (np.maximum(image, 0) / image.max()) * 255.0)

        x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)
        x_tensor = x_tensor.unsqueeze(0)
        x_tensor = x_tensor.type(torch.cuda.FloatTensor)
        pr_mask = model.predict(x_tensor)
        # pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pr_mask = (pr_mask.squeeze().cpu().detach().numpy().round())

        return pr_mask

        # print('pr_mask np array shape: ', pr_mask.shape)
        # masks = []
        # masks_correct = []
        # for i in pr_mask:
        #     masks.append(i)
        # shape = masks[0].shape

        # COLORS = [
        #     (0, 1, 0),
        #     (1, 0, 0),
        #     (0, 0, 1),
        #     (0, 1, 1),
        #     (1, 0, 1),
        # ]

        # for count, i in enumerate(masks):
        #     imgname = 'outfile-mask-' + str(count) + '.png'
        #     print()
        #     print('mask', count, '-', imgname)
        #     # print(i)
        #     print('shape', i.shape)

        #     print('max', i.max(), 'min', i.min())
        #     # mask_scaled = np.uint8(np.where(i > 0, 255, 0))

        #     mask_scaled = np.uint8(np.maximum(i, 0) / i.max() * 255.0)
        #     mask_scaled = np.uint8(np.where(mask_scaled > 0, 255, 0))

        #     print('mask scaled')
        #     # print(mask_scaled)
        #     print('max', mask_scaled.max(), 'min', mask_scaled.min())
        #     with open(imgname, 'wb') as png_file:
        #         w = png.Writer(shape[1], shape[0], greyscale=True)
        #         w.write(png_file, mask_scaled)

        #     masks_correct.append(mask_scaled)

        #     def make_mb_image(i_img, i_gt, color=(0, 1, 0), ds_op=lambda x: x[::1, ::1]):
        #         n_img = (i_img-i_img.mean())/(2*i_img.std())+0.5

        #         # c_img = plt.cm.gray(n_img)[:, :, :3]
        #         c_img = plt.cm.bone(n_img)[:, :, :3]

        #         c_img = mark_boundaries(c_img, label_img=ds_op(
        #             i_gt), color=color, mode='thick')
        #         return c_img

        #     plt.imshow(make_mb_image(
        #         ds_2d_scaled, mask_scaled, color=COLORS[count]))
        #     # mask_color = mask_scaled
        #     # mask_color[mask_scaled == 255] = [0,0,255]
        #     # plt.imshow(mask_scaled, alpha=0.4)
        #     plt.gca().set_axis_off()
        #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
        #                         hspace=0, wspace=0)
        #     plt.margins(0, 0)
        #     plt.savefig('out-mask-{}.png'.format(count), transparent=True,
        #                 bbox_inches='tight', pad_inches=0)
        #     plt.cla()

        # def make_fill_image(i_img, masks, color=(0, 1, 0), ds_op=lambda x: x[::1, ::1]):
        #     n_img = (i_img-i_img.mean())/(2*i_img.std())+0.5

        #     c_img = plt.cm.bone(n_img)[:, :, :3]

        #     for index, mask in enumerate(masks):
        #         c_img = mark_boundaries(c_img, label_img=ds_op(
        #             mask), color=COLORS[index], mode='thick')
        #     return c_img

        # plt.imshow(make_fill_image(ds_2d_scaled, masks_correct))
        # plt.gca().set_axis_off()
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
        #                     hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig('out-full-mask-{}.png'.format(count),
        #             transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.cla()


def make_mb_image(args, i_img, i_gt, masks=[], color=(0, 1, 0), ds_op=lambda x: x[::1, ::1]):
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

    if masks:
        for i, mask in enumerate(masks):
            c_img = mark_boundaries(c_img, label_img=ds_op(
                mask), color=COLORS[i], mode='thick')
    else:
        c_img = mark_boundaries(c_img, label_img=ds_op(
            i_gt), color=color, mode='thick')
    return c_img


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
                    segmentations = mask.apply_fused(input_image)
                if args.model == 'L3Net':
                    segmentations = get_model_segmentation(args, file_path)
                else:
                    model = mask.get_model('unet', args.model)
                    segmentations = mask.apply(input_image, model)
            else:
                segmentations = mask.apply(input_image)

            for s_i, segmentation in enumerate(segmentations):
                try:
                    shape = segmentation.shape

                    mask_scaled = np.uint8(np.maximum(segmentation, 0) /
                                           segmentation.max() * 255.0)
                    mask_scaled = np.uint8(np.where(mask_scaled > 0, 255, 0))

                    if args.mask:
                        with open('{}/{}/{}/{}.png'.format(output_folder, s_i + 1, M_FOLDER, new_filename), 'wb') as png_file:
                            w = png.Writer(shape[1], shape[0], greyscale=True)
                            w.write(png_file, mask_scaled)

                    if args.overlap:
                        image_superimposed = ds_2d_scaled
                        image_superimposed[mask_scaled == 0] = 0

                        with open('{}/{}/{}/{}.png'.format(output_folder, s_i + 1, O_FOLDER, new_filename), 'wb') as png_file:
                            w = png.Writer(shape[1], shape[0], greyscale=True)
                            w.write(png_file, image_superimposed)

                    plt.imshow(make_mb_image(
                        args, ds_2d, mask_scaled, color=COLORS[s_i]))
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                        hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.savefig('{}/{}/{}/{}.png'.format(output_folder, s_i+1,  S_FOLDER, new_filename),
                                transparent=True, bbox_inches='tight', pad_inches=0)
                    plt.cla()
                except Exception as e:
                    print('Error in segmentation "{}":'.format(s_i+1))
                    print(e)
            if len(segmentations) > 1:
                masks = []
                for s_i, segmentation in enumerate(segmentations):
                    shape = segmentation.shape

                    mask_scaled = np.uint8(np.maximum(segmentation, 0) /
                                           segmentation.max() * 255.0)
                    mask_scaled = np.uint8(np.where(mask_scaled > 0, 255, 0))
                    masks.append(mask_scaled)
                plt.imshow(make_mb_image(
                    args, ds_2d, mask_scaled, masks=masks, color=COLORS[s_i]))
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig('{}/{}/{}/{}.png'.format(output_folder, 'all',  S_FOLDER, new_filename),
                            transparent=True, bbox_inches='tight', pad_inches=0)
                plt.cla()

        except Exception as e:
            print('Error while processing "{}":'.format(file_path))
            print(e)

    print('Process started at:', time_start)
    print('Process finished at:', time.ctime())


if __name__ == "__main__":
    main()
