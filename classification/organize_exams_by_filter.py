import os
import glob
import pydicom
import shutil
import re


main_folder = 'exames-anonimizado-parte-1'
output_folder = 'parte-1'


folders = glob.glob('{}/*'.format(main_folder))

os.makedirs(output_folder, exist_ok=True)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


for fold in folders:
    print('Processing', fold)

    exam_folders = glob.glob('{}/*/*'.format(fold))

    top_exam_folder = []

    for exam_folder in exam_folders:
        files = glob.glob('{}/*'.format(exam_folder))
        top_exam_folder.append((len(files), exam_folder))

    top_exam_folder = sorted(top_exam_folder, reverse=True)

    best_exam_folders = top_exam_folder[0:2]

    # print(top_exam_folder)
    print()
    # print(best_exam_folders)

    for exam_len, exam in best_exam_folders:
        slice_file = glob.glob('{}/*'.format(exam))[0]

        # print('slice', slice_file)

        dataset = pydicom.dcmread(slice_file)

        conv_filter = dataset.get('ConvolutionKernel', 'not-found')
        print('ConvolutionKernel', conv_filter)

        conv_number = int(re.findall('\d+', str(conv_filter))[0])

        # print('conv filter', conv_filter, 'number', conv_number)

        exam_type = 'pulmao' if conv_number > 50 else 'mediastino'

        # print(exam_type)

        # print('conv', conv_filter)

        exam_number_folder = '/'.join(exam.split('/')[1:])

        # print(exam_folder)
        # print(exam_number_folder)

        out_exam_folder = '{}/exame-{}/{}/'.format(
            output_folder, exam_type, exam_number_folder)

        print('Copying to ...', out_exam_folder)

        # print(out_exam_folder)
        os.makedirs(out_exam_folder, exist_ok=True)
        copytree(exam, out_exam_folder)

    print()
