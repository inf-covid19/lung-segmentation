import os
import glob
import pydicom
import shutil
import re
import dicom2nifti


main_folder = 'exames-organizados-parte-2'
output_folder = 'nii-parte-2'


folders = sorted(glob.glob('{}/*'.format(main_folder)))

os.makedirs(output_folder, exist_ok=True)


for fold in folders:
    print('Processing', fold)

    exam_folders = sorted(glob.glob('{}/*/*/*'.format(fold)))

    for exam_folder in exam_folders:
        print('Exam folder ...', exam_folder)

        out_exam_folder = '{}/{}.nii.gz'.format(
            output_folder, exam_folder)

        os.makedirs('/'.join(out_exam_folder.split('/')[0:-1]), exist_ok=True)

        try:
            dicom2nifti.dicom_series_to_nifti(
                exam_folder, out_exam_folder, reorient_nifti=False)
        except Exception as e:
            print('Error while converting to', out_exam_folder)
            print(e)

    print()
