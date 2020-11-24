import os
import glob
import pydicom
import shutil
import re
import json
import pandas as pd


OUTPUT_FILE = 'metadata.json'
EXAM_FOLDERS = [
    # '/ssd/share/CT-Original/DICOM-HCPA/exame-pulmao',
    # '/ssd/share/CT-Original/DICOM-HMV/exame-pulmao',
    # '/home/users/chicobentojr/workspace/HCPA-organizado-parte-2'
    '/home/chicobentojr/Desktop/cidia-files/exams'
]


IGNORE_TAGS = ['PixelData']

result_dict = {}

print('Retrieving exams')
print()

exam_folders = []
for x in EXAM_FOLDERS:
    exam_folders.extend(glob.glob('{}/*'.format(x)))

for folder in exam_folders:
    print('Processing', folder)

    for root, dirnames, filenames in os.walk(folder):
        if len(filenames) > 0:
            slice_file = os.path.join(root, filenames[0])

            dataset = pydicom.dcmread(slice_file)

            dataset_dict = {}
            dataset_dict["__SeriesLength"] = len(filenames)

            for name in dataset.trait_names():
                if name not in IGNORE_TAGS and name[0].isupper():
                    dataset_dict[name] = str(dataset.get(name))

            patient_name = str(dataset.get('PatientName', 'not-found'))

            result_dict[patient_name] = dataset_dict


result_json = json.dumps(result_dict, indent=2, ensure_ascii=False)
with open(OUTPUT_FILE, "w") as f:
    f.write(result_json)


df = pd.DataFrame(result_dict).T

df.index.rename('Exam', inplace=True)

df.to_csv(OUTPUT_FILE.replace('.json', '.csv'))
