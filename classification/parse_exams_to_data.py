import os
import glob
import pydicom
import shutil
import re
import json


main_folder = '/home/chicobentojr/Desktop/cidia-files/exams'


folders = glob.glob('{}/*'.format(main_folder))

# os.makedirs(output_folder, exist_ok=True)

result_dict = {}


for fold in folders:
    print('Processing', fold)

    for root, dirnames, filenames in os.walk(fold):
        if len(filenames) > 0:
            slice_file = os.path.join(root, filenames[0])

            dataset = pydicom.dcmread(slice_file)

            study_uid = dataset.get('StudyInstanceUID', 'not-found')
            patient_id = dataset.get('PatientID', 'not-found')

            result_dict[study_uid] = {
                'name': patient_id,
                'tag': patient_id
            }

# print('result JSON')
# print(json.dumps(result_dict, indent=2))

result_json = json.dumps(result_dict, indent=2)
with open("cidia-studies.json", "w") as f:
    f.write(result_json)
