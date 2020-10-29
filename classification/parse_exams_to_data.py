import os
import glob
import pydicom
import shutil
import re
import json


EXAM_FOLDERS = [
    '/home/chicobentojr/Desktop/cidia-files/exams/C11',
    '/home/chicobentojr/Desktop/cidia-files/exams'
]
VIDEO_URL_PREFIX = '/assets/exam-data/preview-video/'
VIDEO_FOLDERS = [
    '/home/chicobentojr/Desktop/cidia-files/MP4-HMV-HCPA',
]
SLICES_FOLDERS = [
    '/home/chicobentojr/Desktop/cidia-files/important-slices',
]
SLICE_PREFIX = '/assets/exam-data/important-slices/'


output_folder = 'exam-data'
os.makedirs(output_folder, exist_ok=True)

result_dict = {}


print('Retrieving exams StudyInstanceUID')
print()

exam_folders = []
for x in EXAM_FOLDERS:
    exam_folders.extend(glob.glob('{}/*'.format(x)))

for fold in exam_folders:
    print('Processing', fold)

    for root, dirnames, filenames in os.walk(fold):
        if len(filenames) > 0:
            slice_file = os.path.join(root, filenames[0])

            dataset = pydicom.dcmread(slice_file)

            study_uid = dataset.get('StudyInstanceUID', 'not-found')
            patient_id = dataset.get('PatientID', 'not-found')

            result_dict[study_uid] = {
                'patientID': patient_id,
            }

print()
print('Retrieving video URL')
print()

for key, patient in result_dict.items():

    patient_id = patient['patientID']

    for video_folder in VIDEO_FOLDERS:
        for root, dirnames, filenames in os.walk(video_folder):
            video_url = next((f for f in filenames if patient_id in f), None)

            if video_url:
                print(f"Video found: {video_url}")

                video_new_folder = f"{output_folder}/preview-videos"
                os.makedirs(video_new_folder, exist_ok=True)

                shutil.copy(f"{root}/{video_url}",
                            f"{video_new_folder}/{video_url}")

                patient['videoURL'] = f"{VIDEO_URL_PREFIX}{video_url}"
print()
print('Retrieving important slices')
print()

for key, patient in result_dict.items():

    patient_id = patient['patientID']

    for video_folder in SLICES_FOLDERS:
        for root, dirnames, filenames in os.walk(video_folder):
            important_slices = [f for f in filenames if patient_id in root]

            if important_slices:
                print(f"Slices found to {patient_id}: {important_slices}")

                slice_new_folder = f"{output_folder}/important-slices"
                os.makedirs(slice_new_folder, exist_ok=True)

                for slc in important_slices:
                    pat_slc_folder = f"{slice_new_folder}/{patient_id}"
                    os.makedirs(pat_slc_folder, exist_ok=True)
                    shutil.copy(f"{root}/{slc}", f"{pat_slc_folder}/{slc}")

                important_slices = [
                    f"{SLICE_PREFIX}{x}" for x in important_slices]
                patient['importantSlices'] = important_slices
print()

result_json = json.dumps(result_dict, indent=2)
with open("cidia-studies.json", "w") as f:
    f.write(result_json)
