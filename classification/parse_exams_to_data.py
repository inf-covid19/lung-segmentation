import os
import glob
import pydicom
import shutil
import re
import json
import pandas as pd


EXAM_FOLDERS = [
    '/ssd/share/CT-Original/DICOM-HCPA/exame-pulmao',
    '/ssd/share/CT-Original/DICOM-HMV/exame-pulmao',
    '/home/users/chicobentojr/workspace/HCPA-organizado-parte-2'
    # '/home/chicobentojr/Desktop/cidia-files/exams'

]
VIDEO_URL_PREFIX = '/assets/exam-data/preview-video/'
VIDEO_FOLDERS = [
    '/ssd/share/CT-Segmentado/03-P-HNN/MP4-HMV-HCPA',
    # '/home/chicobentojr/Desktop/cidia-files/MP4-HMV-HCPA'
]
SLICES_FOLDERS = [
    '/ssd/share/CT-Segmentado/03-P-HNN/NII-HCPA/exame-pulmao/',
    '/ssd/share/CT-Segmentado/03-P-HNN/NII-HMV/exame-pulmao/',
    '/ssd/share/CT-Segmentado/03-P-HNN/NII-HCPA-parte-2/exame-pulmao/',
    # '/home/chicobentojr/Desktop/cidia-files/important-slices'
]
SLICE_PREFIX = '/assets/exam-data/important-slices/'

STATISTICS_FILES = [
    '/ssd/share/CT-Segmentado/03-P-HNN/statistics/hmv_statistics_all.csv',
    '/ssd/share/CT-Segmentado/03-P-HNN/statistics/hcpa_statistics_all.csv',
    # 'statistics/hcpa_statistics_all.csv',
    # 'statistics/hmv_statistics_all.csv',
]

RESULT_CSV_FILE = 'fapergs3.csv'
RESULT_COLUMNS = {
    "Radiology ": "Radiology",
    "AVG ALL": "Avegare All",
    "1: Spherical": "Spherical",
    "2: Slices 2D": "Slices 2D",
    "3: VR": "Volume Rendering",
    "AVG 2 e 3": "Average 2-3",
    "AVG 1 e 2": "Average 1-2",
    "AVG 1 e 3": "Average 1-3",
    "% total ggo": "% Total GGO",
    "% right lung ggo": "% Right Lung GGO",
    "% left lung ggo": "% Left Lung GGO",
    "# lesions": "# Lesions",
    "# lesions right lung": "# Lesions Right Lung",
    "# lesions left lung": "# Lesions Left Lung",

    # "Fold number": 2,
    # "Patient ID": "C114",
    # "Img. esf\u00e9ricas": "2,79%",
    # "Slices 2D": "40,00%",
    # "Vistas 3D (Eixos 1-2-4)": "10,16%",
    # "Fold": 2,
    # "Patient ID.1": "C114",
    # "Radiology .1": "typical",
    # "AVG ALL": NaN,
    # "1: Spherical": NaN,
    # "AVG 1 e 2": NaN,
    # "AVG 1 e 3": NaN,
    # "Erros 2": 1.0,
    # "Erros 3": 1.0,
    # "Erros 2 e 3": 1.0
}

EXAM_NAME_MAP = {
    "EXAME1": "C1",
    "EXAME2": "C2",
    "EXAME3": "C3",
    "EXAME4": "C4",
    "EXAME5": "C5",
    "EXAME6": "C6",
    "EXAME7": "C7",
    "EXAME8": "C8",
    "EXAME9": "C9",
    "EXAME10": "C10",
    "EXAME11": "C11",
    "EXAME12": "C12",
    "EXAME13": "C13",
    "EXAME14": "C14",
    "EXAME15": "C15",
    "EXAME16": "C16",
    "EXAME17": "C17",
    "EXAME18": "C18",
    "EXAME19": "C19",
    "EXAME20": "C20",
}


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

            if patient_id in EXAM_NAME_MAP:
                patient_id = EXAM_NAME_MAP[patient_id]

            result_dict[study_uid] = {
                'patientID': patient_id,
                'labels': {
                    'Name': patient_id,
                }
            }

print()
print('Retrieving models results')
print()

result_df = pd.read_csv(RESULT_CSV_FILE)

for key, patient in result_dict.items():

    patient_id = patient['patientID']

    row_df = result_df[result_df['Patient ID'] == patient_id]

    if row_df.empty:
        continue

    columns = row_df.columns.tolist()

    labels = patient['labels']

    for i, r in row_df.iterrows():
        for col, name in RESULT_COLUMNS.items():
            if col in r and r[col]:
                labels[name] = r[col]

    patient['labels'] = labels

print()
print('Retrieving models statististics')
print()

for stat_file in STATISTICS_FILES:
    result_df = pd.read_csv(stat_file, sep=';')

    for key, patient in result_dict.items():

        patient_id = patient['patientID']

        row_df = result_df[result_df['pacient'] == patient_id]

        if row_df.empty:
            continue

        columns = row_df.columns.tolist()

        labels = patient['labels']

        for i, r in row_df.iterrows():
            for col, name in RESULT_COLUMNS.items():
                if col in r and r[col]:
                    v = r[col]

                    if type(v) is float:
                        labels[name] = f"{v:.4f}"
                    else:
                        labels[name] = v

        patient['labels'] = labels

print()
print('Retrieving video URL')
print()

for key, patient in result_dict.items():

    patient_id = patient['patientID']

    for video_folder in VIDEO_FOLDERS:
        for root, dirnames, filenames in os.walk(video_folder):
            video_url = next(
                (f for f in filenames if f"{patient_id}.mp4" in f), None)

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

    for slice_folder in SLICES_FOLDERS:
        for root, dirnames, filenames in os.walk(slice_folder):
            important_slices = [
                f for f in filenames
                if (patient_id == root.split('/')[-1] or patient_id == root.split('/')[-2])
                and ".png" in f]

            if important_slices:
                print(f"Slices found to {patient_id}: {important_slices}")

                slice_new_folder = f"{output_folder}/important-slices"
                os.makedirs(slice_new_folder, exist_ok=True)

                for slc in important_slices:
                    pat_slc_folder = f"{slice_new_folder}/{patient_id}"
                    os.makedirs(pat_slc_folder, exist_ok=True)
                    shutil.copy(f"{root}/{slc}", f"{pat_slc_folder}/{slc}")

                important_slices = [
                    f"{SLICE_PREFIX}{patient_id}/{x}" for x in important_slices]
                patient['importantSlices'] = important_slices
print()

result_json = json.dumps(result_dict, indent=2, ensure_ascii=False)
with open("cidia-studies.json", "w") as f:
    f.write(result_json)
