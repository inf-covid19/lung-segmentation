from dicomweb_client.api import DICOMwebClient
from io import BytesIO

import numpy as np
from PIL import Image


client = DICOMwebClient(
    url="https://cidia-pacs.ufrgs.dev",
    qido_url_prefix="pacs/dicom-web",
    wado_url_prefix="pacs/dicom-web",
    # wado_url_prefix="pacs/wado",
)

print("client")
print(client)
print()

# studies = client.search_for_studies()


# print('studies')
# print(studies)
# print()


# study = client.search_for_studies(search_filters={'PatientID': 'EXAME1'})


# print('study')
# print(study)
# print()

# instance = client.retrieve_instance(
#     study_instance_uid='1.2.840.113704.7.1.0.1364618414717319.1587167780.0',
#     series_instance_uid='1.2.840.113704.7.32.03704.7.1.0.1364618414717319.1587167808.1120',
#     sop_instance_uid='1.2.840.113704.7.1.0.192317454216202.1587240222.2268',
# )
# instance = client.retrieve_instance(
#     study_instance_uid='1.2.840.113704.7.1.0.253219132234129147.1587168555.1281',
#     series_instance_uid='1.2.840.113704.7.32.004.7.1.0.253219132234129147.1587168635.3928',
#     sop_instance_uid='1.2.840.113704.7.1.0.192317454216202.1587243260.27398',
# )

# print('instance')
# print(instance)
# print()


# print("Writing test file")
# instance.save_as('teste-dicom-exame10')
# print("File saved.")


frames = client.retrieve_instance_frames(
    study_instance_uid='1.2.840.113704.7.1.0.253219132234129147.1587168555.1281',
    series_instance_uid='1.2.840.113704.7.32.004.7.1.0.253219132234129147.1587168635.3928',
    sop_instance_uid='1.2.840.113704.7.1.0.192317454216202.1587243260.27398',
    frame_numbers=[1],
    media_types=('image/jpeg', ),
)

print('frames', len(frames))
# print(frames)
print()

out_f = 'teste-pac-server'

for i, frame in enumerate(frames):
    filename = f"{out_f}/frame-{i}.jpeg"

    with open(filename, 'wb') as outfile:
        outfile.write(frame)
    # image = Image.open(BytesIO(frames[0]))

    # print('image', image)

    image.save(filename)

    array = np.array(image)

    print('array', array)

    print('image shape', array.shape)

    img = Image.fromarray(array.astype(np.uint8))

    img.save(filename)


# data = client.retrieve_bulkdata(
#     'https://cidia-pacs.ufrgs.dev/pacs/dicom-web/studies')

# print('data')
# print(data)
# print()


# instance = client.retrieve_instance(
#     study_instance_uid='1.2.840.113704.7.1.0.1364618414717319.1587167780.0',
#     series_instance_uid='1.2.840.113704.7.1.0.1364618414717319.1587167780.0',
#     media_types=(('application/dicom', '1.2.840.10008.1.2.4.90', ), )
# )

# print('instance')
# print(instance)
# print()

# exit(0)


# "1.2.840.113704.7.1.0.1364618414717319.1587167780.0"
# "1.2.826.0.1.3680043.8.1055.1.20111103111148288.98361414.79379639"

# exam_1 = "1.2.840.113704.7.1.0.1364618414717319.1587167780.0"
# exam_1 = "1.2.840.113704.7.32.03704.7.1.0.1364618414717319.1587167808.1120"


# metadata = client.retrieve_study_metadata(exam_1)

# print('metadata')
# print(metadata)
# print()

# metadata_datasets = [load_json_dataset(ds) for ds in metadata]
# print('metadata ds')
# print(metadata_datasets)
# print()
