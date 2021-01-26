import json
from dicomweb_client.api import DICOMwebClient


studies_file = 'server-studies.json'


client = DICOMwebClient(
    url="https://cidia-pacs.ufrgs.dev",
    qido_url_prefix="/dicom-web",
    wado_url_prefix="/dicom-web",
    #     wado_url_prefix="pacs/wado",
)


studies = client.search_for_studies()
# studies = client.search_for_studies(search_filters={'PatientID': '*11'})

print(len(studies), "studies found")
print()


studies_dict = {}


for study in studies:
    try:
        patient_id = str(study["00100020"]["Value"][0])
        study_uid = str(study["0020000D"]["Value"][0])

        studies_dict[patient_id] = {
            "patientID": patient_id,
            "studyUID": study_uid
        }
        print(studies_dict[patient_id])
        print()
    except:
        print(study)
        break


with open(studies_file, 'w') as file:
    file.write(json.dumps(studies_dict, indent=2, ensure_ascii=False))

print("All studies saved!")
