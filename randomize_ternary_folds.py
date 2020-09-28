import pandas as pd
import random
import os

data_description = '../annotations.xlsx'

annotations = pd.read_excel(data_description)

typical_patients = []
negative_patients = []
indetermined_patients = []

#adding HMV patients
for i in range(len(annotations)): 
	if str(annotations["nome"][i]) != "nan":
		p_num = int(annotations["nome"][i][1:])
		p_id = 'C' + str(p_num)
		classification = annotations["Classificação"][i]
		pcr = annotations["PCR_FINAL"][i]
		if classification == "2 - típico" and p_num != 53 and p_num != 95 and p_num != 153:
			typical_patients.append(p_id)
		elif classification == "1 - negativo" and p_num != 1:
			negative_patients.append(p_id)
		elif classification == "3 - indeterminado":
			indetermined_patients.append(p_id)

#adding HCPA patients
for patient in os.listdir('../HCPA-Processadas'+ '/' + 'Negative'):
	patient_name = 'NEG-' + patient
	negative_patients.append(patient_name)
	
for patient in os.listdir('../HCPA-Processadas' + '/' + 'Typical'):
	patient_name = 'TYP-' + patient
	typical_patients.append(patient_name)
	
for patient in os.listdir('../HCPA-Processadas' + '/' + 'Indetermined'):
	patient_name = 'IND-' + patient
	indetermined_patients.append(patient_name)

print('Class negative patients has size: ', len(negative_patients))
print(negative_patients)
print('')
print('Class typical patients has size', len(typical_patients))
print(typical_patients)
print('')
print('Class indetermined patients has size', len(indetermined_patients))
print(indetermined_patients)
print('')

all_patients = negative_patients + typical_patients + indetermined_patients
all_typical_patients = typical_patients.copy()
all_negative_patients = negative_patients.copy()
all_indetermined_patients = indetermined_patients.copy()

print('Total number of patients in one of the 3 classes: ', 
	      len(all_patients))
	      
print(all_patients)
print('')

#defining lists of train and validation folds
train_folds = []
validation_folds = []
#defining lists of validation folds by class to the final display
typical_validation_folds = []
negative_validation_folds = []
indetermined_validation_folds = []
#random.seed(a=0)
for i in range(1,6): #select each train and validation fold
	indetermined_range = 12 #there are 60 indetermined patients, so we'll define that every fold will have 12 of them.
	if i == 5:
		typical_range = 23 #107 valid typical patients, 4*21 + 23 = 107
		negative_range = 10 #46 valid negative patients, 4*9 + 10 = 46
	else:
		typical_range = 21
		negative_range = 9
	validation = []
	train = []
	typical_validation = []
	negative_validation = []
	indetermined_validation = []
#selecting typical patients to validate
	for i in range(typical_range):
		random_int = random.randint(0, len(typical_patients) - 1)
		validation.append(typical_patients[random_int])
		typical_validation.append(typical_patients[random_int])
		typical_patients.pop(random_int)
#selecting negative patients to validate
	for i in range(negative_range):
		random_int = random.randint(0, len(negative_patients) - 1)
		validation.append(negative_patients[random_int])
		negative_validation.append(negative_patients[random_int])
		negative_patients.pop(random_int)
#selecting indetermined patients to validate
	for i in range(indetermined_range):
		random_int = random.randint(0, len(indetermined_patients) - 1)
		validation.append(indetermined_patients[random_int])
		indetermined_validation.append(indetermined_patients[random_int])
		indetermined_patients.pop(random_int)		
#appending the validation and training folds into the final list
	all_validation_patients = typical_validation + negative_validation + indetermined_validation
	train = list(set(all_patients) - set(all_validation_patients))
#	print('')
#	print('Debugging: Train folder', train)
#	print('')
	validation_folds.append(validation)
	train_folds.append(train)
	typical_validation_folds.append(typical_validation)
	negative_validation_folds.append(negative_validation)
	indetermined_validation_folds.append(indetermined_validation)
	
print('patients not used in any validation folder:', typical_patients+negative_patients+indetermined_patients) #should print an empty list
print('printing folds: ')
for i in range(5):
	print('')
	_list = validation_folds[i]
	print('Validation Fold number ', str(i+1),  ':', _list)
	_list = train_folds[i]
	print('Train Fold number ', str(i+1), ':', _list)
	print('')

test_list = []
print('Printing Validation Folds per class: ')
for i in range(5):
	test_list.extend(typical_validation_folds[i])
	test_list.extend(negative_validation_folds[i])
	test_list.extend(indetermined_validation_folds[i])
	_list = typical_validation_folds[i]
	print('typical validation fold ', str(i+1), ':', _list)
	print('')
	_list = negative_validation_folds[i]
	print('negative validation fold ', str(i+1), ':', _list)
	print('')
	_list = indetermined_validation_folds[i]
	print('indetermined validation fold', str(i+1), ':', _list)
	print('')
if len(set(all_patients) - set(test_list))>0:
	print('error')
	
print('')
print('Printing Train Folds per class: ')
for i in range(5):
	train_typical = []
	train_negative = []
	train_indetermined = []
	train_fold = train_folds[i]
	#verifying if any of the patients included in validation fold is in trian fold too
	for valid_item in validation_folds[i]:
		for train_item in train_fold:
			if valid_item == train_item:
				print('error occured with patient: ', train_item)
	for train_patient in train_fold:
		found = False
		for negative_patient in all_negative_patients:
			if train_patient == negative_patient:
				found = True
				train_negative.append(train_patient)
		for indetermined_patient in all_indetermined_patients:
			if train_patient == indetermined_patient:
				found = True
				train_indetermined.append(train_patient)
		if not found:
			train_typical.append(train_patient)
			
	_l = train_typical
	print('typical train fold ', str(i+1), ':', _l)
	print('')
	_l = train_negative
	print('negative train fold ', str(i+1), ':', _l)
	print('')
	_l = train_indetermined
	print('indetermined train fold ', str(i+1), ':', _l)
	print('')
