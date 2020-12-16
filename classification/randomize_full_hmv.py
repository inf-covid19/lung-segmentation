'''
script to randomize dividing them into 5 folds for the binary classification task

two classes: typ covid patients, other patients
total

'''

import pandas as pd
import random
import os

data_description = '../annotations.xlsx'

annotations = pd.read_excel(data_description)

typical_patients = []
other_patients = []

#adding HCPA patients
typical_patients = ['TYP-' + i for i in os.listdir('../HCPA-Processadas/Typical')]
other_patients = ['NEG-' + i for i in os.listdir('../HCPA-Processadas/Negative')]
other_patients += ['ATY-' + i for i in os.listdir('../HCPA-Processadas/Atypical')]
other_patients += ['IND-' + i for i in os.listdir('../HCPA-Processadas/Indeterminate')]

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
			other_patients.append(p_id)
		elif classification == "3 - indeterminado":
			other_patients.append(p_id)
		elif classification == "4 - atípico":
			other_patients.append(p_id)
			

#adding new HMV patients.
new_typ_patients = ['N160', 'N167', 'N170', 'N190', 'N201', 'N203', 'N213', 'N230', 'N238', 'N266', 'N285', 'N286', 'N300']

typical_patients = typical_patients + new_typ_patients

new_other_patients = ['N163', 'N164', 'N166', 'N168', 'N169', 'N174', 'N177', 'N180', 'N181', 'N185', 'N186', 'N188', 'N191', 'N192', 'N193', 'N195', 'N197', 'N206', 'N207', 'N210', 'N212', 'N214', 'N216', 'N221', 'N222', 'N223', 'N226', 'N227', 'N228', 'N229', 'N231', 'N232', 'N235', 'N236', 'N239', 'N243', 'N245', 'N252', 'N253', 'N255', 'N257', 'N260', 'N263', 'N264', 'N265', 'N274', 'N276', 'N278', 'N279', 'N282', 'N283', 'N284', 'N288', 'N289', 'N290', 'N291', 'N295', 'N296', 'N297', 'N299', 'N161', 'N171', 'N172', 'N173', 'N176', 'N178', 'N182', 'N184', 'N189', 'N194', 'N196', 'N200', 'N202', 'N204', 'N205', 'N208', 'N220', 'N234', 'N237', 'N240', 'N242', 'N246', 'N247', 'N249', 'N251', 'N254', 'N256', 'N259', 'N261', 'N267', 'N269', 'N272', 'N273', 'N275', 'N281', 'N293', 'N294', 'N165', 'N175', 'N183', 'N187', 'N198', 'N209', 'N219', 'N233', 'N241', 'N244', 'N248', 'N258', 'N270', 'N271', 'N280']

other_patients = other_patients + new_other_patients

print('')
print('Class typical patients has size', len(typical_patients))
print(typical_patients)
print('')
print('Class non-covid patients has size', len(other_patients))
print(other_patients)

all_patients = other_patients + typical_patients
all_typical_patients = typical_patients.copy()
all_other_patients = other_patients.copy()

print('Total number of patients in one of the 2 classes: ', 
	      len(all_patients))
	      
print(all_patients)
print('')

#defining lists of train and validation folds
train_folds = []
validation_folds = []
#defining lists of validation folds by class to the final display
typical_validation_folds = []
other_validation_folds = []
#random.seed(a=0)
for i in range(1,6): #select each train and validation fold
	typical_range = 24
	other_range = 53
	
	validation = []
	train = []
	typical_validation = []
	other_validation = []
#selecting typical patients to validate
	for i in range(typical_range):
		random_int = random.randint(0, len(typical_patients) - 1)
		validation.append(typical_patients[random_int])
		typical_validation.append(typical_patients[random_int])
		typical_patients.pop(random_int)
#selecting other patients to validate
	for i in range(other_range):
		random_int = random.randint(0, len(other_patients) - 1)
		validation.append(other_patients[random_int])
		other_validation.append(other_patients[random_int])
		other_patients.pop(random_int)		
#appending the validation and training folds into the final list
	all_validation_patients = typical_validation + other_validation
	train = list(set(all_patients) - set(all_validation_patients))

	validation_folds.append(validation)
	train_folds.append(train)
	typical_validation_folds.append(typical_validation)
	other_validation_folds.append(other_validation)
	
print('patients not used in any validation folder:', typical_patients+other_patients) #should print an empty list
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
	test_list.extend(other_validation_folds[i])
	_list = typical_validation_folds[i]
	print('typical validation fold ', str(i+1), ':', _list)
	print('')
	_list = other_validation_folds[i]
	print('other validation fold', str(i+1), ':', _list)
	print('')
if len(set(all_patients) - set(test_list))>0:
	print('error')
	
print('')
print('Printing Train Folds per class: ')
for i in range(5):
	train_typical = []
	train_ = []
	train_other = []
	train_fold = train_folds[i]
	#verifying if any of the patients included in validation fold is in trian fold too
	for valid_item in validation_folds[i]:
		for train_item in train_fold:
			if valid_item == train_item:
				print('error occured with patient: ', train_item)
	for train_patient in train_fold:
		found = False
		for other_patient in all_other_patients:
			if train_patient == other_patient:
				found = True
				train_other.append(train_patient)
		if not found:
			train_typical.append(train_patient)
			
	_l = train_typical
	print('typical train fold ', str(i+1), ':', _l)
	print('')
	_l = train_other
	print('other train fold ', str(i+1), ':', _l)
	print('')
