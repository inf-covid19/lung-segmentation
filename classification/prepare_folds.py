#train and validation folds to do a Cross Validation for binary classifier which the following classes:
#	covid (contains patients classified as:
#					1 - Tipico, pcr = 1)
#	others (contains patients classified as: 
#					2 - Negativo, pcr = 2
#					3 - Indeterminado, pcr = 2)
import pandas as pd
import random

data_description = 'selected_inputs/annotations.xlsx'

annotations = pd.read_excel(data_description)

covid_patients = []
other_patients = []
c = 0

for i in range(len(annotations)): #just considering the first 50 patients
	if str(annotations["nome"][i]) != "nan":
		c+=1
		p_num = int(annotations["nome"][i][1:])
		p_id = 'C' + str(p_num)
		classification = annotations["Classificação"][i]
		pcr = annotations["PCR_FINAL"][i]
		if classification == "2 - típico" and pcr == 1 and p_num != 53 and p_num != 95 and p_num != 153:#patients without CTs
			covid_patients.append(p_id)
		elif (classification == "1 - negativo" and pcr == 2) or (classification == "3 - indeterminado" and pcr == 2):
			other_patients.append(p_id)
		else:
			c -= 1 #in case of patients that were not included in this analysis

print('included patients from annotated excel file: ', c)
print('')
print('Class other patients has size: ', len(other_patients))
other_p = sorted([int(p[1:]) for p in other_patients])
print(other_p)
print('')
print('Class covid patients has size', len(covid_patients))
covid_p = sorted([int(p[1:]) for p in covid_patients])
print(covid_p)
print('')

all_patients = other_patients + covid_patients

print('Total number of patients in one of the two classes: ', 
	      len(all_patients))
	      
all_p = sorted([int(p[1:]) for p in all_patients])
print(all_p)
print('')

#defining lists of train and validation folds
train_folds = []
validation_folds = []
#defining lists of validation folds by class for sanity check
covid_validation_folds = []
other_validation_folds = []
for i in range(1,5): #select each train and validation fold
	validation = []
	train = []
	covid_validation = []
	other_validation = []
#selecting 14 covid patients to validate
	for i in range(14):
		random_int = random.randint(0, len(covid_patients) - 1)
		validation.append(covid_patients[random_int])
		covid_validation.append(covid_patients[random_int])
		covid_patients.pop(random_int)
#selecting 10 non covid patients to validate
	for i in range(10):
		random_int = random.randint(0, len(other_patients) - 1)
		validation.append(other_patients[random_int])
		other_validation.append(other_patients[random_int])
		other_patients.pop(random_int)
#appending the validation and training folds into the final list
	train = list(set(all_patients) - set(covid_validation))
	train = list(set(train) - set(other_validation))
	validation_folds.append(validation)
	train_folds.append(train)
	covid_validation_folds.append(covid_validation)
	other_validation_folds.append(other_validation)
	
#defining the last fold
validation = []
train = []
covid_validation = []
other_validation = []
#selecting 17 covid patients
for i in range(17):
	random_int = random.randint(0, len(covid_patients) - 1)
	validation.append(covid_patients[random_int])
	covid_validation.append(covid_patients[random_int])
	covid_patients.pop(random_int)
#selecting 12 covid patients
for i in range(12):
	random_int = random.randint(0, len(other_patients) - 1)
	validation.append(other_patients[random_int])
	other_validation.append(other_patients[random_int])
	other_patients.pop(random_int)
#appending the validation and training folds into the final list
	train = list(set(all_patients) - set(covid_validation))
	train = list(set(train) - set(other_validation))
	validation_folds.append(validation)
	train_folds.append(train)
	covid_validation_folds.append(covid_validation)
	other_validation_folds.append(other_validation)
	
print('patients not used in any validation folder:', covid_patients+other_patients) #should print an empty list
print('printing folds: ')
for i in range(5):
	print('')
	ordered_list = sorted([int(p[1:]) for p in validation_folds[i]])
	print('Validation Fold number ', str(i+1),  ':', ['C' + str(p) for p in ordered_list])
	ordered_list = sorted([int(p[1:]) for p in train_folds[i]])
	print('Train Fold number ', str(i+1), ':', ['C' + str(p) for p in ordered_list])
	print('')

print('Sanity Check: ')
for i in range(5):
	l = covid_validation_folds[i] + other_validation_folds[i]
	ordered_l = sorted([int(p[1:]) for p in l])
	print('This should be equal to the validation Fold number ', str(i+1), ':', ['C' + str(p) for p in ordered_l])
	
