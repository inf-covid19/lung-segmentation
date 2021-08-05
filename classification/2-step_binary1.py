'''
	Randomize data split for 2-step classifier
	
	Binary classification, with the following classes:
	
	Class Covid: typ
	Class Others: atyp, neg, ind
'''

import pandas as pd
import random

def getExamsLists(annotations_file):
	'''
		Given the path to the annotations file, returns 4 lists containing the exams of each radiologic classification
	'''
	
	annotations = pd.read_csv(annotations_file)
	
	typ_exams = [annotations["nome"][i] for i in range(len(annotations)) if annotations["covid"][i] == "POSITIVE"]
	ind_exams = [annotations["nome"][i] for i in range(len(annotations)) if annotations["covid"][i] == "INDETERMINATE"]
	neg_exams = [annotations["nome"][i] for i in range(len(annotations)) if annotations["covid"][i] == "NEGATIVE"]
	atyp_exams = [annotations["nome"][i] for i in range(len(annotations)) if annotations["covid"][i] == "ATYPICAL"]
	
	return typ_exams, ind_exams, neg_exams, atyp_exams
		
def getClassesLists(class1_labels, class2_labels, typ_exams, ind_exams, neg_exams, atyp_exams):
	'''
		Inputs: 
			2 lists containing the radiologic classifications that define the 2 classes of the problem,
			4 lists containing the exams of each radiologic classification
			
		Outputs:
			2 lists containing the exams of each class
	'''
	classes_dict = {
		"typ": typ_exams,
		"atyp": atyp_exams,
		"ind": ind_exams,
		"neg": neg_exams
	}	
	class1_exams = []
	class2_exams = []

	for c in class1_labels:
		class1_exams += classes_dict[c]
	for c in class2_labels:
		class2_exams += classes_dict[c]
	
	return class1_exams, class2_exams
	
def getRandomExams(class_exams, num_exams):
	'''
		Given the exams of one class and the number of random exams, returns a list of random exams from that class and 
		the original list with that random subset subtracted
	''' 
	
	random_exams = []
	for i in range(num_exams):
		random_int = random.randint(0, len(class_exams) - 1)
		random_exams.append(class_exams[random_int])
		class_exams.pop(random_int)
	
	return class_exams, random_exams
	
def main():

	#defining the data (exams) split of each step
	val_exams_c1 = 24 #class 1 exams destined to validate the whole pipeline (1 + 2 steps)
	val_exams_c2 = 53

	fstep_train_c1 = 36#class 1 exams destined to train the 1 step classifier
	fstep_val_c1 = 12
	fstep_train_c2 = 78
	fstep_val_c2 = 27

	secstep_train_c1 = 36
	secstep_val_c1 = 12
	secstep_train_c2 = 78
	secstep_val_c2 = 27

	val_c1_list = []
	val_c2_list = []
	fstep_train_c1_list = []
	fstep_val_c1_list = []
	fstep_train_c2_list = []
	fstep_val_c2_list = []
	secstep_train_c1_list = []
	secstep_val_c1_list = []
	secstep_train_c2_list = []
	secstep_val_c2_list = []

	annotations_file = '../hmv_hcpa_annotations.csv'
	c1_labels = ["typ"]
	c2_labels = ["ind", "neg", "atyp"]
	typ_exams, ind_exams, neg_exams, atyp_exams = getExamsLists(annotations_file)
	c1_exams, c2_exams = getClassesLists(c1_labels, c2_labels, typ_exams, ind_exams, neg_exams, atyp_exams)
	c1_exams_num, c2_exams_num = len(c1_exams), len(c2_exams)
	print("Class 1 exams: ", c1_exams_num)
	print("Class 2 exams: ", c2_exams_num)
	
	c1_exams, val_c1_list = getRandomExams(c1_exams, val_exams_c1)
	c1_exams, fstep_train_c1_list = getRandomExams(c1_exams, fstep_train_c1)
	c1_exams, fstep_val_c1_list = getRandomExams(c1_exams, fstep_val_c1)
	c1_exams, secstep_train_c1_list = getRandomExams(c1_exams, secstep_train_c1)
	c1_exams, secstep_val_c1_list = getRandomExams(c1_exams, secstep_val_c1)
	
	c2_exams, val_c2_list = getRandomExams(c2_exams, val_exams_c2)
	c2_exams, fstep_train_c2_list = getRandomExams(c2_exams, fstep_train_c2)
	c2_exams, fstep_val_c2_list = getRandomExams(c2_exams, fstep_val_c2)
	c2_exams, secstep_train_c2_list = getRandomExams(c2_exams, secstep_train_c2)
	c2_exams, secstep_val_c2_list = getRandomExams(c2_exams, secstep_val_c2)
	
	assert not(c1_exams) and not(c2_exams) #assert that there's no leftover patients
	assert len(set(val_c1_list + fstep_train_c1_list + fstep_val_c1_list + secstep_train_c1_list + secstep_val_c1_list)) == c1_exams_num
	assert len(set(val_c2_list + fstep_train_c2_list + fstep_val_c2_list + secstep_train_c2_list + secstep_val_c2_list)) == c2_exams_num

	print('')
	print('Class 1')
	print('Pipeline validation exams =', val_c1_list)
	print('First step classifier training exams =', fstep_train_c1_list)
	print('First step classifier validation exams =', fstep_val_c1_list)
	print('Second step classifier training exams =', secstep_train_c1_list)
	print('Second step classifier validation exams =', secstep_val_c1_list)
	print('')
	print('Class 2')
	print('Pipeline validation exams =', val_c2_list)
	print('First step classifier training exams =', fstep_train_c2_list)
	print('First step classifier validation exams =', fstep_val_c2_list)
	print('Second step classifier training exams =', secstep_train_c2_list)
	print('Second step classifier validation exams =', secstep_val_c2_list)
		
if __name__  == "__main__":
	main()

