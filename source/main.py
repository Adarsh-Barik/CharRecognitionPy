"""
main file to character recognition
Reference: http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf
github: https://github.com/Adarsh-Barik/CharRecognitionPy
Discussion: https://docs.google.com/document/d/1Ccrn5BP424HiEsqlq86opxBvWaF_ki8TPLz7nwiEi70/edit
author: mohith, adarsh
"""

# REQUIRED PACKAGES #
from sift_descriptor import bmp_to_key, key_to_descriptor_array
from get_cluster_centers import get_cluster_centers
from get_svm_model import get_svm_model
from get_image_vector import get_image_vector
from cross_validation_cl import CrossValidation
from sklearn import svm
from os import listdir, path
import numpy as np
import csv
# to store class objects as it is,
# this is called serializing
from sys import version_info
if version_info < (3, 0):
	import cPickle as pickle
else:
	import pickle as pickle

# PREPROCESSING: IMAGE TO IMAGE VECTOR #
# convert bmp files to corresponding key files
bmp_dir = "../data/trainResized"
key_dir = "../data/trainResizedKeys"

# we should not generate keys everytime we run the code
# make it 0 once keys have been generated
generate_key = 0
if generate_key != 0:
	for filename in listdir(bmp_dir):
		outkey = key_dir + "/" + filename + ".key"
		imagename = bmp_dir + "/" + filename
		if path.exists(imagename) and not path.exists(outkey):
			bmp_to_key(imagename, outkey)
		elif not path.exists(imagename):
			print (imagename, " doesn't exist.")
		else:
			print (outkey, " exists.")


# TRAIN, VALIDATE, TEST #
# We have 6283 labelled samples
# We'll train the model with 60% of sample data ~ 3770
# Validate the model with 20% of sample data ~ 1257
# Test the model with 20% of sample data ~ 1256
# training parameters
num_train_samples = 3770
train_image_name_list = [0 for i in range(num_train_samples)]
train_image_vector = np.zeros((num_train_samples, 310))
train_num_descriptor_array = np.zeros(num_train_samples)
train_labels = [0 for i in range(num_train_samples)]
# validation parameters
num_val_samples = 1257
val_image_name_list = [0 for i in range(num_val_samples)]
val_image_vector = np.zeros((num_val_samples, 310))
val_labels = [0 for i in range(num_val_samples)]
# test parameters
num_test_samples = 1256
test_image_name_list = [0 for i in range(num_test_samples)]
test_image_vector = np.zeros((num_test_samples, 310))
test_labels = [0 for i in range(num_test_samples)]

# concatenate all sift descriptors in a single numpy array to be used in k-means
# i have stored these descriptors in txt file ../data/storage/alldescriptors.txt
# regenerate only if required
alltraindescriptorsaved = 0

if alltraindescriptorsaved != 0:
	iterkey = 0
	for filename in listdir(key_dir):
		iterkey = iterkey + 1
		name_of_image = path.splitext(filename)[0]

		if iterkey <= num_train_samples:
			outkey = key_dir + "/" + filename
			mydescriptorarray = key_to_descriptor_array(outkey)
			train_image_name_list[iterkey - 1] = path.splitext(name_of_image)[0]
			train_num_descriptor_array[iterkey - 1] = len(mydescriptorarray)
			if iterkey == 1:
				alltraindescriptorarray = mydescriptorarray
			else:
				alltraindescriptorarray = np.concatenate((alltraindescriptorarray, mydescriptorarray), axis=0)
		elif (iterkey > num_train_samples and iterkey <= (num_train_samples + num_val_samples)):
			val_image_name_list[iterkey - 1 - num_train_samples] = path.splitext(name_of_image)[0]
		else:
			test_image_name_list[iterkey - 1 - num_train_samples - num_val_samples] = path.splitext(name_of_image)[0]

	np.savetxt("../data/storage/train/alltraindescriptors.txt", alltraindescriptorarray)
	np.savetxt("../data/storage/train/train_image_name_list.txt", train_image_name_list, fmt="%s")
	np.savetxt("../data/storage/train/train_num_descriptor_array.txt", train_num_descriptor_array)
	np.savetxt("../data/storage/val/val_image_name_list.txt", val_image_name_list, fmt="%s")
	np.savetxt("../data/storage/test/test_image_name_list.txt", test_image_name_list, fmt="%s")

else:
	alltraindescriptorarray = np.genfromtxt("../data/storage/train/alltraindescriptors.txt")
	train_image_name_list = np.genfromtxt("../data/storage/train/train_image_name_list.txt", dtype=None)
	train_num_descriptor_array = np.genfromtxt("../data/storage/train/train_num_descriptor_array.txt", dtype=None)
	val_image_name_list = np.genfromtxt("../data/storage/val/val_image_name_list.txt", dtype=None)
	test_image_name_list = np.genfromtxt("../data/storage/test/test_image_name_list.txt", dtype=None)


# geenrate bag of words
# generate 5 words per class , total = 5 * 62 = 310
num_cluster_centers = 310

# it takes lot of time to generate cluster centers
# i have saved them in ../data/storage/clustercenters.txt
cluster_centers_labels_stored = 1

if cluster_centers_labels_stored != 0:
	# max iterations defaults to 300 and we should probably try it to increase
	# and see if centers change (not doing it right now because it takes lot of time to run this)
	kmeans_model = get_cluster_centers(num_cluster_centers, alltraindescriptorarray)
	cluster_centers, labels = kmeans_model.cluster_centers_, kmeans_model.labels_
	pickle.dump(kmeans_model, open("../data/storage/train/kmeans_model.p", 'wb'))
else:
	kmeans_model = pickle.load(open("../data/storage/train/kmeans_model.p", 'rb'))
	cluster_centers, labels = kmeans_model.cluster_centers_, kmeans_model.labels_


# extracting image classes from the csv file
imageclass = 0
if imageclass != 0:
	with open('../data/trainLabels.csv', 'r') as train_labels_file:
		csv_train_labels = csv.reader(train_labels_file)
		labels_list = list(csv_train_labels)

	for i in range(num_train_samples):
		for row in labels_list:
			if row[0] == str(train_image_name_list[i]):
				train_labels[i] = row[1]
	for i in range(num_val_samples):
		for row in labels_list:
			if row[0] == str(val_image_name_list[i]):
				val_labels[i] = row[1]
	for i in range(num_test_samples):
		for row in labels_list:
			if row[0] == str(test_image_name_list[i]):
				test_labels[i] = row[1]
	np.savetxt("../data/storage/train/train_labels.txt", train_labels, fmt="%s")
	np.savetxt("../data/storage/val/val_labels.txt", val_labels, fmt="%s")
	np.savetxt("../data/storage/test/test_labels.txt", test_labels, fmt="%s")
else:
	train_labels = np.genfromtxt("../data/storage/train/train_labels.txt", dtype='U')
	val_labels = np.genfromtxt("../data/storage/val/val_labels.txt", dtype='U')
	test_labels = np.genfromtxt("../data/storage/test/test_labels.txt", dtype='U')

# generating vector for svm
vectorarray = 0
if vectorarray != 0:
	count = 0
	image_vector_array = []
	for i in range(num_train_samples):
		image_vector = np.zeros(310)
		for j in range(int(train_num_descriptor_array[i])):
			index = int(labels[j + count])
			image_vector[index] = image_vector[index] + 1
			# print len(image_vector)
		train_image_vector[i] = image_vector
		count = count + train_num_descriptor_array[i]
	np.savetxt("../data/storage/train/train_image_vector.txt", train_image_vector)
else:
	train_image_vector = np.genfromtxt("../data/storage/train/train_image_vector.txt")

# lets run svm
# by default for rbf kernel
store_svm_model = 1

if store_svm_model:
	# good para C=2.5, 3.5, gamma=auto 35%, C=4.5, gamma=auto 37%
	svm_model = get_svm_model(train_image_vector, train_labels, Cpara=30.)
	pickle.dump(svm_model, open("../data/storage/train/svm_model.p", 'wb'))
else:
	svm_model = pickle.load(open("../data/storage/train/svm_model.p", 'rb'))

# VALIDATE AND TEST #
# get val image vectors
for i in range(num_val_samples):
	if type(val_image_name_list[i]) == np.int64:
		keyfile = key_dir + "/" + str(val_image_name_list[i]) + ".Bmp.key"
	else:
		keyfile = key_dir + "/" + val_image_name_list[i] + ".Bmp.key"
	descriptorarray = key_to_descriptor_array(keyfile)
	if len(descriptorarray) != 0:
		val_image_vector[i] = get_image_vector(kmeans_model, descriptorarray)

# get test image vectors
for i in range(num_test_samples):
	if type(test_image_name_list[i]) == np.int64:
		keyfile = key_dir + "/" + str(test_image_name_list[i]) + ".Bmp.key"
	else:
		keyfile = key_dir + "/" + test_image_name_list[i] + ".Bmp.key"
	descriptorarray = key_to_descriptor_array(keyfile)
	if len(descriptorarray) != 0:
		test_image_vector[i] = get_image_vector(kmeans_model, descriptorarray)

length1, width1 = train_image_vector.shape
length2 = len(val_image_vector)
length3 = len(test_image_vector)

all_image_vectors = np.zeros((length1 + length2 + length3, width1))
all_image_labels = np.zeros(length1 + length2 + length3, dtype='<U1')
all_image_vectors[0:length1, :] = train_image_vector
all_image_labels[0:length1] = train_labels
all_image_vectors[length1:length1 + length2, :] = val_image_vector
all_image_labels[length1:length1 + length2] = val_labels
all_image_vectors[length1 + length2:length1 + length2 + length3, :] = test_image_vector
all_image_labels[length1 + length2:length1 + length2 + length3] = test_labels

parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 20, 30, 50], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
svr = svm.SVC()
clf = CrossValidation(svr, parameters, cv_n=10)
clf.fit(all_image_vectors, all_image_labels)

# # Adding train + validation dataset to get train vectors
# trainvalue = 0
# if trainvalue != 0:
# 	trainval_image_vector = np.concatenate((train_image_vector, val_image_vector))
# 	trainval_labels = np.concatenate((np.asarray(train_labels), np.asarray(val_labels)))
# 	np.savetxt("../data/storage/train/trainval_image_vector.txt", trainval_image_vector)
# 	np.savetxt("../data/storage/train/trainval_labels.txt", trainval_labels, fmt="%s")
# else:
# 	trainval_image_vector = np.genfromtxt("../data/storage/train/trainval_image_vector.txt")
# 	trainval_labels = np.genfromtxt("../data/storage/train/trainval_labels.txt")


# # lets tune hyperparameter C
# '''trainscore = np.zeros(15)
# valscore = np.zeros(15)
# testscore = np.zeros(15)
# myrange = [0.1, 0.5, 1.0, 5., 10., 15., 20., 25., 30., 35., 40., 50., 60., 70., 100]
# for i in range(len(myrange)):
# 	svm_model_gen = get_svm_model(train_image_vector, train_labels, Cpara=1.0 * myrange[i])
# 	trainscore[i] = svm_model_gen.score(train_image_vector, train_labels)
# 	valscore[i] = svm_model_gen.score(val_image_vector, val_labels)
# 	testscore[i] = svm_model_gen.score(test_image_vector, test_labels)'''

# plots for hyperparameter estimation (C)
# plot_needed = 0

# if plot_needed:
# 	import matplotlib.pyplot as plt
# 	fig = plt.figure()
# 	l1, l2, l3 = plt.plot(myrange, trainscore, 'ro', myrange, valscore, 'bs', myrange, testscore, 'k^')
# 	plt.xlabel('C')
# 	plt.ylabel('Accuracy')
# 	plt.title('Parameter Estimation for C')
# 	fig.legend((l1, l2, l3), ('Training', 'Validation', 'Testing'), loc='lower right')
# 	plt.show()

# getting svm model for one vs rest
# store_svm_model_ovr = 0
# if store_svm_model_ovr:
# 	# good para C=2.5, 3.5, gamma=auto 35%, C=4.5, gamma=auto 37%
# 	svm_model_ovr = get_svm_model(train_image_vector, train_labels, Cpara=30., svmtype='ovr')
# 	pickle.dump(svm_model_ovr, open("../data/storage/train/svm_model_ovr.p", 'wb'))
# else:
# 	svm_model_ovr = pickle.load(open("../data/storage/train/svm_model_ovr.p", 'rb'))

# # testing against different types of kernels
# strore_diff_kernel_svm = 0
# if strore_diff_kernel_svm:
# 	svm_model_lin = get_svm_model(train_image_vector, train_labels, Cpara=30., kernelpara='linear')
# 	pickle.dump(svm_model_lin, open("../data/storage/train/svm_model_lin.p", 'wb'))
# 	svm_model_poly = get_svm_model(train_image_vector, train_labels, Cpara=30., kernelpara='poly')
# 	pickle.dump(svm_model_poly, open("../data/storage/train/svm_model_poly.p", 'wb'))
# else:
# 	svm_model_lin = pickle.load(open("../data/storage/train/svm_model_lin.p", 'rb'))
# 	svm_model_poly = pickle.load(open("../data/storage/train/svm_model_poly.p", 'rb'))
