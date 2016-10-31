"""
main file to character recognition
Reference: http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf
github: https://github.com/Adarsh-Barik/CharRecognitionPy
Discussion: https://docs.google.com/document/d/1Ccrn5BP424HiEsqlq86opxBvWaF_ki8TPLz7nwiEi70/edit
author: mohith, adarsh
"""

from sift_descriptor import bmp_to_key, key_to_descriptor_array
from get_cluster_centers import get_cluster_centers
from get_svm_model import get_svm_model
from get_image_vector import get_image_vector
from os import listdir, path
import numpy as np
import csv
# to store class objects as it is,
# this is called serializing
import pickle

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

# concatenate all sift descriptors in a single numpy array to be used in k-means
# i have stored these descriptors in txt file ../data/storage/alldescriptors.txt
# regenerate only if required
alldescriptorsaved = 0

if alldescriptorsaved != 0:
	iterkey = 0
	imagename_vector = []
	number_descriptor = []
	for filename in listdir(key_dir):
		iterkey = iterkey + 1
		outkey = key_dir + "/" + filename
		mydescriptorarray = key_to_descriptor_array(outkey)
		a = path.splitext(filename)[0]
		imagename_vector = np.append(imagename_vector, path.splitext(a)[0])
		number_descriptor = np.append(number_descriptor, len(mydescriptorarray))
		if iterkey == 1:
			alldescriptorarray = mydescriptorarray
		else:
			alldescriptorarray = np.concatenate((alldescriptorarray, mydescriptorarray), axis=0)
	np.savetxt("../data/storage/alldescriptors.txt", alldescriptorarray)
	np.savetxt("../data/storage/imagename_vector.txt", imagename_vector, fmt="%s")
	np.savetxt("../data/storage/number_descriptor.txt", number_descriptor)
else:
	alldescriptorarray = np.genfromtxt("../data/storage/alldescriptors.txt")

# geenrate bag of words
# generate 5 words per class , total = 5 * 62 = 310
num_cluster_centers = 310

# it takes lot of time to generate cluster centers
# i have saved them in ../data/storage/clustercenters.txt
cluster_centers_labels_stored = 1

if cluster_centers_labels_stored != 0:
	# max iterations defaults to 300 and we should probably try it to increase
	# and see if centers change (not doing it right now because it takes lot of time to run this)
	kmeans_model = get_cluster_centers(num_cluster_centers, alldescriptorarray)
	cluster_centers, labels = kmeans_model.cluster_centers_, kmeans_model.labels_
	pickle.dump(kmeans_model, open("../data/storage/kmeans_model.p", 'wb'))
else:
	kmeans_model = pickle.load("../data/storage/kmeans_model.p")
	cluster_centers, labels = kmeans_model.cluster_centers_, kmeans_model.labels_


# extracting image classes from the csv file
imageclass = 0
if imageclass != 0:
	train_labels = open('../data/trainLabels.csv')
	csv_train_labels = csv.reader(train_labels)
	imagename_vector = np.genfromtxt("../data/storage/imagename_vector.txt", dtype=None)
	image_class = []
	for row in csv_train_labels:
		for i in imagename_vector:
			if row[0] == str(i):
				image_class = np.append(image_class, ord(row[1]))
	np.savetxt("../data/storage/image_class.txt", image_class,)
else:
	image_class = np.genfromtxt("../data/storage/imagename_vector.txt")

# generating vector for svm
vectorarray = 0
if vectorarray != 0:
	count = 0
	image_vector_array = []
	imagename_vector = np.genfromtxt("../data/storage/imagename_vector.txt", dtype=None)
	number_descriptor = np.genfromtxt("../data/storage/number_descriptor.txt", dtype=None)
	labels = np.genfromtxt("../data/storage/labels.txt", dtype=None)
	for i in range(len(imagename_vector)):
		image_vector = np.zeros((1, 310))
		for j in range(int(number_descriptor[i])):
			index = int(labels[j + count])
			image_vector[0][index] = image_vector[0][index] + 1
			# print len(image_vector)
		if i == 0:
			image_vector_array = image_vector
		else:
			image_vector_array = np.concatenate((image_vector_array, image_vector), axis=0	)
		count = count + number_descriptor[i]
	np.savetxt("../data/storage/image_vector_array.txt", image_vector_array)
else:
	image_vector_array = np.genfromtxt("../data/storage/image_vector_array.txt")

# lets run svm
# by default for rbf kernel
store_svm_model = 1

if store_svm_model:
	svm_model = get_svm_model(image_vector_array, image_class)
	pickle.dump(svm_model, open("../data/storage/svm_model.p", 'wb'))
else:
	svm_model = pickle.load("../data/storage/svm_model.p")

# now that we have the model, let the prediction begin
test_bmp_dir = "../data/testResized"
test_key_dir = "../data/testResizedKeys"

test_generate_key = 1
if test_generate_key != 0:
	for imagefilename in listdir(test_bmp_dir):
		testoutkey = test_key_dir + "/" + imagefilename + ".key"
		testimagename = test_bmp_dir + "/" + imagefilename
		if path.exists(testimagename) and not path.exists(testoutkey):
			bmp_to_key(testimagename, testoutkey)
		elif not path.exists(testimagename):
			print (testimagename, " doesn't exist.")
		else:
			print (testoutkey, " exists.")
alltestdescriptosaved = 1

if alltestdescriptosaved != 0:
	testiterkey = 0
	test_imagename_vector = []
	test_image_vector = []
	test_number_descriptor = []
	test_predicted_labels = []
	for testfilename in listdir(test_key_dir):
		testiterkey = testiterkey + 1
		testoutkey = test_key_dir + "/" + testfilename
		mytestdescriptorarray = key_to_descriptor_array(testoutkey)
		my_image_vector = get_image_vector(kmeans_model, mytestdescriptorarray)
		my_image_label = svm_model.predict(my_image_vector)
		a = path.splitext(testfilename)[0]
		test_image_vector = np.append(test_image_vector, my_image_vector)
		test_imagename_vector = np.append(test_imagename_vector, path.splitext(a)[0])
		test_number_descriptor = np.append(test_number_descriptor, len(mytestdescriptorarray))
		test_predicted_labels = np.append(test_predicted_labels, my_image_label)
		if testiterkey == 1:
			alltestdescriptorarray = mytestdescriptorarray
		else:
			alltestdescriptorarray = np.concatenate((alltestdescriptorarray, mytestdescriptorarray), axis=0)
	np.savetxt("../data/storage/alltestdescriptors.txt", alltestdescriptorarray)
	np.savetxt("../data/storage/test_imagename_vector.txt", test_imagename_vector, fmt="%s")
	np.savetxt("../data/storage/test_image_vector.txt", test_image_vector, fmt="%s")
	np.savetxt("../data/storage/test_number_descriptor.txt", test_number_descriptor)
	np.savetxt("../data/storage/test_predicted_labels.txt", test_predicted_labels)
else:
	alltestdescriptorarray = np.genfromtxt("../data/storage/alltestdescriptors.txt")
	test_imagename_vector = np.genfromtxt("../data/storage/test_imagename_vector.txt")
	test_image_vector = np.genfromtxt("../data/storage/test_image_vector.txt")
	test_number_descriptor = np.genfromtxt("../data/storage/test_number_descriptor.txt")
	test_predicted_labels = np.genfromtxt("../data/storage/test_predicted_labels.txt")
