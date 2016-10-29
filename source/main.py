"""
main file to character recognition
Reference: http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf
github: https://github.com/Adarsh-Barik/CharRecognitionPy
Discussion: https://docs.google.com/document/d/1Ccrn5BP424HiEsqlq86opxBvWaF_ki8TPLz7nwiEi70/edit
author: mohith, adarsh
"""

from sift_descriptor import bmp_to_key, key_to_descriptor_array
from get_cluster_centers import get_cluster_centers
from os import listdir, path
import numpy as np
import csv

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
cluster_centers_labels_stored = 0

if cluster_centers_labels_stored != 0:
	# max iterations defaults to 300 and we should probably try it to increase
	# and see if centers change (not doing it right now because it takes lot of time to run this)
	cluster_centers, labels = get_cluster_centers(num_cluster_centers, alldescriptorarray)
	np.savetxt("../data/storage/clustercenters.txt", cluster_centers)
	np.savetxt("../data/storage/labels.txt", labels)
else:
	cluster_centers = np.genfromtxt("../data/storage/clustercenters.txt")
	labels = np.genfromtxt("../data/storage/labels.txt")

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
	for i in range(len(image_class)):
		image_class[i] = ord(image_class[i])

# generating vector for svm
vectorarray = 1
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
