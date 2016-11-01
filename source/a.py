from sift_descriptor import bmp_to_key, key_to_descriptor_array
from get_cluster_centers import get_cluster_centers
from get_image_vector import get_image_vector
from os import listdir, path
import numpy as np
import csv
import pickle as pickle

bmp_dir = "../data/trainResized"
key_dir = "../data/trainResizedKeys"

generate_key = 1
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

alldescriptorsaved = 1

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

num_cluster_centers = 310


cluster_centers_labels_stored = 1

if cluster_centers_labels_stored != 0:
	kmeans_model = get_cluster_centers(num_cluster_centers, alldescriptorarray)
	cluster_centers, labels = kmeans_model.cluster_centers_, kmeans_model.labels_
	pickle.dump(kmeans_model, open("../data/storage/kmeans_model.p", 'wb'))
else:
	kmeans_model = pickle.load(open("../data/storage/kmeans_model.p", 'rb'))
	cluster_centers, labels = kmeans_model.cluster_centers_, kmeans_model.labels_
imageclass = 1
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
	image_class = np.genfromtxt("../data/storage/image_class.txt")
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
		if i == 0:
			image_vector_array = image_vector
		else:
			image_vector_array = np.concatenate((image_vector_array, image_vector), axis=0	)
		count = count + number_descriptor[i]
	np.savetxt("../data/storage/image_vector_array.txt", image_vector_array)
else:
	image_vector_array = np.genfromtxt("../data/storage/image_vector_array.txt")
