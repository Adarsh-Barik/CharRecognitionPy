# author : adarsh
from sys import exit, version_info
from os import path, listdir, system
if version_info < (3, 0):
	import ConfigParser
else:
	import configparser as ConfigParser
import csv
import numpy as np
from skimage.io import load_sift
from sklearn.cluster import KMeans
if version_info < (3, 0):
	import cPickle as pickle
else:
	import pickle as pickle


def check_config_file():
	if path.exists('../configuration/config'):
		return True
	else:
		return False


def bmp_to_key(imagename, outkey):
	"""converts <image>.Bmp to <image>.key ASCII file"""
	# check if image file exists
	if not path.exists(imagename):
		print ("Image file does not exist.")
		exit()
	# check if sift binary exists
	if not path.exists("./sift"):
		print ("sift binary is missing.")
		exit()
	# convert bmp to ppm
	command1 = "bmptopnm " + imagename + " > temp.ppm"
	system(command1)
	# convert ppm to pgm
	command2 = "ppmtopgm temp.ppm > temp.pgm"
	system(command2)
	# convert pgm to key
	command3 = "./sift <temp.pgm >" + outkey
	system(command3)
	# clean up
	command4 = "rm -f temp.ppm temp.pgm"
	system(command4)
	print ("generated", outkey)


def key_to_descriptor_array(keyfile):
	""" changes keys to an array """
	if not path.exists(keyfile):
		print ("Key file doesn't exist.")
		exit()
	f = open(keyfile)
	my_sift_data = load_sift(f)
	f.close()
	return my_sift_data['data']


def generate_descriptors(key_dir, load_them):
	if load_them:
		alldescriptorarray = np.genfromtxt("../data/storage/alldescriptors.txt")
		allimagenames = np.genfromtxt("../data/storage/image_names.txt")
		allnumdescriptors = np.genfromtxt("../data/storage/allnumdescriptors.txt")
	else:
		num_of_files = len([a for a in listdir(key_dir)])
		# allimagenames = np.empty([num_of_files, 1], dtype=str)
		allimagenames = [0 for i in range(num_of_files)]
		allnumdescriptors = np.zeros(num_of_files,)
		i = 0
		for filename in listdir(key_dir):
			name_of_image = path.splitext(filename)[0]
			outkey = key_dir + "/" + filename
			mydescriptorarray = key_to_descriptor_array(outkey)
			num_descriptors = len(mydescriptorarray)
			allimagenames[i] = path.splitext(name_of_image)[0]
			allnumdescriptors[i] = num_descriptors
			if i == 0:
				alldescriptorarray = mydescriptorarray
			else:
				alldescriptorarray = np.concatenate((alldescriptorarray, mydescriptorarray), axis=0)
			i = i + 1
		np.savetxt("../data/storage/alldescriptors.txt", alldescriptorarray)
		np.savetxt("../data/storage/image_names.txt", allimagenames, fmt="%s")
		np.savetxt("../data/storage/allnumdescriptors.txt", allnumdescriptors)
	return alldescriptorarray, allimagenames, allnumdescriptors


def get_cluster_centers(n, Xarray, load_them, iterations=300):
	"get cluster centers with given number of clusters and numpy array of points"
	if load_them:
		kmeans = pickle.load(open("../data/storage/kmeans_model.p", 'rb'))
	else:
		kmeans = KMeans(n_clusters=n, max_iter=iterations, random_state=0).fit(Xarray)
		pickle.dump(kmeans, open("../data/storage/kmeans_model.p", 'wb'))
	return kmeans.cluster_centers_, kmeans.labels_


def get_image_labels(imagenames, load_them):
	if load_them:
		image_labels = np.genfromtxt("../data/storage/image_labels.txt", dtype='U')
	else:
		with open('../data/trainLabels.csv', 'r') as train_labels_file:
			csv_train_labels = csv.reader(train_labels_file)
			labels_list = list(csv_train_labels)

		image_labels = [0 for i in range(len(imagenames))]
		for i in range(len(imagenames)):
			for row in labels_list:
				if row[0] == str(int(imagenames[i])):
					image_labels[i] = row[1]
		np.savetxt("../data/storage/image_labels.txt", image_labels, fmt="%s")
	return image_labels


def get_image_vectors(numsamples, allnumdescriptors, labels, load_them):
	if load_them:
		image_vectors = np.genfromtxt("../data/storage/image_vectors.txt")
	else:
		count = 0
		image_vectors = np.zeros((numsamples, 310))
		for i in range(numsamples):
			image_vector = np.zeros(310)
			for j in range(int(allnumdescriptors[i])):
				index = int(labels[j + count])
				image_vector[index] = image_vector[index] + 1
				# print len(image_vector)
			image_vectors[i] = image_vector
			count = count + allnumdescriptors[i]
		np.savetxt("../data/storage/image_vectors.txt", image_vectors)
	return image_vectors


def preprocess():
	print ("### Preprocessing started ###")
	# read configuration file
	print ("Checking for config file...")
	config = ConfigParser.ConfigParser()
	if not check_config_file():
		print ("config file is missing. Exiting.")
		exit()
	config.read('../configuration/config')
	print("Done.")

	do_processing = 'True' in config.get('RUNTIME_CONFIG', 'DO_PREPROCESSING')
	if do_processing:
		bmp_dir = config.get('DEFAULT_CONFIG', 'BMP_DIR')
		key_dir = config.get('DEFAULT_CONFIG', 'KEY_DIR')

		print ("Generating key files from images...")
		load_keys_from_file = 'True' in config.get('STORAGE_CONFIG', 'LOAD_KEYS_FROM_FILE')
		if not load_keys_from_file:
			for filename in listdir(bmp_dir):
				outkey = key_dir + "/" + filename + ".key"
				imagename = bmp_dir + "/" + filename
				if path.exists(imagename) and not path.exists(outkey):
					bmp_to_key(imagename, outkey)
				elif not path.exists(imagename):
					print (imagename, " doesn't exist.")
				else:
					print (outkey, " exists.")
		print ("Done.")

		print ("Generating image descriptors from images...")
		load_descriptors_from_file = 'True' in config.get('STORAGE_CONFIG', 'LOAD_DESCRIPTORS_FROM_FILE')
		alldescriptors, allimagenames, allnumdescriptors = generate_descriptors(key_dir, load_descriptors_from_file)
		print ("Done.")

		print ("Generating bag of words...")
		num_cluster_centers = int(config.get('RUNTIME_CONFIG', 'NUM_OF_WORDS'))
		load_bag_of_words = 'True' in config.get('STORAGE_CONFIG', 'LOAD_BAG_OF_WORDS')
		cluster_centers, labels = get_cluster_centers(num_cluster_centers, alldescriptors, load_bag_of_words)
		print("Done.")

		print ("Generating image labels from csv file...")
		load_image_labels = 'True' in config.get('STORAGE_CONFIG', 'LOAD_IMAGE_LABELS')
		image_labels = get_image_labels(allimagenames, load_image_labels)
		print ("Done.")

		print ("Generating image vectors...")
		load_image_vectors = 'True' in config.get('STORAGE_CONFIG', 'LOAD_IMAGE_VECTORS')
		numsamples = len(image_labels)
		image_vectors = get_image_vectors(numsamples, allnumdescriptors, labels, load_image_vectors)
		print ("Done.")

	else:
		print ("Loading variables from text files...")
		image_vectors = np.genfromtxt("../data/storage/image_vectors.txt")
		image_labels = np.genfromtxt("../data/storage/image_labels.txt", dtype='U')
		allimagenames = np.genfromtxt("../data/storage/image_names.txt")
		print ("Done.")

	print ("### Preprocessing finished. ###")
	return image_vectors, image_labels, allimagenames


if __name__ == '__main__':
	image_vectors, image_labels, image_names = preprocess()
