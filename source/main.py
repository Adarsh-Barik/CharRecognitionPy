"""
main file to character recognition
Reference: http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf
github: https://github.com/Adarsh-Barik/CharRecognitionPy 
Discussion: https://docs.google.com/document/d/1Ccrn5BP424HiEsqlq86opxBvWaF_ki8TPLz7nwiEi70/edit
author: adarsh
"""

from sift_descriptor import bmp_to_key, key_to_descriptor_array
from get_cluster_centers import get_cluster_centers
from os import listdir, path
import numpy as np 


# convert bmp files to corresponding key files 
bmp_dir = "../data/trainResized"
key_dir = "../data/trainResizedKeys"

# we should not generate keys everytime we run the code
# make it 0 once keys have been generated
generate_key = 0

if generate_key:
	for filename in listdir(bmp_dir):
		outkey = key_dir+ "/" + filename + ".key"
		imagename = bmp_dir+ "/" + filename
		if path.exists(imagename) and not path.exists(outkey):
			bmp_to_key(imagename, outkey) 
		elif not path.exists(imagename):
			print imagename, " doesn't exist."
		else:
			print outkey, " exists."   
		
# concatenate all sift descriptors in a single numpy array to be used in k-means
# i have stored these descriptors in txt file ../data/storage/alldescriptors.txt 
# regenerate only if required
alldescriptorsaved = 1

if not alldescriptorsaved:
	iterkey= 0
	for filename in listdir(key_dir):
		iterkey = iterkey + 1
		outkey = key_dir+ "/" + filename
		mydescriptorarray = key_to_descriptor_array(outkey)
		if iterkey == 1:
			alldescriptorarray = mydescriptorarray
		else:
			alldescriptorarray = np.concatenate((alldescriptorarray, mydescriptorarray), axis=0)
	np.savetxt("../data/storage/alldescriptors.txt", alldescriptorarray)
else:
	alldescriptorarray = np.genfromtxt("../data/storage/alldescriptors.txt")

# geenrate bag of words
# generate 5 words per class , total = 5 * 62 = 310
num_cluster_centers = 310

# it takes lot of time to generate cluster centers
# i have saved them in ../data/storage/clustercenters.txt
cluster_centers_stored = 1

if not cluster_centers_stored:
	# max iterations defaults to 300 and we should probably try it to increase 
	# and see if centers change (not doing it right now because it takes lot of time to run this)
	cluster_centers = get_cluster_centers(num_cluster_centers, alldescriptorarray)
	np.savetxt("../data/storage/clustercenters.txt", cluster_centers)
else:
	cluster_centers = np.genfromtxt("../data/storage/clustercenters.txt")
