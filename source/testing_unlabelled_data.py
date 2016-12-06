from sys import version_info
if version_info < (3, 0):
	import ConfigParser
else:
	import configparser as ConfigParser
 
from preprocessing import check_config_file, bmp_to_key, generate_descriptors, get_image_vectors, preprocess
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC


print ("### Testing started ###")
# read configuration file
print ("Checking for config file...")
config = ConfigParser.ConfigParser()
if not check_config_file():
	print ("config file is missing. Exiting.")
	exit()
config.read('../configuration/config')
print("Done.")

test_csv = config.get('TEST_CONFIG', 'TEST_CSV')
test_dir = config.get('TEST_CONFIG', 'TEST_DIR')
test_data = np.genfromtxt(test_csv, delimiter=',', dtype="|S5")
test_files = test_data[:, 0]

for file in test_files:
	imagename = '../data/testResized/' + file
	outkey = test_dir + 'keys/' + imagename + '.key'
	bmp_to_key(imagename, outkey)
print ("Generating test descriptors...")
alltestdescriptors, alltestimagenames, alltestnumdescriptors = generate_descriptors(test_dir + 'keys/', False)
print ("Done.")
print ("Generating train descriptors...")
train_key_dir = config.get('DEFAULT_CONFIG', 'KEY_DIR')
alldescriptors, allimagenames, allnumdescriptors = generate_descriptors(train_key_dir, True)
print ("Done.")

print ("Generating bag of words...")
num_cluster_centers = int(config.get('RUNTIME_CONFIG', 'NUM_OF_WORDS'))
kmeans = KMeans(num_cluster_centers, alldescriptors)
print ("Done.")

print ("Matching bag of words with test descriptors...")
labels = kmeans.predict(alltestdescriptors)
print ("Done.")

print ("Generating test image vectors...")
num_test_samples = int(config.get('TEST_CONFIG', 'NUM_SAMPLES'))
test_image_vectors = get_image_vectors(num_test_samples, alltestnumdescriptors, labels, False)
print ("Done.")

print ("Loading train image vectors, labels...")
train_image_vectors, train_image_labels, image_names = preprocess()
print ("Done.")
# bootstrap optimum model C=20, gamma=0.1, kernel=rbf
print ("Training SVM...")
clf = SVC(C=20, gamma=0.1)
clf.fit(train_image_vectors, train_image_labels)
print ("Done.")

print ("Predicting...")
predicted_labels = clf.predict(test_image_vectors)
print ("Done.")








