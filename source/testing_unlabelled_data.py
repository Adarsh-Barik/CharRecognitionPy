from sys import version_info
if version_info < (3, 0):
	import ConfigParser
else:
	import configparser as ConfigParser
 
from preprocessing import key_to_descriptor_array, check_config_file, bmp_to_key, generate_descriptors, get_image_vectors, preprocess
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from os import listdir, path


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
	imagename = '../data/testResized/' + file.decode('UTF-8') + '.Bmp'
	outkey = test_dir + 'keys/' + file.decode('UTF-8') + '.Bmp.key'
	print (imagename, outkey)
	bmp_to_key(imagename, outkey)



print ("Generating test descriptors...")
key_dir = test_dir + 'keys/'
num_of_files = len([a for a in listdir(key_dir)])
		# allimagenames = np.empty([num_of_files, 1], dtype=str)
alltestimagenames = [0 for i in range(num_of_files)]
alltestnumdescriptors = np.zeros(num_of_files,)
i = 0
for filename in listdir(key_dir):
	name_of_image = path.splitext(filename)[0]
	outkey = key_dir + "/" + filename
	mydescriptorarray = key_to_descriptor_array(outkey)
	num_descriptors = len(mydescriptorarray)
	alltestimagenames[i] = path.splitext(name_of_image)[0]
	alltestnumdescriptors[i] = num_descriptors
	if i == 0:
		alltestdescriptorarray = mydescriptorarray
	else:
		alltestdescriptorarray = np.concatenate((alltestdescriptorarray, mydescriptorarray), axis=0)
	i = i + 1
np.savetxt("../data/storage/alltestdescriptors.txt", alltestdescriptorarray)
np.savetxt("../data/storage/test_image_names.txt", alltestimagenames, fmt="%s")
np.savetxt("../data/storage/alltestnumdescriptors.txt", alltestnumdescriptors)
print ("Done.")

print ("Generating train descriptors...")
train_key_dir = config.get('DEFAULT_CONFIG', 'KEY_DIR')
alldescriptors, allimagenames, allnumdescriptors = generate_descriptors(train_key_dir, True)
print ("Done.")

print ("Generating bag of words...")
num_cluster_centers = int(config.get('RUNTIME_CONFIG', 'NUM_OF_WORDS'))
kmeans = KMeans(num_cluster_centers).fit(alldescriptors)
print ("Done.")

print ("Matching bag of words with test descriptors...")
labels = kmeans.predict(alltestdescriptorarray)
print ("Done.")

print ("Generating test image vectors...")
num_test_samples = int(config.get('TEST_CONFIG', 'NUM_SAMPLES'))
#test_image_vectors = get_image_vectors(num_test_samples, alltestnumdescriptors, labels, False)
count = 0
test_image_vectors = np.zeros((num_test_samples, 310))
for i in range(num_test_samples):
	test_image_vector = np.zeros(310)
	for j in range(int(alltestnumdescriptors[i])):
		index = int(labels[j + count])
		test_image_vector[index] = test_image_vector[index] + 1
		# print len(image_vector)
	test_image_vectors[i] = test_image_vector
	count = count + alltestnumdescriptors[i]
np.savetxt("../data/storage/test_image_vectors.txt", test_image_vectors)
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








