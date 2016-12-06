import ConfigParser
from preprocessing import check_config_file, bmp_to_key, generate_descriptors, get_image_vectors, preprocess
import numpy as np
from sklearn.cluster import Kmeans
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

alltestdescriptors, alltestimagenames, alltestnumdescriptors = generate_descriptors(test_dir + 'keys/', False)
train_key_dir = config.get('DEFAULT_CONFIG', 'KEY_DIR')
alldescriptors, allimagenames, allnumdescriptors = generate_descriptors(train_key_dir, True)

num_cluster_centers = int(config.get('RUNTIME_CONFIG', 'NUM_OF_WORDS'))
kmeans = Kmeans(num_cluster_centers, alldescriptors)

labels = kmeans.predict(alltestdescriptors)

num_test_samples = int(config.get('TEST_CONFIG', 'NUM_SAMPLES'))
test_image_vectors = get_image_vectors(num_test_samples, alltestnumdescriptors, labels, False)

train_image_vectors, train_image_labels, image_names = preprocess()

# bootstrap optimum model C=20, gamma=0.1, kernel=rbf
clf = SVC(C=20, gamma=0.1)
clf.fit(train_image_vectors, train_image_labels)
predicted_labels = clf.predict(test_image_vectors)







