import csv
import os
import numpy as np
from sift_descriptor import key_to_descriptor_array
'''train_labels=open('../data/trainLabels.csv')
csv_train_labels=csv.reader(train_labels)
imagename_vector=np.genfromtxt("../data/storage/imagename_vector.txt",dtype=None)
image_class=[]
for row in csv_train_labels:
	for i in imagename_vector:
		if row[0]==str(i):
			image_class=np.append(image_class,row[1])
print(image_class)
np.savetxt("../data/storage/image_class.txt", image_class,fmt="%s")
b=[]
key_dir = "../data/trainResizedKeys"
number_descriptor=[]
for filename in os.listdir(key_dir):
	a=os.path.splitext(filename)[0]
	b=np.append(b,os.path.splitext(a)[0])
	outkey = key_dir+ "/" + filename
	mydescriptorarray = key_to_descriptor_array(outkey)
	number_descriptor=np.append(number_descriptor,len(mydescriptorarray))
print(b)
print(number_descriptor)
#np.savetxt("../data/storage/imagename_vector.txt", b, fmt="%s")
np.savetxt("../data/storage/number_descriptor.txt", number_descriptor, fmt="%d")'''
number_descriptor=np.genfromtxt("../data/storage/number_descriptor.txt",dtype=None)
print(number_descriptor)
print(len(number_descriptor))
print(number_descriptor[0])