"""
get image_vector from descriptor array
author: adarsh
"""

import numpy as np


def get_image_vector(kmeans_model, image_descriptor_array):
	" return image vectors using image descriptors and kmeans model"
	length_of_descriptor_array = len(image_descriptor_array)
	labels = kmeans_model.predict(image_descriptor_array)
	image_vector = np.zeros((1, 310))
	for i in range(length_of_descriptor_array):
		index = int(labels[i])
		image_vector[0][index] = image_vector[0][index] + 1
	return image_vector
