from cross_validation_cl import start_cross_validation
from os import system, path


def effect_of_sample_size(image_vectors, image_labels, out_dir, n_range=[50, 1000, 3000, 6000], param_lin=None, param_poly=None, param_rbf=None):
	print ("Checking effects of sample size... ")
	for i in n_range:
		print ("size: ", i)
		temp_image_vectors = image_vectors[0:i, :]
		temp_image_labels = image_labels[0:i]
		temp_out_dir = out_dir + "_" + str(i) + "/"
		if not path.exists(temp_out_dir):
			command1 = "mkdir " + temp_out_dir
			system(command1)
		start_cross_validation(temp_image_vectors, temp_image_labels, temp_out_dir, param_lin=param_lin, param_poly=param_poly, param_rbf=param_rbf)
	print ("Done")
