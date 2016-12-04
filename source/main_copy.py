"""
main file to character recognition
Reference: http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf
github: https://github.com/Adarsh-Barik/CharRecognitionPy
Discussion: https://docs.google.com/document/d/1Ccrn5BP424HiEsqlq86opxBvWaF_ki8TPLz7nwiEi70/edit
author: mohith, adarsh
"""
from preprocessing import preprocess
from cross_validation_cl import start_cross_validation
from effect_of_sample_size import effect_of_sample_size


# generate image vectors and corresponding labels
image_vectors, image_labels, image_names = preprocess()

# parameters to test for
parameters_lin = {'kernel': ['linear'], 'C': [1, 10, 20, 30]}
parameters_poly = {'kernel': ['poly'], 'C': [1, 10, 20, 30], 'degree': [2, 3, 5]}
parameters_rbf = {'kernel': ['rbf'], 'C': [1, 10, 20, 30], 'gamma': [0.000001, 0.0001, 0.1, 1]}

# start cross validation: default svm type - one-vs-one
# kfold: cv_n=3 (default)
start_cross_validation(image_vectors, image_labels, "../data/storage/", param_lin=parameters_lin, param_poly=parameters_poly, param_rbf=parameters_rbf)
# bootstrap: cv_n=3 (default)
start_cross_validation(image_vectors, image_labels, "../data/storage/", param_lin=parameters_lin, param_poly=parameters_poly, param_rbf=parameters_rbf, cv_type='bootstrap')

# effect of changing svm type to one-vs-rest, kfold
start_cross_validation(image_vectors, image_labels, "../data/storage/ovr/", param_lin=parameters_lin, param_poly=parameters_poly, param_rbf=parameters_rbf, svm_type='ovr')

# effect of changing sample size
effect_of_sample_size(image_vectors, image_labels, "../data/storage/samplesize/", n_range=[50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000], param_lin=parameters_lin, param_poly=parameters_poly, param_rbf=parameters_rbf)
