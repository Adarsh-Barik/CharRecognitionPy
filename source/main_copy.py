"""
main file to character recognition
Reference: http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf
github: https://github.com/Adarsh-Barik/CharRecognitionPy
Discussion: https://docs.google.com/document/d/1Ccrn5BP424HiEsqlq86opxBvWaF_ki8TPLz7nwiEi70/edit
author: mohith, adarsh
"""
from preprocessing import preprocess
from sklearn import svm
from cross_validation_cl import CrossValidation


# generate image vectors and corresponding labels
image_vectors, image_labels, image_names = preprocess()

# start cross validation
parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 20, 30, 50], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
svr = svm.SVC()
clf = CrossValidation(svr, parameters, cv_n=10)
clf.fit(image_vectors, image_labels)
