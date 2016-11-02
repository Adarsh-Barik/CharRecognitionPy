"""
runs svm
Reference: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
author: mohith, adarsh
"""

from sklearn.svm import SVC


# kernel can be "linear", "poly", "rbf"
# degree is for polynomial kernel only
def get_svm_model(all_image_vectors, all_image_class, kernelpara='rbf', degree=3, Cpara=1.0, coef0=0.0, gammapara='auto'):
	" generates svm model based "
	clf = SVC(C=Cpara, gamma=gammapara, kernel=kernelpara)
	clf.fit(all_image_vectors, all_image_class)
	return clf
