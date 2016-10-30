"""
runs svm
Reference: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
author: adarsh
"""

from sklearn.svm import SVC


# kernel can be "linear", "poly", "rbf"
# degree is for polynomial kernel only
def get_svm_model(all_image_vectors, all_image_class, kernel='rbf', degree=3, C=1.0, coef0=0.0, gamma='auto'):
	" generates svm model based "
	clf = SVC()
	clf.fit(all_image_vectors, all_image_class)
	return clf
