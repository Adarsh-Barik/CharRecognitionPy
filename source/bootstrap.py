"""
Python Module to generate a csv of the results of cross validation
author: mohith
"""

from sklearn.svm import SVC
import numpy as np
import random
'''def cross_validation(image_vector,image_labels,C,degree,gamma,outcsv):
	#parameters = {'kernel':kernel,'C':C,'gamma':gamma}
	param_grid = [{'C': C, 'kernel': ['linear']},
  {'C': C, 'gamma': gamma, 'kernel': ['rbf']},
   {'C': C, 'degree': degree, 'kernel': ['poly']}]
	svr = svm.SVC()
	clf = GridSearchCv(svr,param_grid)
	clf.fit(image_vector,image_labels)
	df = DataFrame(data=_)
	DataFrame.to_csv(df,outcsv)
	return clf'''


def fit(image_vector, image_labels, paramgrid, cvtype='Bootstrap', cvnum='B'):
	X = image_vector
	y = image_labels
	n, m = X.shape
	for i in range(cvnum):
		u = np.zeros(n)
		S = []
		T = []
		for j in range(n):
			k = round(random.uniform(1, n))
			u[j] = k
			if k not in S:
				S.append(k)
		for a in range(n):
			if a not in S:
				T.append(a)
		S = np.asarray(S)
		T = np.asarray(T)
		clf = SVC()
		Xu = X[u, :]
		yu = y[u, :]
		Xt = X[T, :]
		yt = y[T, :]
		clf.fit(Xu, yu)
		train_score = clf.score(Xu, yu)
		test_score = clf.score(Xt, yt)