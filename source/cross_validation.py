"""
Python Module to generate a csv of the results of cross validation
Reference: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
author: mohith
"""

from pandas import DataFrame
from sklearn import svm
from sklearn.model_selection import GridSearchCV
def cross_validation(image_vector,image_labels,C,degree,gamma,outcsv):
	#parameters = {'kernel':kernel,'C':C,'gamma':gamma}
	param_grid = [{'C': C, 'kernel': ['linear']},
  {'C': C, 'gamma': gamma, 'kernel': ['rbf']},
   {'C': C, 'degree': degree, 'kernel': ['poly']}]
	svr = svm.SVC()
	clf = GridSearchCv(svr,param_grid)
	clf.fit(image_vector,image_labels)
	df = DataFrame(data=clf.cv_results_)
	DataFrame.to_csv(df,outcsv)
	return clf