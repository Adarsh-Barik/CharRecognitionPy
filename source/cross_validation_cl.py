import fit
from sklearn.svm import SVC
from generate_2d_plots import plot_all_graphs
import pandas
from sys import version_info
if version_info < (3, 0):
	import cPickle as pickle
else:
	import pickle as pickle


class CrossValidation():
	""" class to run cross-validation"""
	def __init__(self, estimator, param_grid, cv_type='kfold', cv_n=3):
		self.estimator = estimator
		self.param_grid = param_grid
		self.cv_type = cv_type
		self.cv_n = cv_n
		self.cv_results_ = {}
		self.best_score_ = 0.
		self.best_params_ = None

	def fit(self, X, y):
		fit.fit(X, y, self.estimator, self.cv_results_, self.best_score_, self.best_params_, 
			self.param_grid, self.cv_n, self.cv_type)
		best_index = self.cv_results_['mean_test_score'].index(max(self.cv_results_['mean_test_score']))
		self.best_score_ = max(self.cv_results_['mean_test_score'])
		self.best_params_ = self.cv_results_['params'][best_index]


def store_cv_object(cvobj, out_pickle_file, out_csv_file):
	print (" Storing cross validation object..")
	pickle.dump(cvobj, open(out_pickle_file, 'wb'))
	df = pandas.DataFrame(cvobj.cv_results_)
	df.to_csv(out_csv_file)
	print ("Done.")

	
def start_cross_validation(image_vectors, image_labels, out_dir, cv_type='kfold', cv_n=3, param_lin=None, param_poly=None, param_rbf=None, svm_type=None):
	print ("Starting cross validation...")
	svr = SVC(decision_function_shape=svm_type)

	if param_lin:
		print ("cross validation for linear kernel")
		clf_lin = CrossValidation(svr, param_lin, cv_type=cv_type, cv_n=cv_n)
		clf_lin.fit(image_vectors, image_labels)
		if svm_type == None:
			svt = ""
		else:
			svt = svm_type
		out_p = out_dir + "clf_lin_" + cv_type + str(cv_n) + svt + ".p"  
		out_csv = out_dir + "clf_lin_" + cv_type + str(cv_n) + svt + ".csv"  
		store_cv_object(clf_lin, out_p, out_csv)
		plot_all_graphs(clf_lin, out_dir)

	if param_poly:
		print ("cross validation for poly kernel")
		clf_poly = CrossValidation(svr, param_poly, cv_type=cv_type, cv_n=cv_n)
		clf_poly.fit(image_vectors, image_labels)
		if svm_type == None:
			svt = ""
		
		else:
			svt = svm_type

		out_p = out_dir + "clf_poly_" + cv_type + str(cv_n) + svt + ".p"  
		out_csv = out_dir + "clf_poly_" + cv_type + str(cv_n) + svt + ".csv"  
		store_cv_object(clf_poly, out_p, out_csv)
		plot_all_graphs(clf_poly, out_dir)

	if param_rbf:
		print ("cross validation for rbf kernel")
		clf_rbf = CrossValidation(svr, param_rbf, cv_type=cv_type, cv_n=cv_n)
		clf_lin.fit(image_vectors, image_labels)
		if svm_type == None:
			svt = ""
		else:
			svt = svm_type
		out_p = out_dir + "clf_rbf_" + cv_type + str(cv_n) + svt + ".p"  
		out_csv = out_dir + "clf_rbf_" + cv_type + str(cv_n) + svt + ".csv"  
		store_cv_object(clf_rbf, out_p, out_csv)
		plot_all_graphs(clf_rbf, out_dir)
	print ("Done.")



