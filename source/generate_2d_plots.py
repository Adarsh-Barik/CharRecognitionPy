"""
generates 2d plot for some result (accuracy, time) vs. hyperparameter (C, gamma, degree) for a given kernel
uses GridSearchCV object as main result
author: mohith, adarsh
"""
import matplotlib.pyplot as plt
from cross_validation_cl import CrossValidation


def check_fix_params(fixedparams, myparams):
	for key in fixedparams.keys():
		if fixedparams[key] != myparams[key]:
			return False
	return True


def generate_2d_plots(gridsearchcv, out_dir, fixedparams={'kernel': 'rbf'}, hyperparameter='C', result='accuracy', log=False):
	if type(gridsearchcv) != CrossValidation:
		print("Please provide a valid CrossValidation object.")
		return
	if result == 'accuracy':
		trainresult = 'mean_train_score'
		testresult = 'mean_test_score'
	elif result == 'time':
		trainresult = 'mean_fit_time'
		testresult = 'mean_score_time'
	elif result == 'std_accuracy':
		trainresult = 'std_train_score'
		testresult = 'std_test_score'
	elif result == 'std_time':
		trainresult = 'std_fit_time'
		testresult = 'std_score_time'

	hyper_para_list = []
	train_result_list = []
	test_result_list = []
	tot_params = gridsearchcv.cv_results_['params']
	tot_train_results = gridsearchcv.cv_results_[trainresult]
	tot_test_results = gridsearchcv.cv_results_[testresult]
	for i in range(len(tot_params)):
		if check_fix_params(fixedparams, tot_params[i]):
			hyper_para_list.append(tot_params[i][hyperparameter])
			train_result_list.append(tot_train_results[i])
			test_result_list.append(tot_test_results[i])

	fig = plt.figure()
	l1, l2 = plt.plot(hyper_para_list, train_result_list, 'ro-', hyper_para_list, test_result_list, 'bs-')
	plt.xlabel(hyperparameter)
	plt.ylabel(result)
	if log:
		plt.xscale('log')
	fixedtitle =  hyperparameter + "_"
	for key in fixedparams.keys():
		fixedtitle = fixedtitle + str(key) + "_" + str(fixedparams[key]) + "_"
	fixedtitle = fixedtitle + "_" + gridsearchcv.cv_type + "_"
	plt.title(fixedtitle)
	fig.legend((l1, l2), ('Training', 'Validation'), loc='upper right')
	
	plt.savefig(out_dir + fixedtitle + ".png")
	plt.show()

def plot_all_graphs(gridsearchcv,out_dir):
	degree = list(set(gridsearchcv.cv_results_['param_degree']))
	gamma = list(set(gridsearchcv.cv_results_['param_gamma']))
	C = list(set(gridsearchcv.cv_results_['param_C']))
	kernels = list(set(gridsearchcv.cv_results_['param_kernel']))
	for kernel in kernels:
		if kernel == 'poly':
			for d in degree:
				fixedparams = {'kernel':kernel,'degree':d}
				generate_2d_plots(gridsearchcv,out_dir, fixedparams=fixedparams,hyperparameter = 'C')
			for hyp in C:
				fixedparams = {'kernel':kernel,'C':hyp}
				generate_2d_plots(gridsearchcv, out_dir,fixedparams=fixedparams,hyperparameter = 'degree')
		
		if kernel == 'rbf':
			for g in gammas:
				fixedparams = {'kernel':kernel,'degree':g}
				generate_2d_plots(gridsearchcv, out_dir, fixedparams=fixedparams,hyperparameter = 'C',log=True)
			for hyp in C:
				fixedparams = {'kernel':kernel,'C':hyp}
				generate_2d_plots(gridsearchcv, out_dir, fixedparams=fixedparams,hyperparameter = 'gammas',log=True)
	
		if kernel == 'linear'
			generate_2d_plots(gridsearchcv, out_dir, fixedparams={'kernel'='linear'},hyperparameter = 'C')
