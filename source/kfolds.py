import numpy as np
import itertools


def floor(i):
	return int(np.floor(i))


def setdiff(A, B):
	X = [x for x in A]
	for i in B:
		if i in A:
			X.remove(i)
	return X


def fit(X, y, estimator, cv_results_, best_score_, best_params_, param_grid, cv_n):
	kernels = ['rbf']
	Cs = [1.0]
	degrees = [3]
	gammas = ['auto']
	for key in param_grid:
		if key == 'kernel':
			kernels = param_grid['kernel']
		elif key == 'C':
			Cs = param_grid['C']
		elif key == 'degree':
			degrees = param_grid['degree']
		elif key == 'gamma':
			gammas = param_grid['gamma']

	cv_results_['param_kernel'] = []
	cv_results_['param_gamma'] = []
	cv_results_['param_degree'] = []
	cv_results_['param_C'] = []
	cv_results_['params'] = []
	for i in range(cv_n):
		cv_results_['split' + i + 'test_score'] = []
		cv_results_['split' + i + 'train_score'] = []
	cv_results_['mean_train_score'] = []
	cv_results_['mean_test_score'] = []
	all_combinations = [a for a in itertools.product(kernels, Cs, degrees, gammas)]

	n = len(X)
	N = range(n)
	for params in all_combinations:
		estimator.set_params('kernel'=params[0], 'C'=parmas[1], 'degree'=params[2], 'gammas'=params[3])
		cv_results_['params'].append({'kernel': params[0], 'C': parmas[1], 'degree': params[2], 'gammas': params[3]})
		cv_results_['param_kernel'].append(params[0])
		cv_results_['param_gamma'].append(params[3])
		cv_results_['param_degree'].append(params[2])
		cv_results_['param_C'].append(params[1])

		# kfold starts from here
		for i in range(cv_n):
			T = range(floor(n * (i - 1) / cv_n), floor(n * i / cv_n))
			S = setdiff(N, T)
			Xi = X[S, :]
			yi = y[S, :]
			Xt = X[T, :]
			yt = y[T, :]
			model = estimator.fit(Xi, yi)
			cv_results_['split' + i + 'train_score'].append(model.score(Xi, yi))
			cv_results_['split' + i + 'train_score'].append(model.score(Xt, yt))



