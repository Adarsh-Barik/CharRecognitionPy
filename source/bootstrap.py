"""
Python Module to perform bootstrapping
author: mohith
"""
import numpy as np
import random
import itertools

#generate a uniform distribution of numbers
def uniform(a,n):
	return round(random.uniform(a,n))

#generate all possible combinations of parameters
def get_all_param_combinations(param_grid):
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
	all_combinations = [a for a in itertools.product(kernels, Cs, degrees, gammas)]
	return all_combinations

#fit bootstrap to generated paramgrid
def fit(X, y, estimator, cv_results_, best_score_, best_params_, param_grid, cv_B):
	cv_results_['param_kernel'] = []
	cv_results_['param_gamma'] = []
	cv_results_['param_degree'] = []
	cv_results_['param_C'] = []
	cv_results_['params'] = []
	for i in range(cv_B):
		cv_results_['split' + i + 'test_score'] = []
		cv_results_['split' + i + 'train_score'] = []
	cv_results_['mean_train_score'] = []
	cv_results_['mean_test_score'] = []
	all_combinations = get_all_param_combinations(param_grid)

	for params in all_combinations:
		estimator.set_params(kernel=params[0], C=params[1], degree=params[2], gamma=params[3])
		cv_results_['params'].append({'kernel': params[0], 'C': params[1], 'degree': params[2], 'gammas': params[3]})
		cv_results_['param_kernel'].append(params[0])
		cv_results_['param_gamma'].append(params[3])
		cv_results_['param_degree'].append(params[2])
		cv_results_['param_C'].append(params[1])
		#bootstrap
		n,m = X.shape
		for i in range(cv_B):
			u = np.zeros(n)
			S = []
			T = []
			for j in range(n):
				k = uniform(1,n)
				u[j] = k
				if k not in S:
					S.append(k)
			for a in range(n):
				if a not in S:
					T.append(a)
			S = np.asarray(S)
			T = np.asarray(T)
			Xu = X[u, :]
			yu = y[u, :]
			Xt = X[T, :]
			yt = y[T, :]
			model = estimator.fit(Xu,yu)
			cv_results_['split' + i + 'train_score'].append(model.score(Xu, yu))
			cv_results_['split' + i + 'train_score'].append(model.score(Xt, yt))
			