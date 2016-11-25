import numpy as np
import itertools
from random import shuffle


def floor(i):
	return int(np.floor(i))


def setdiff(A, B):
	X = [x for x in A]
	for i in B:
		if i in A:
			X.remove(i)
	return X


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


def kfolds(X_rand, y_rand, estimator, cv_n, n, N, cv_results_):
	# kfold starts from here
	for i in range(cv_n):
		T = range(floor(n * (i) / cv_n), floor(n * (i + 1) / cv_n))
		S = setdiff(N, T)
		# print estimator.get_params()
		# print X.shape
		# print y.shape
		Xi = X_rand[S, :]
		yi = y_rand[S]
		Xt = X_rand[T, :]
		yt = y_rand[T]
		model = estimator.fit(Xi, yi)
		cv_results_['split' + str(i) + 'train_score'].append(model.score(Xi, yi))
		cv_results_['split' + str(i) + 'test_score'].append(model.score(Xt, yt))


def bootstrap(X_rand, y_rand, estimator, B, n, N, cv_results_):
	for i in range(B):
		u = [0 for a in range(n)]
		S = []
		for j in range(n):
			k = np.random.randint(0, n)
			u[j] = k
			if k not in S:
				S.append(k)
		T = setdiff(N, S)
		# print u
		Xu = X_rand[u, :]
		yu = y_rand[u]
		Xt = X_rand[T, :]
		yt = y_rand[T]
		model = estimator.fit(Xu, yu)
		cv_results_['split' + str(i) + 'train_score'].append(model.score(Xu, yu))
		cv_results_['split' + str(i) + 'test_score'].append(model.score(Xt, yt))


def fit(X, y, estimator, cv_results_, best_score_, best_params_, param_grid, cv_n, cv_type):
	cv_results_['param_kernel'] = []
	cv_results_['param_gamma'] = []
	cv_results_['param_degree'] = []
	cv_results_['param_C'] = []
	cv_results_['params'] = []
	cv_results_['mean_train_score'] = []
	cv_results_['mean_test_score'] = []
	for i in range(cv_n):
		cv_results_['split' + str(i) + 'test_score'] = []
		cv_results_['split' + str(i) + 'train_score'] = []
	cv_results_['mean_train_score'] = []
	cv_results_['mean_test_score'] = []
	all_combinations = get_all_param_combinations(param_grid)

	n = len(X)
	N = range(n)
	shuffled_index = range(n)
	shuffle(shuffled_index)
	X_rand = X[shuffled_index, :]
	y_rand = y[shuffled_index]
	for params in all_combinations:
		estimator.set_params(kernel=params[0], C=params[1], degree=params[2], gamma=params[3])
		cv_results_['params'].append({'kernel': params[0], 'C': params[1], 'degree': params[2], 'gammas': params[3]})
		cv_results_['param_kernel'].append(params[0])
		cv_results_['param_gamma'].append(params[3])
		cv_results_['param_degree'].append(params[2])
		cv_results_['param_C'].append(params[1])

		if cv_type == 'kfold':
			kfolds(X_rand, y_rand, estimator, cv_n, n, N, cv_results_)
		elif cv_type == 'bootstrap':
			bootstrap(X_rand, y_rand, estimator, cv_n, n, N, cv_results_)
	for i in range(len(cv_results_['params'])):
		cv_results_['mean_train_score'].append(0)
		cv_results_['mean_test_score'].append(0)
		for j in range(cv_n):
			cv_results_['mean_train_score'][i] += cv_results_['split' + str(j) + 'train_score'][i]
			cv_results_['mean_test_score'][i] += cv_results_['split' + str(j) + 'test_score'][i]
		cv_results_['mean_train_score'][i] /= cv_n * 1.0
		cv_results_['mean_test_score'][i] /= cv_n * 1.0

