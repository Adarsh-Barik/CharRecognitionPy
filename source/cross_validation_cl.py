import fit


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
<<<<<<< HEAD
		if self.cv_type == 'kfold':
			kfolds.fit(X, y, self.estimator, self.cv_results_, self.best_score_, self.best_params_, self.param_grid, self.cv_n)
		else:
			bootstrap.fit(X, y, self.estimator, self.cv_results_, self.best_score_, self.best_score_, self.param_grid, self.cv_n)
=======
		fit.fit(X, y, self.estimator, self.cv_results_, self.best_score_, self.best_params_, self.param_grid, self.cv_n, self.cv_type)
>>>>>>> 4cccee0242e72b19cbc70d44bbe8020e8f2de5ae
