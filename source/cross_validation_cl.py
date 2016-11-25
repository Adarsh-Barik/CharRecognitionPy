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
		fit.fit(X, y, self.estimator, self.cv_results_, self.best_score_, self.best_params_, self.param_grid, self.cv_n, self.cv_type)
