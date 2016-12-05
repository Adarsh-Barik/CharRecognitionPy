from os import listdir
import pandas
from generate_2d_plots import plot_all_graphs
import ast
class myclf():
	def __init__(self, cv_results_, cv_type):
		self.cv_type = cv_type
		self.cv_results_ = cv_results_

for file in listdir('../data/storage/'):
	if file.endswith(".csv"):
		f = '../data/storage/' + file
		cv_type =  file.split('_')[2].split('.')[0][:-1]
		# print (f)
		clfdf = pandas.DataFrame.from_csv(f)
		cv_results_ = pandas.DataFrame.to_dict(clfdf)
		for x, y in cv_results_.items():
			cv_results_[x] = list(y.values())

		a = [ast.literal_eval(i) for i in cv_results_['params']]
		cv_results_['params'] = a


		clf = myclf(cv_results_, cv_type)
		# print (clf.cv_results_)
		plot_all_graphs(clf, '../data/storage/figures/')
