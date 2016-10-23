"""
Python module to run k-means algorithm and get cluster centers for given array of points
Reference: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
author : adarsh
"""

from sklearn.cluster import KMeans


def get_cluster_centers(n, Xarray, iterations=300):
	"get cluster centers with given number of clusters and numpy array of points"
	kmeans = KMeans(n_clusters=n, max_iter=iterations, random_state=0).fit(Xarray)
	return kmeans.cluster_centers_

if __name__ == '__main__':
	import numpy as np 
	X = np.array([[1,2], [1,4], [1,0], [4,2], [4,4], [4,0]])
	a = get_cluster_centers(2, X)
	print a