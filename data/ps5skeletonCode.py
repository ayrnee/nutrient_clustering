
def randIndex(truth, predicted):
	"""
	The function is to measure similarity between two label assignments
	truth: ground truth labels for the dataset (1 x 1496)
	predicted: predicted labels (1 x 1496)
	"""
	if len(truth) != len(predicted):
		print "different sizes of the label assignments"
		return -1
	elif (len(truth) == 1):
		return 1
	sizeLabel = len(truth)
	agree_same = 0
	disagree_same = 0
	count = 0
	for i in range(sizeLabel-1):
		for j in range(i+1,sizeLabel):
			if ((truth[i] == truth[j]) and (predicted[i] == predicted[j])):
				agree_same += 1
			elif ((truth[i] != truth[j]) and (predicted[i] != predicted[j])):
				disagree_same +=1
			count += 1
	return (agree_same+disagree_same)/float(count)

# Code Sample
import scipy.cluster.hierarchy as sch
import numpy as np
import pylab as pl

# Plot dendogram and cut the tree to find resulting clusters
fig = pl.figure()
data = np.array([[1,2,3],[1,1,1],[5,5,5]])
datalable = ['first','second','third']
hClsMat = sch.linkage(data, method='complete') # Complete clustering
sch.dendrogram(hClsMat, labels= datalable, leaf_rotation = 45)
fig.savefig("thing.pdf")
resultingClusters = sch.fcluster(hClsMat,t= 3, criterion = 'distance')
print resultingClusters


# Your code starts from here ....

# 1.
# Scaling min max
# STUDENT CODE TODO

# 2.
# K-means http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# STUDENT CODE TODO

# 3.
# Compute Rand Index
# STUDENT CODE TODO

# 4.
# Examining K-mean objective
# STUDENT CODE TODO

# 5.
# Dendogram plot
# Dendogram - http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# Linkage - http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.cluster.hierarchy.linkage.html
# STUDENT CODE TODO

# 6.
# Hierarchical clustering
# SciPy's Cluster - http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster
# STUDENT CODE TODO

# 7.
# K-means for Sub-cluster
# STUDENT CODE TODO
