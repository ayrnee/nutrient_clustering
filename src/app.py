#!/usr/bin/env python

from collections import Counter
from collections import defaultdict
import os
import numpy as np
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import scipy.cluster.hierarchy as sch


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

def read_file(f_name, subdir = 'dat/'):
    samples = []
    names = []
    with open('./data/' + subdir + f_name) as file:
        for line in file:
            raw = line.strip().split('^')[:-1]
            names.append(raw[:1])
            samples.append(map(float, raw[1:]))
    return names, samples

def read_data(dat_dir):
    combined_data = [[], [], []]
    seperated_data = []
    seperated_names = []
    f_names = []
    label = 0
    for file in os.listdir(dat_dir):
        f_names.append(file)
        temp_lists = read_file(file)
        combined_data[0] += temp_lists[0]
        combined_data[1] += temp_lists[1]
        combined_data[2] += [label]*len(temp_lists[0])
        seperated_data.append(np.array(temp_lists[1]))
        seperated_names.append(temp_lists[0])
        label += 1
    data = [np.array(combined_data[0]), np.array(combined_data[1]), np.array(combined_data[2])]
    return data, seperated_data, f_names, seperated_names

def normalize_vals(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data[1])

def k_means(data):
    kms = KMeans(n_clusters = 4, n_jobs =  -1)
    return kms.fit_predict(data[1])

def k_means_centroids(data):
    kms = KMeans(
        n_clusters = 4,
        init = 'random',
        n_init = 1,
        n_jobs =  -1
    )
    kms.fit(data)
    return kms.inertia_, kms.labels_

def cluster_test(data, f_names):
    start = 0
    samples = []

    for index in range(len(data)):
        idx = np.random.randint(len(data[index]), size=30)
        subset = data[index][idx]
        samples.extend(subset)

    datalabels = []
    for i in range(len(f_names)):
        datalabels.extend([i]*30)

    fig = pl.figure()
    hClsMat = sch.linkage(np.array(samples), method='complete') # Complete clustering
    sch.dendrogram(hClsMat, labels= datalabels, leaf_rotation = 45)
    fig.savefig("things.pdf")

def sep_clusters(samples, labels):
    hCLsMat = sch.linkage(np.array(samples), method = 'complete')
    clusters = sch.fcluster(hCLsMat, t = 3.8, criterion = 'distance')

    results = []
    t_vals = []
    for i in [2.05]:
        clusters = sch.fcluster(hCLsMat, t = i, criterion = 'distance')

        results.append(randIndex(clusters, labels))
        t_vals.append(i)
        print len(set(clusters))
    max = np.argmax(results)
    print "maximal value was ", results[max], "with cluster of ", t_vals[max]

def rand_display(clusters, samples, names):
    for key in clusters.keys():
        if len(clusters[key]) < 10:
            for val in clusters[key]:
                #print names[val]
                pass
        else:
            rng = np.random.random_integers(low = 0, high = len(clusters[key]) - 1, size = 10)
            for idx in rng:
                #print names[clusters[key][idx]]
                pass
        # print "***********************************************"

def analyze_distribution(clusters, names):
    max = -1
    idx = -1
    for key in clusters:
        if len(clusters[key]) > max:
            max = len(clusters[key])
            idx = key
    attributes = []
    for sample in clusters[idx]:
        attributes += names[sample][0].strip().split(',')
    most_common = Counter(attributes).most_common(2)
    print "max cluster size", max
    print most_common



def sub_clusters(data, names):
    steps = [5,10,25,50,75]

    for step in steps:
        clusters = defaultdict(list)
        kms = KMeans(n_clusters = step, n_jobs =  -1)
        kms_labels = kms.fit_predict(data)
        for index in range(len(kms_labels)):
            clusters[kms_labels[index]].append(index)


        rand_display(clusters, data, names)

        analyze_distribution(clusters, names)






def main():
    data, sep_data, f_names, sep_names= read_data('./data/dat')
    # print data[1].shape
    data[1] = normalize_vals(data)
    # for sample in data:
    #     print sample
    kms_labels =  k_means(data)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(data[1]))
    rand_data, rand_labels = data[1][perm], data[2][perm]
    # rand_score = randIndex(data[2], rand_labels)
    # print rand_score
    unique_scores = set()
    # for i in range(20):
    #     kms_inertia, kms_labels = k_means_centroids(data[1])
    #     rand_score = randIndex(kms_labels, rand_labels)
    #     unique_scores.add(kms_inertia)
    #     print kms_inertia


    # cluster_test(sep_data, f_names)
    # sep_clusters(data[1],data[2])
    sub_clusters(sep_data[1], sep_names[1])






if __name__ == '__main__':
    main()
