from __future__ import division
import numpy as np 
import sys
import os
import copy
from cws import get_data
from sklearn.metrics import pairwise
from matplotlib import pyplot as plt

def l_1(v1, v2):
	v1 = np.reshape(v1.shape[0], -1)
	v2 = np.reshape(v2.shape[0], -1)
	return pairwise.paired_manhattan_distances(v1, v2)

def l_2(v1, v2):
	v1 = np.reshape(v1.shape[0], -1)
	v2 = np.reshape(v2.shape[0], -1)
	return pairwise.paired_euclidean_distances(v1, v2)

def l_inf(v1, v2):
	v1 = np.reshape(v1.shape[0], -1)
	v2 = np.reshape(v2.shape[0], -1)
	return np.max(np.absolute(v1-v2), axis=-1)

def hamming_dist(v1, v2):
	return np.sum(np.absolute(v1-v2), axis=-1)

def label_dist(normal, advs):
	labels = np.argmax(normal, axis=-1)
	return np.absolute([v1[i][labels[i]]-v2[i][labels[i]] for i in np.arange(len(labels))])

def target_dist(normal, advs):
	labels = np.argmax(advs, axis=-1)
	return np.absolute([advs[i][labels[i]]-normal[i][labels[i]] for i in np.arange(len(labels))])

def main(dist, advs_path='cifar100_advs_500.npy'):
	labels = np.load('cifar100_labels_10000.npy')
	normal_softmax = np.load('softmax_normal_cifar100.npy')
	valids = np.hstack(np.argmax(normal_softmax, axis=-1)) == labels
	samples = np.load('cifar100_advs_10000.npy')[valids]
	normal_softmax = normal_softmax[valids]
	advs = np.load(advs_path)[valids]
	if dist == 'l1':
		distortion = l_1(samples, advs)
	elif dist == 'l2':
		distortion = l_2(samples, advs)
		advs_softmax = np.load('softmax_cws_advs_cifar100.npy')[valids]
	elif dist == 'linf':
		distortion = l_inf(samples, advs)

	dist1 = hamming_dist(normal_softmax, advs_softmax)
	dist2 = label_dist(normal_softmax, advs_softmax)
	dist3 = target_dist(normal_softmax, advs_softmax)

	plt.subplot(3, 1, 1)
	plt.scatter(distortion, dist1, color='plum')
	plt.title('hamming distance')
	plt.subplot(3, 2, 1)
	plt.scatter(distortion, dist2, color='gold')
	plt.title('label distance')
	plt.subplot(3, 3, 1)
	plt.scatter(distortion, dist3, color='cornflowerblue')
	plt.title('target distance')
	plt.savefig('distortion.png')	

if __name__ == '__main__':
	main(sys.argv[-1])