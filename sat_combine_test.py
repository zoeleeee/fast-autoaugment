import numpy as np
import sys
from functools import reduce


def evaluation(nb_res):
	rep = np.load('2_label_permutation_cifar100.npy')[10:40]
	samples = np.load('res_48_{}.npy'.format(nb_res)).T[10:40]
	print(samples.shape)
	labels = np.load('cifar100_labels_{}.npy'.format(nb_res))
	orders = np.random.permutation(samples.shape[0])
	ex_labels = np.array([[rep[i][j] for j in labels] for i in orders])
	samples = np.array([samples[i] for i in orders])

	valid = (samples == ex_labels).T
	sep_valid = np.array([[np.max(val[j:j+3]) for j in np.arange(valid.shape[-1])[::3]] for val in valid])
	valid = np.sum(sep_valid, axis=-1)
	print(sep_valid.shape, valid.shape)
	print('acc:', np.sum(valid==10))

def predict_hamming(nb_res):
	return

def predict_setting(nb_res):
	rep = np.load('2_label_permutation_cifar100.npy')[10:40]
	samples = np.load('res_48_{}.npy'.format(nb_res)).T[10:40]
	print(samples.shape)
	labels = np.load('cifar100_labels_{}.npy'.format(nb_res))
	orders = np.random.permutation(samples.shape[0])
	samples = np.array([samples[i] for i in orders])
	a = (reduce(np.union1d, (rep[i][rep[i] == samples[i][j]], rep[i+1][rep[i+1]==samples[i+1][j]], rep[i+2][rep[i+1]==samples[i+1][j]])))
	# res = reduce(np.intersect1d, (reduce(np.union1d, a))
	
	

def predict_voting(nb_res):
	rep = np.load('2_label_permutation_cifar100.npy')[10:40]
	samples = np.load('res_48_{}.npy'.format(nb_res)).T[10:40]
	print(samples.shape, rep.shape)
	labels = np.load('cifar100_labels_{}.npy'.format(nb_res))
	orders = np.random.permutation(samples.shape[0])
	label_sets = [[rep[i][rep[i] == 0], rep[i][rep[i] == 1]] for i in np.arange(rep.shape[0])]
	print(np.array(label_sets).shape)
	res_sets = []
	sets = []
	def intersect(beg, j, s):
		print(beg, j)
		if len(s) == 0:
			return s
		if beg >= rep.shape[0]:
			sets.append(s)
			return
		for i in np.arange(3)+beg:
			intersect(beg+3, j, np.intersect1d(label_sets[orders[i]][samples[orders[i]][j]],s))

	for j in np.arange(samples.shape[1]):
		sets = []
		for i in np.arange(3):
			intersect(3, j, label_sets[orders[i]][samples[orders[i]][j]])
		print(sets)
		if sets == []:
			sets = [[]]
		res_sets.append(np.bincount(np.hstack(sets).astype(np.int64), minlength=100))
	np.save('res_sets_{}.npy'.format(nb_res), res_sets)
	res = np.argmax(np.array(res_sets), axis=-1)
	print('acc:', np.sum(res==labels))
	print(np.sum(res==labels)/len(labels))


if __name__ == '__main__':
	# evaluation(sys.argv[-1])
	predict_voting(sys.argv[-1])
