import numpy as np
import sys
from functools import reduce
import time


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
	rep = np.load('2_label_permutation_cifar100.npy')[10:40].T
	samples = np.load('res_48_{}.npy'.format(nb_res)).T[10:40].T
	print(samples.shape)
	labels = np.load('cifar100_labels_{}.npy'.format(nb_res))
	# orders = np.random.permutation(samples.shape[1])
	# _rep = np.array([rep[j] for j in len(rep)]).T
	scores = np.load('_score_48_{}'.format(nb_res)).T[10:40].T #[10000, 30]
	print(score.shape)
	preds = []
	preds_dist = []
	preds_score = []
	for i in np.arange(samples.shape[0]):
		_pred = np.repeat(samples[i].reshape((-1, len(samples[i]))), _rep.shape[0], axis=0)
		dists = np.sum(np.absolute(_pred - rep), axis=1)
		max_dist = np.max(dists)
		pred_labels = np.arange(len(dists))[dists==max_dist]
		pred_scores = [np.sum(scores[i][samples[i] == rep[j]]) for j in pred_labels]
		pred_label = pred_labels[np.argmax(pred_scores)]
		preds.append(pred_label)
		preds_dist.append(dists[pred_label])
		preds_score.append(pred_scores[pred_label])
	np.save('hamming_labels.npy', preds)
	np.save('hamming_labels_dists.npy', preds_dist)
	np.save('hamming_labels_score.npy', preds_score)

	print('avg Hamming distance:', np.mean(preds_dist))
	print('acc:', np.mean(np.array(preds) == labels))


	

def predict_setting(nb_res):
	rep = np.load('2_label_permutation_cifar100.npy')[10:40]
	samples = np.load('res_48_{}.npy'.format(nb_res)).T[10:40]
	print(samples.shape)
	labels = np.load('cifar100_labels_{}.npy'.format(nb_res))
	orders = np.random.permutation(samples.shape[0])
	label_sets = [[np.arange(rep.shape[-1])[rep[i] == 0], np.arange(rep.shape[-1])[rep[i] == 1]] for i in np.arange(rep.shape[0])]
	# samples = np.array([samples[i] for i in orders])
	pred = []
	invalid = []
	for j in np.arange(samples.shape[-1]):
		sets = []
		for i in np.arange(samples.shape[0])[::3]:
			a = reduce(np.union1d, [label_sets[orders[i]][samples[orders[i]][j]], label_sets[orders[i+1]][samples[orders[i+1]][j]], label_sets[orders[i+2]][samples[orders[i+2]][j]]])
			sets.append(a)
		if len(sets) == 0:
			invalid.append(j)
			sets = [[]]
		pred.append(reduce(np.intersect1d, sets))
	print(pred)
	

def predict_voting(nb_res):
	begin_time = time.time()
	rep = np.load('2_label_permutation_cifar100.npy')[10:40]
	samples = np.load('res_48_{}.npy'.format(nb_res)).T[10:40]
	print(samples.shape, rep.shape)
	labels = np.load('cifar100_labels_{}.npy'.format(nb_res))
	orders = np.random.permutation(samples.shape[0])
	label_sets = [[np.arange(rep.shape[-1])[rep[i] == 0], np.arange(rep.shape[-1])[rep[i] == 1]] for i in np.arange(rep.shape[0])]
	print(np.array(label_sets).shape)
	res_sets = []
	sets = []
	invalids = []
	def intersect(beg, j, s):
		# print(beg, j)
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
		# print(sets)
		if np.hstack(sets).shape[0] == 0:
			invalids.append(j)
		# if len(sets) == 0:
		# 	invalids.append(j)
		# 	sets = [[]]
		res_sets.append(np.bincount(np.hstack(sets).astype(np.int64), minlength=100))
	print('time cost:', time.time()-begin_time)
	np.save('res_sets_{}.npy'.format(nb_res), res_sets)	
	res = np.argmax(np.array(res_sets), axis=-1)
	res[np.array(invalids).astype(np.int)] = -1
	np.save('pred_label_{}.npy'.format(nb_res), res)
	np.save('invalid_label_{}.npy'.format(nb_res), invalids)
	print('acc:', np.sum(res==labels), np.sum(res==labels)/len(labels))
	print('invalid:', len(invalids), len(invalids)/len(labels))
	for i in np.arange(len(res))[res!=labels]:
		if res[i] != -1:
			print('wr#{}: {} {}_{}, right:{}_{}'.format(i, res_sets[i], np.argmax(res_sets[i]), res_sets[i][np.argmax(res_sets[i])], labels[i], res_sets[i][labels[i]]))



if __name__ == '__main__':
	# evaluation(sys.argv[-1])
	predict_voting(sys.argv[-1])
	predict_hamming(sys.argv[-1])
