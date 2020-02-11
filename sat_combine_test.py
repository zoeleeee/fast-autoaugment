import numpy as np
import sys
from functools import reduce
import time
import os

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

def std(nb_res):
	rep = np.load('2_label_permutation_cifar100.npy')[10:40].T
	samples = np.load('res_48_{}.npy'.format(nb_res)).T[10:40].T
	print(samples.shape, rep.shape)
	labels = np.load('cifar100_labels_{}.npy'.format(nb_res))
	res = [np.arange(rep.shape[0])[np.sum(np.absolute((np.repeat(samples[j].reshape(-1,len(samples[j])), rep.shape[0], axis=0) - rep)), axis=-1) == 0] for j in np.arange(samples.shape[0])]
	# print(res)
	# preds = [np.argmin([0 if np.sum(rep[i]-samples[j]) == 0 else 1 for i in np.arange(rep.shape[0])]) for j in np.arange(samples.shape[0])]
	_res = np.hstack([v if len(v) == 1 else np.array([-1]) for v in res])
	print('acc:', np.mean(_res == labels))
	print('invalid:', np.mean(_res == -1))
	print('wrong:', 1-np.mean(_res == labels)-np.mean(_res == -1))
	
def non_labels_analysis():
	rep = np.load('2_label_permutation_cifar100.npy')
	label_sets = [[np.arange(rep.shape[-1])[rep[i] == 0], np.arange(rep.shape[-1])[rep[i] == 1]] for i in np.arange(rep.shape[0])]
	# print(label_sets[0][0])
	# print(label_sets[0][1])
	for t in np.arange(10, 551):
		flag = True
		for i in np.arange(rep.shape[1]):
			sets = []
			# _tmp = label_sets[0][rep[0][i]]
			for j in np.arange(t):
				# print(j, label_sets[j][rep[j][i]])
				# _tmp = np.intersect1d(_tmp, label_sets[j][rep[j][i]])
				sets.append(label_sets[j][rep[j][i]])
			# print(len(sets))
			tmp = reduce(np.intersect1d, sets)
			# print(tmp)
			# print(_tmp)
			if len(tmp) > 1:
				flag = False
				break
		if flag:
			print(t)
			break


def predict_hamming(nb_res, t):
	_beg = 0
	_end = 30
	idxs = np.hstack((np.arange(10), np.arange(20,30))).reshape(-1)
	labels = np.load('cifar100_labels_{}.npy'.format(10000))
	if not os.path.exists('_models/hamming_labels_{}.npy'.format(nb_res)):
		rep = np.load('2_label_permutation_cifar100.npy')[idxs].T
		samples = np.load('_res_30_{}.npy'.format(nb_res)).T[idxs].T
		print(samples.shape)
		
		# orders = np.random.permutation(samples.shape[1])
		# _rep = np.array([rep[j] for j in len(rep)]).T
		scores = np.load('_score_30_{}.npy'.format(nb_res))[idxs].T #[10000, 30]
		print(scores.shape)
		preds = []
		preds_dist = []
		preds_score = []
		for i in np.arange(samples.shape[0]):
			_pred = np.repeat(samples[i].reshape((-1, len(samples[i]))), rep.shape[0], axis=0)
			dists = np.sum(np.absolute(_pred - rep), axis=1)
			max_dist = np.min(dists)
			pred_labels = np.arange(len(dists))[dists==max_dist]
			pred_scores = [np.sum([scores[i][k] if samples[i][k] == rep[j][k] else 1-scores[i][k] for k in np.arange(len(scores[i]))]) for j in pred_labels]
			pred_label = pred_labels[np.argmax(pred_scores)]
			preds.append(pred_label)
			preds_dist.append(dists[pred_label])
			preds_score.append(np.max(pred_scores))
		np.save('_models/hamming_labels_{}.npy'.format(nb_res), preds)
		np.save('_models/hamming_labels_dists_{}.npy'.format(nb_res), preds_dist)
		np.save('_models/hamming_labels_score_{}.npy'.format(nb_res), preds_score)
		preds = np.array(preds)
		preds_dist = np.array(preds_dist)
	else:
		preds = np.load('_models/hamming_labels_{}.npy'.format(nb_res))
		preds_dist = np.load('_models/hamming_labels_dists_{}.npy'.format(nb_res))

	print('avg Hamming distance:{}, max:{}, min:{}, med:{}'.format(np.mean(preds_dist), np.max(preds_dist), np.min(preds_dist), np.median(preds_dist)))
	# t = 1
	
	print(t, 'acc:', np.sum(preds_dist[preds == labels] < t) / len(labels))
	print(t, 'acc:', np.sum(preds_dist[preds != labels] < t) / len(labels))


	

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
	# predict_voting(sys.argv[-1])
	# std(sys.argv[-1])
	predict_hamming(sys.argv[-2],int(sys.argv[-1]))
	# non_labels_analysis()