from __future__ import print_function
import torch
import imp
import numpy as np
import os
import sys
import foolbox
from foolbox.v1.attacks import ProjectedGradientDescentAttack, CarliniWagnerL2Attack
from foolbox.distances import *
from foolbox.criteria import TargetClass
import copy
from keras.datasets import mnist
from mnist import get_data
from keras import backend as K
from numpy import linalg as LA
import torch.nn.functional as F


def epoch_attack(img, idxs):
	return

def target_model(save_path, nb_labels = 2):
	model = get_model(C.get()['model'], num_class(C.get()['dataset'], nb_labels))
	if save_path and os.path.exists(save_path):
		data = torch.load(save_path)
		if 'model' in data or 'state_dict' in data:
			key = 'model' if 'model' in data else 'state_dict'
			model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
		else:
			model.load_state_dict({k: v for k, v in data.items()})
		del data
	return model

def predict(img, idxs, t=1):
	preds = []
	scores = []
	labels = np.load('2_label_permutation.npy')[idxs].T
	for i in idxs:
		model = torch.load('mnist2/mnist2_{}.pth'.format(i)).eval()
		output = F.softmax(model(torch.from_numpy(img.reshape(-1, 1, 28, 28))), dim=-1)
		score, i = torch.max(output, 1)
		preds.append(i.cpu().numpy()[0])
		scores.append(score.detach().cpu().numpy())
		del model
	_pred = np.repeat(np.array(preds).reshape(1, -1), labels.shape[0], axis=0)
	dists = np.sum(np.absolute(_pred - labels), axis=1)
	if np.min(dists) > t:
		return None, None, preds
	pred_labels = np.arange(len(dists))[dists==np.min(dists)]
	pred_scores = [np.sum([scores[i] if preds[i]==labels[v][i] else 1-scores[i] for i in np.arange(len(preds))]) for v in pred_labels]
	pred_label = pred_labels[np.argmax(pred_scores)]
	return pred_label, labels[pred_label], np.array(preds).reshape(-1)


def find_closest(preds, idxs, label):
	labels = np.load('2_label_permutation.npy')[idxs].T
	dists = [np.sum(l-preds) if np.sum(l-label) > 0 else 1e10 for l in labels]
	res = np.argmin(dists)
	return res, labels[res]


def loop_attack(img, label, idxs, org, distance='l_inf', threshold=10000, file_name=''):
	preds = copy.deepcopy(label)
	res, aim = find_closest(preds, idxs, label)
	cnt = 0
	adv = copy.deepcopy(img)
	print(adv.shape)
	t = 1
	pred_label = org

	model_dir = 'models'
	files = os.listdir(model_dir)
	entries = {int(file.split('_')[-5]) if len(file.split('_')) == 9 else -1 : os.path.join(model_dir, file)  for file in files}
	if -1 in entries.keys():
		del entries[-1]
	
	change_classifier = np.zeros(len(preds)).astype(np.bool)
	# MainModel = imp.load_source('MainModel', 'mnist2/mnist2_450.py')
	while(cnt < threshold):
		if pred_label is not None:
			if pred_label != org:
				break
		cnt += 1
		res, aim  = find_closest(preds, idxs, label)
		print('{}# aim:{}_{}'.format(cnt, res, np.sum(np.absolute(aim-preds))))
		tmp = preds != aim
		# print(change_classifier)
		# print(tmp)
		print('{}#classifier:{}, \nadded classifier:{}'.format(cnt, np.arange(len(preds))[tmp], np.arange(len(preds))[np.array([change_classifier[i]^tmp[i] if change_classifier[i]==False else False for i in np.arange(len(preds))])]))
		change_classifier = tmp
		for i in np.arange(len(idxs)):
			if preds[i] == aim[i]:
				continue
			model = target_model(entries[idxs[i]], nb_labels=nb_labels).eval()		
			fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=2)
			if distance == 'l_inf':
				attack = ProjectedGradientDescentAttack(fmodel, distance=Linfinity)
				order = np.inf
			elif distance == 'l_2':
				attack = CarliniWagnerL2Attack(fmodel, distance=MeanSquaredDistance)
				order = 2
			adv = attack(adv, preds[i])#np.array(preds[i]).astype(np.int))
			if type(adv) == type(None):
				break
			del model
		if type(adv) == type(None):
			break
		else:
			np.save('whites/1_analysis/{}_{}_{}_{}.npy'.format(file_name.split('/')[-1][:-4], cnt, res, org), adv)
		pred_label, pred_rep, preds = predict(adv, idxs)
	if type(adv)!= type(None) and cnt < threshold:
		np.save('whites/{}_{}/adv_{}_{}_{}_{}.npy'.format(t, distance, cnt, pred_label, org, LA.norm((img-adv).reshape(-1), order)), [adv, img])
	print(cnt, (LA.norm((img-adv).reshape(-1), order) if type(adv) != type(None) else 'None'))


def main():
	x_test = np.load('cifar100_advs_10000.npy')
	img_rows, img_cols, nb_channels = 32, 32, 3
	if K.image_data_format() != 'channels_first':
	    x_test = x_test.reshape(x_test.shape[0], nb_channels, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, nb_channels)
	    input_shape = (img_rows, img_cols, 1)
	y_test = np.load('cifar100_labels_10000.npy')
	# (x_train, x_test, y_train, y_test), _, _ = get_data()
	# print('max:', np.max(x_test))
	idxs = np.arange(30)
	labels = np.load('2_label_permutation_cifar100.npy')[idxs].T
	print(labels.shape)
	nb = 0
	for img, i in zip(x_test, y_test):
		print(nb, labels[i])
		nb += 1
		loop_attack(img, labels[i], idxs, i, distance='l_inf')

def analysis():
	file_name=sys.argv[-1]
	img = np.load(file_name)[1]
	label = int(file_name.split('_')[-2])

	idxs = np.arange(500)
	rep = np.load('2_label_permutation.npy')[idxs].T
	loop_attack(img, rep[label], idxs, label, file_name=file_name)

if __name__ == '__main__':
	main()
	# analysis()
