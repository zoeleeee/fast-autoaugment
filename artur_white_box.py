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
from keras import backend as K
from numpy import linalg as LA
import torch.nn.functional as F
from theconf import Config as C, ConfigArgumentParser
from FastAutoAugment.networks import get_model, num_class
import torch.nn as nn

class NET(nn.Module):
	def __init__(self, model, output, dim):
		super(NET, self).__init__()
		self.model = model
		self.fc = torch.zeros(output, 2)
		self.fc[dim][0], self.fc[dim][1] = -1, 1
		self.bias = torch.Tensor([[1, 0]])

	def forward(self, x):
		x = self.model(x)
		x = torch.sigmoid(x)
		x = torch.matmul(x, self.fc) + self.bias
		return x


def target_model(save_path, nb_labels = 30):
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

def predict(img, idxs, model, t=1):
	model.eval()
	labels = np.load('2_label_permutation_cifar100.npy')[idxs].T
	outputs = model(torch.Tensor(img.reshape(-1, img.shape[0], img.shape[1], img.shape[2])))
	scores = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1)
	preds = np.array([1 if u >= 0.5 else 0 for u in scores])
	_pred = np.repeat(np.array(preds).reshape(1, -1), labels.shape[0], axis=0)
	dists = np.sum(np.absolute(_pred - labels), axis=1)
	if np.min(dists) > t:
		return None, None, preds
	pred_labels = np.arange(len(dists))[dists==np.min(dists)]
	pred_scores = [np.sum([scores[i] if preds[i]==labels[v][i] else 1-scores[i] for i in np.arange(len(preds))]) for v in pred_labels]
	pred_label = pred_labels[np.argmax(pred_scores)]
	return pred_label, labels[pred_label], np.array(preds).reshape(-1)

def find_closest(preds, idxs, label):
	labels = np.load('2_label_permutation_cifar100.npy')[idxs].T
	dists = [np.sum(l-preds) if np.sum(l-label) > 0 else 1e10 for l in labels]
	res = np.argmin(dists)
	return res, labels[res]

def loop_attack(img, label, idxs, org, distance='l_inf', threshold=10000, file_name='cifar100_pyramid272_30outputs_500epochs.pth'):
	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	preprocessing = dict(mean=[0,0,0], std=[1,1,1], axis=-3)
	model = target_model(file_name)
	pred_label, pred_rep, preds = predict(img, idxs, model)

	if np.sum(preds.reshape(-1)-label) != 0:
		return
	res, aim = find_closest(preds, idxs, label)
	cnt = 0
	adv = copy.deepcopy(img)
	print(adv.shape)
	t = 1
	pred_label = org

	change_classifier = np.zeros(len(preds)).astype(np.bool)

	while(cnt < threshold):
		if pred_label is not None:
			if pred_label != org:
				break
		cnt += 1
		res, aim  = find_closest(preds, idxs, label)
		print('{}# aim:{}_{}'.format(cnt, res, np.sum(np.absolute(aim-preds))))
		tmp = preds != aim
		print('{}#classifier:{}, \nadded classifier:{}'.format(cnt, np.arange(len(preds))[tmp], np.arange(len(preds))[np.array([change_classifier[i]^tmp[i] if change_classifier[i]==False else False for i in np.arange(len(preds))])]))
		change_classifier = tmp
		for i in np.arange(len(idxs)):
			print(cnt,i)
			if preds[i] == aim[i]:
				continue
			if preds[i] == 0:
				continue

			net = NET(model, len(idxs), i).eval()
			fmodel = foolbox.models.PyTorchModel(net, bounds=(-3, 3), num_classes=2, preprocessing=preprocessing)
			if distance == 'l_inf':
				attack = ProjectedGradientDescentAttack(fmodel, distance=Linfinity)
				order = np.inf
			elif distance == 'l_2':
				attack = CarliniWagnerL2Attack(fmodel, distance=MeanSquaredDistance)
				order = 2		
			adv = attack(adv, preds[i])
			if type(adv) == type(None):
				break
		if type(adv) == type(None):
			break
		else:
			np.save('whites/artur_1_analysis/{}_{}_{}_{}.npy'.format(file_name.split('/')[-1][:-4], cnt, res, org), adv)
		pred_label, pred_rep, preds = predict(adv, idxs, model)
	if type(adv)!= type(None) and cnt < threshold:
		np.save('whites/artur_{}_{}/adv_{}_{}_{}_{}.npy'.format(t, distance, cnt, pred_label, org, LA.norm((img-adv).reshape(-1), order)), [adv, img])
	print(cnt, (LA.norm((img-adv).reshape(-1), order) if type(adv) != type(None) else 'None'))



def main():
	_ = C('confs/pyramid272_cifar100_2.yaml')
	x_test = np.load('cifar100_advs_10000.npy')
	img_rows, img_cols, nb_channels = 32, 32, 3
	if K.image_data_format() != 'channels_first':
	    x_test = x_test.reshape(x_test.shape[0], nb_channels, img_rows, img_cols)
	    input_shape = (nb_channels, img_rows, img_cols)
	else:
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, nb_channels)
	    input_shape = (img_rows, img_cols, nb_channels)
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

if __name__ == '__main__':
	main()