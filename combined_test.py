# python combined_test.py nb_labels
from __future__ import division
import numpy as np 
import torch
from FastAutoAugment.networks import get_model, num_class
from FastAutoAugment.data import get_dataloaders
from theconf import Config as C, ConfigArgumentParser
import sys
import os
from torch.utils import data
import copy
from FastAutoAugment.metrics import accuracy

def label_permutation(labels, nb_labels, classifier_id):
    permutated_vec = np.load('{}_label_permutation_cifar100.npy'.format(nb_labels))[int(classifier_id)]
    tmp = copy.deepcopy(labels)
    for i in range(np.max(labels)+1):
    	labels[tmp==i] = permutated_vec[i]
    return labels

def target_model(save_path):
	model = get_model(C.get()['model'], num_class(C.get()['dataset'], 2))
	if save_path and os.path.exists(save_path):
		data = torch.load(save_path)
		if 'model' in data or 'state_dict' in data:
			key = 'model' if 'model' in data else 'state_dict'
			model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
		else:
			model.load_state_dict({k: v for k, v in data.items()})
		del data
	return model

def check_combined(imgs, label_path, nb_labels):
	res = []
	valids = np.ones(imgs.shape[0])
	model_dir = 'models'
	files = os.listdir(model_dir)
	entries = {int(file.split('_')[-5]): os.path.join(model_dir, file) for file in files}
	nb_files = 10#len(files)

	if not os.path.exists('res_{}.npy'.format(nb_files)):
		for i in np.arange(nb_files):
			labels = label_permutation(np.load(label_path), nb_labels, i)
			dataset = data.TensorDataset(torch.Tensor(imgs), torch.Tensor(labels))
			loader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=32, pin_memory=True, drop_last=False)
			preds = []
			valid = []
			path = entries[i]
			model = target_model(path)
			model.eval()
			for images, label in loader:
				outputs = model(images)
				_, predicted = torch.max(outputs, 1)

				_predicted = predicted.to('cpu').numpy()
				_label = label.to('cpu').numpy()
				if preds == []:
					preds = _predicted
					valid = (_predicted == _label)
				else:
					preds = np.hstack((preds, _predicted))
					valid = np.hstack((valid, (_predicted==_label)))
			valids = [valids[j] and valid[j] for j in range(len(valids))]
			res.append(preds)
			del loader
			del dataset
			del labels
		
		np.save('res_{}.npy'.format(nb_files), res)
		np.save('valid_{}.npy'.format(nb_files), valids)
	else:
		res = np.load('res_{}.npy'.format(nb_files))
		valids = np.load('valid_{}.npy'.format(nb_files))

	permutated_labels = np.load('{}_label_permutation_cifar100.npy'.format(nb_labels))[:nb_files].T
	res = np.array(res).T
	wr = {}
	labels = np.load(label_path)
	for i in np.arange(len(valids))[valids==0]:
		for j in np.arange(len(permutated_labels)):
			if list(res[i]) == list(permutated_labels[j]):
				wr[i] = j
				print('wrong#', i, labels[i], j)
	print('acc:', np.mean(valids))
	print('adversarial acc:', len(wr)/imgs.shape[0])

def check_origin(imgs, label_path, path='cifar100_pyramid272_top1_11.74.pth'):
	model = target_model(path)
	labels = np.load(label_path)
	dataset = data.TensorDataset(torch.Tensor(imgs), torch.Tensor(labels))
	loader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=32, pin_memory=True, drop_last=False)
	preds = []
	valid = []
	model = target_model(path)
	model.eval()
	for images, label in loader:
		outputs = model(images)
		_, predicted = torch.max(outputs, 1)

		_predicted = predicted.to('cpu').numpy()
		_label = label.to('cpu').numpy()
		if preds == []:
			preds = _predicted
			valid = (_predicted == _label)
		else:
			preds = np.hstack((preds, _predicted))
			valid = np.hstack((valid, (_predicted==_label)))
	print('acc:', np.mean(valid))

if __name__ == '__main__':
	idx = sys.argv[-2]
	label_path = 'cifar100_labels_{}.npy'.format(idx)
	imgs = np.load('cifar100_advs_{}.npy'.format(idx))
	nb_labels = sys.argv[-3]
	if sys.argv[-1] == 'origin':
		_ = C('confs/pyramid272_cifar100_2.yaml')
		check_origin(imgs, label_path)
	elif sys.argv[-1] == 'combined':
		_ = C('confs/pyramid272_cifar100_2_tl.yaml')
		check_combined(imgs, label_path, nb_labels)
