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
from cws import get_data

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

def check_combined(imgs, label_path, nb_labels, idx):
	res = []
	scores = []
	valids = np.ones(imgs.shape[0])
	model_dir = 'models'
	files = os.listdir(model_dir)
	entries = {int(file.split('_')[-5]) if len(file.split('_')) == 9 else -1 : os.path.join(model_dir, file)  for file in files}
	if -1 in entries.keys():
		del entries[-1]
	print(entries)
	nb_files = len(entries)
	# order = np.random.permutation(nb_files)+10
	if not os.path.exists('res_{}_{}.npy'.format(nb_files,idx)):
		for i in np.arange(nb_files):
			labels = label_permutation(np.load(label_path), nb_labels, i)
			# print(labels.shape)
			dataset = data.TensorDataset(torch.Tensor(imgs), torch.Tensor(labels))
			loader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=32, pin_memory=True, drop_last=False)
			preds = []
			valid = []
			score = []
			path = entries[i]
			model = target_model(path)
			model.eval()
			for images, label in loader:
				# print(images.size(), label.size())
				outputs = model(images)
				sc, predicted = torch.max(outputs, 1)

				_predicted = predicted.to('cpu').numpy()
				_label = label.to('cpu').numpy()
				_score = sc.to('cpu').numpy()
				if len(preds) == 0:
					preds = _predicted
					valid = (_predicted == _label)
					score = _score
				else:
					preds = np.hstack((preds, _predicted))
					valid = np.hstack((valid, (_predicted==_label)))
					score = np.hstack((score, _score))
			valids = [valids[j] and valid[j] for j in range(len(valids))]
			res.append(preds)
			scores.append(score)
			del loader
			del dataset
			del labels
		
		res = np.array(res).T
		valids = np.array(valids)
		np.save('res_{}_{}.npy'.format(nb_files, idx), res)
		np.save('valid_{}_{}.npy'.format(nb_files, idx), valids)
		np.save('score_{}_{}.npy'.format(nb_files, idx), scores)
	else:
		res = np.load('res_{}_{}.npy'.format(nb_files, idx))
		valids = np.load('valid_{}_{}.npy'.format(nb_files, idx))
		scores = np.load('score_{}_{}.npy'.format(nb_files, idx))

	permutated_labels = np.load('{}_label_permutation_cifar100.npy'.format(nb_labels))[:nb_files].T
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

def get_normal_data():
	loader = get_data()
	imgs = []
	labels = []
	for images, label in loader:
		images, label = images.numpy(), label.numpy()
		print(images.shape, label.shape)
		if len(imgs) == 0:
			imgs = images
			labels = label
		else:
			imgs = np.vstack((imgs, images))
			labels = np.hstack((labels, label))

	np.save('cifar100_advs_10000.npy', imgs)
	np.save('cifar100_labels_10000.npy', labels)

if __name__ == '__main__':
	# get_normal_data()
	idx = sys.argv[-2]
	label_path = 'cifar100_labels_{}.npy'.format(idx)
	imgs = np.load('cifar100_advs_{}.npy'.format(idx))
	nb_labels = sys.argv[-3]
	if sys.argv[-1] == 'origin':
		_ = C('confs/pyramid272_cifar100_2.yaml')
		check_origin(imgs, label_path)
	elif sys.argv[-1] == 'combined':
		_ = C('confs/pyramid272_cifar100_2_tl.yaml')
		check_combined(imgs, label_path, nb_labels, idx)
