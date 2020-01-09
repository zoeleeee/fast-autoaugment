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

def label_permutation(labels, nb_labels):
    permutated_vec = np.load('{}_label_permutation_cifar100.npy'.format(nb_labels))[int(args.classifier_id)]
    tmp = copy.deepcopy(labels)
    for i in range(np.max(labels)+1):
    	labels[tmp==i] = permutated_vec[i]
    return labels

def target_model(save_path):
	model = get_model(C.get()['model'], num_class(C.get()['dataset']))
	if save_path and os.path.exists(save_path):
		data = torch.load(save_path)
		if 'model' in data or 'state_dict' in data:
			key = 'model' if 'model' in data else 'state_dict'
			model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
		else:
			model.load_state_dict({k: v for k, v in data.items()})
		del data
	return model

if __name__ == '__main__':	
	imgs = np.load('cifar100_advs.npy')
	labels = permutate_vec(np.load('cifar100_labels.npy'), sys.argv[-1])
	dataset = data.TensorDataset(torch.Tensor(imgs), torch.Tensor(labels))
	dataloader = data.Dataloader(dataset, batch_size=64, shuffle=False, num_workers=32, pin_memory=True, drop_last=False)

	res = []
	valids = np.ones(len(dataset))
	model_dir = 'cifar100_2_models'
	files = os.listdir(model_dir)
	entries = {int(file.split('_')[-5]): os.path.join(model_dir, file) for file in files}
	for i in len(files):
		preds = []
		valid = []
		path = entries[i]
		model = target_model(path)
		for images, label in loader:
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)

			if preds == []:
				preds = predicted
				valid = (predicted == label)
			else:
				preds = np.hstack((preds, predicted))
				valid = np.hstack((valid, predicted==label))
		valids = [valids[i] and valid[i] for i in range(len(valids))]
		res.append(preds)
	
	permutated_labels = np.load('{}_label_permutation_cifar100.npy'.format(sys.argv[-1]))[:len(files)].T
	wr = []
	for i in np.arange(len(valids))[valids==0]:
		if (True if res.T[i] == permuted_label else False for permuted_label in permutated_labels):
			wr.append(i)
	print('acc:', np.mean(valids))
	print('wrong valid:', wr/len(valids))

	print('normal acc:', normal_correct / len(dataloader.dataset))
	print('adversarial acc:', adv_correct / len(dataloader.dataset)) 
