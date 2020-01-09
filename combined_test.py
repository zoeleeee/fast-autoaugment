import numpy as np 
import torch
from FastAutoAugment.networks import get_model, num_class
from FastAutoAugment.data import get_dataloaders
from theconf import Config as C, ConfigArgumentParser
import sys
import os
from torch.utils import data

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
	model = target_model(sys.argv[-1])
	imgs = np.load('cifar100_advs.npy')
	labels = np.load('cifar100_labels.npy')
	dataset = data.TensorDataset(torch.Tensor(imgs), torch.Tensor(labels))
	dataloader = data.Dataloader(dataset)
	