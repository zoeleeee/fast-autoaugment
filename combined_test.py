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
	dataloader = data.Dataloader(dataset, batch_size=64, shuffle=False, num_workers=32, pin_memory=True, drop_last=False)
	for images, label in loader:
		# images, labels = foolbox.utils.samples(dataset='cifar100', batchsize=64, data_format='channels_first', bounds=(0, 1))
		normal_correct += np.sum(fmodel.forward(images).argmax(axis=-1) == label)
		attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=foolbox.distances.MeanSquaredDistance)
		adversarials = attack(images, labels, unpack=False)
		adv_imgs += [a.perturbed for a in adversarials]
		adversarial_classes = np.asarray([a.adversarial_class for a in adversarials])
		adv_correct += np.mean(adversarial_classes == label)  # will always be 0.0
		labels += label
	print('normal acc:', normal_correct / len(dataloader.dataset))
	print('adversarial acc:', adv_correct / len(dataloader.dataset)) 
