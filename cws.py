import foolbox
import numpy as np 
import torch
from torchvision.transforms import transforms
from FastAutoAugment.networks import get_model, num_class
from theconf import Config as C, ConfigArgumentParser
import torchvision
import sys
import os

def get_data(path = '/home/zhuzby/data'):
	_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
	transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
		])
	testset = torchvision.datasets.CIFAR100(root=path, train=False, download=True)
	testloader = torch.utils.data.DataLoader(
		testset, batch_size=64, shuffle=False, num_workers=32, pin_memory=True,
		drop_last=False
	)
	return testloader

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
	_ = C(sys.argv[-1])
	model = target_model(sys.argv[-2])
	preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
	fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=100)#, preprocessing=preprocessing)

	normal_correct = 0
	adv_correct = 0
	adv_imgs = []
	labels = []
	loader = get_data()
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
	np.save('cifar100_advs.npy', adv_imgs)
	np.save('cifar100_labels.npy', labels)

	# The `Adversarial` objects also provide a `distance` attribute. Note that the distances
	# can be 0 (misclassified without perturbation) and inf (attack failed).
	distances = np.asarray([a.distance.value for a in adversarials])
	print("{:.1e}, {:.1e}, {:.1e}".format(distances.min(), np.median(distances), distances.max()))
	print("{} of {} attacks failed".format(sum(adv.distance.value == np.inf for adv in adversarials), len(adversarials)))
	print("{} of {} inputs misclassified without perturbation".format(sum(adv.distance.value == 0 for adv in adversarials), len(adversarials)))