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
	testset = torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(
		testset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, drop_last=False
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

def pgd(fmodel, loader):
	adv_imgs, labels, distances, adv_classes = [], [], [], []
	adv_correct = 0
	attack = foolbox.attacks.ProjectedGradientDescentAttack(fmodel, distance=foolbox.distances.Linfinity)
	for idx, (images, labels) in enumerate(loader):
		print(idx)
		if idx < 439:
			continue
		images, labels = images.numpy(), labels.numpy()
		print(images.shape)
		adversarials = attack(images, labels, unpack=False)
		adv_imgs += [a.perturbed for a in adversarials]
		distances += [a.distance.value for a in adversarials]
		adversarial_classes = [a.adversarial_class for a in adversarials]
		adv_correct += np.mean(adversarial_classes == labels)  # will always be 0.0
		np.save('pgd_advs/cifar100_pgd_advs_{}.npy'.format(idx), adv_imgs)
		np.save('pgd_advs/cifar100_pgd_dist_{}.npy'.format(idx), distances)
	print('adversarial acc:', adv_correct / len(loader.dataset))


def ead(fmodel, loader):
	adv_imgs, labels, distances, adv_classes = [], [], [], []
	attack = foolbox.attacks.EADAttack(fmodel, distance=foolbox.distances.MeanAbsoluteDistance)
	adv_correct = 0
	for idx, (images, labels) in enumerate(loader):
		print(idx)
		images, labels = images.numpy(), labels.numpy()
		adversarials = attack(images, labels, unpack=False)
		adv_imgs += [a.perturbed for a in adversarials]
		distances += [a.distance.value for a in adversarials]
		adversarial_classes = [a.adversarial_class for a in adversarials]
		adv_correct += np.mean(adversarial_classes == labels)  # will always be 0.0
		np.save('ead_advs/cifar100_ead_advs_{}.npy'.format(idx), adv_imgs)
		np.save('ead_advs/cifar100_ead_dist_{}.npy'.format(idx), distances)
	print('adversarial acc:', adv_correct / len(loader.dataset))



def cws(model, fmodel, loader):
	normal_correct = 0
	adv_correct = 0
	adv_imgs, labels, distances, adv_classes = [], [], [], []
	# loader = get_data()
	print(len(loader.dataset))
	cnt = 0
	for images, label in loader:
		cnt += 1
		if cnt < 11:
			continue
		print(cnt,'/5000')
		outputs = model(images)
		_, predicted = torch.max(outputs, 1)
		images, label = images.numpy(), label.numpy()
		# print(label.shape)
		# print(np.max(images), np.min(images))
		# images, labels = foolbox.utils.samples(dataset='cifar100', batchsize=64, data_format='channels_first', bounds=(0, 1))
		_predicted = fmodel.forward(images).argmax(axis=-1)
		print(np.sum(_predicted == predicted.cpu().numpy()))

		normal_correct += np.sum(predicted.cpu().numpy() == label)
		attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=foolbox.distances.MeanSquaredDistance)
		adversarials = attack(images, label, unpack=False)
		adv_imgs += [a.perturbed for a in adversarials]
		distances += np.asarray([a.distance.value for a in adversarials])
		adversarial_classes = [a.adversarial_class for a in adversarials]
		adv_classes += adversarial_classes
		adv_correct += np.mean(adversarial_classes == label)  # will always be 0.0
		if labels == []:
			labels = label
		else:
			labels = np.hstack((labels,label))
		np.save('cifar100_advs_{}.npy'.format(cnt), adv_imgs)
		np.save('cifar100_labels_{}.npy'.format(cnt), labels)
		# break
		# labels += label
	
	print(np.max(adv_imgs), np.min(adv_imgs))
	print('normal acc:', normal_correct / len(loader.dataset))
	print('adversarial acc:', adv_correct / len(loader.dataset))
	

	# # The `Adversarial` objects also provide a `distance` attribute. Note that the distances
	# # can be 0 (misclassified without perturbation) and inf (attack failed).
	distances = np.asarray([a.distance.value for a in adversarials])
	print("{:.1e}, {:.1e}, {:.1e}".format(distances.min(), np.median(distances), distances.max()))
	print("{} of {} attacks failed".format(sum(adv.distance.value == np.inf for adv in adversarials), len(adversarials)))
	print("{} of {} inputs misclassified without perturbation".format(sum(adv.distance.value == 0 for adv in adversarials), len(adversarials)))

def main():
	model = target_model(sys.argv[-2]).eval()
	preprocessing = dict(mean=[0,0,0], std=[1,1,1], axis=-3)
	loader = get_data(sys.argv[-4])
	fmodel = foolbox.models.PyTorchModel(model, bounds=(-3, 3), num_classes=100, preprocessing=preprocessing)
	if sys.argv[-3] == 'ead':
		ead(fmodel, loader)
	elif sys.argv[-3] == 'pgd':
		pgd(fmodel, loader)
	elif sys.argv[-3] == 'cws':
		cws(model, fmodel, loader)

if __name__ == '__main__':
	_ = C(sys.argv[-1])
	main()
	
