import numpy as np 
def evaluation(nb_res):
	rep = np.load('2_label_permutation_cifar100.npy')[10:40]
	samples = np.load('res_48_{}.npy'.format(nb_res)).T[10:40]
	labels = np.load('cifar100_labels_{}.npy'.format(nb_res))
	orders = np.random.permutation(samples.shape[0])
	ex_labels = [[rep[i][j] for j in labels] for i in orders]
	valid = samples == ex_labels
	sep_valid = [[np.max(val[j:j+3]) for j in np.arange(valid.shape[-1])[::3]] for val in valid]
	valid = np.sum(sep_valid, axis=-1)==10
	print('acc:', np.sum(valid))

if __name__ == '__main__':
	evaluation(sys.argv[-1])
