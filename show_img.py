import numpy as np 
import matplotlib.pyplot as plt
from keras.datasets import mnist 
from keras import backend as K
import os
import sys
import torchvision
import torch
from white_box_attack import predict
import imp


def show_img():
	imgs = np.load('tests.npy')
	# labels = np.load('labels.npy')
	# err = [469, 864, 1730, 2312, 2880, 2923, 2979, 3272, 3658, 3756, 3949, 4687, 5787, 5823, 6407, 6458, 7828, 8354, 8865, 8894, 9478, 9529]
	err = [8865, 2312]
	# (x_train, y_train), (x_test, y_test) = mnist.load_data()
	# img_rows, img_cols = 28, 28

	# if K.image_data_format() == 'channels_first':
	#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	#     input_shape = (1, img_rows, img_cols)
	# else:
	#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	#     input_shape = (img_rows, img_cols, 1)

	# x_train = x_train.astype('float32')
	# x_test = x_test.astype('float32')
	# x_train /= 255
	# x_test /= 255

	for i in err:
		img = imgs[i]
		# print(i, labels[i])
		plt.imshow(np.reshape(img, (28, 28)), cmap=plt.get_cmap('gray'))
		plt.savefig('{}th_{}.png'.format(i, 0))#labels[i]))
		# plt.show()

def show_generated_imgs(file, order):
	idxs = np.arange(30)
	info = file.split('_')
	imgs = np.load(file)
	normal_label, _, normal_pred = predict(imgs[1], idxs)
	adv_label, _, adv_pred = predict(imgs[0], idxs)
	if adv_label != int(info[-3]):
		print('error in your code')
		return
	grid = torchvision.utils.make_grid([torch.Tensor(imgs[0]), torch.Tensor(imgs[1])], nrow=2)
	plt.imshow(grid.numpy().transpose((1, 2, 0)))
	if normal_label != int(info[3]):
		plt.title('[false positive]\nadv:{}_org_pred:{}_org:{}: {} distance:{}'.format(info[-3], normal_label, info[-2], order, info[-1][:-4]))
		plt.savefig('whites/1_l_inf_imgs/wr/{}.png'.format(file.split('/')[-1][:-4]))
	else:
		plt.title('adv:{}_org_pred:{}_org:{}: {} distance:{}'.format(info[-3], normal_label, info[-2], order, info[-1][:-4]))
		plt.savefig('whites/1_l_inf_imgs/advs/{}.png'.format(file.split('/')[-1][:-4]))

def main(path):
	files = os.listdir(path)
	for file in files:
		show_generated_imgs(os.path.join(path, file), np.inf)


if __name__ == '__main__':
	main(sys.argv[-1])