import torchvision
import numpy as np
import torch
from cws import get_data

# use to reverse transform to recover image
invTrans = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
                                torchvision.transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465],
                                                     std = [ 1., 1., 1. ]),
                               ])
loader = get_data('/home/b90/Desktop/data')
for images, label in loader:
	img_ = invTrans(images[0])
	break

images = np.load('cifar100_adv.npy')
img = invTrans(torch.Tensor(images[0]))
# _img = invTrans(torch.from_numpy(images[0]))

print(np.sum(img_.numpy() - img.numpy()))
torchvision.utils.save_image(img, 'img.png')
torchvision.utils.save_image(img_, '_img.png')