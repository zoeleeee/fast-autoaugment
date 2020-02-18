import torch
import torch.nn as nn
from FastAutoAugment.networks.pyramidnet import PyramidNet

class Flower(nn.Module):

	def __init__(self, dataset, depth, alpha, num_classes, bottleneck=True):
		super(Flower, self).__init__()
		self.model = PyramidNet(dataset, depth=depth, alpha=alpha, num_classes=1, bottleneck=bottleneck)
		self.fc = nn.Linear(1, num_classes)

	def forward(self, x):
		x = self.model(x)
		x = self.fc(x)
		x = torch.sigmoid(x)
		return x