import torch
import torch.nn as nn
from FastAutoAugment.networks.pyramidnet import PyramidNet

class Plus(nn.Module):

	def __init__(self, dataset, depth, alpha, num_classes, bottleneck=True):
		super(Plus, self).__init__()
		self.model = PyramidNet(dataset, depth=depth, alpha=alpha, num_classes=100, bottleneck=bottleneck)
		self.fc = nn.Linear(100, num_classes)

	def forward(self, x):
		x = self.model(x)
		x = self.fc(x)
		return x
