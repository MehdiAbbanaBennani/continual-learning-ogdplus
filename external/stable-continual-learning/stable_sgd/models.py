import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d


class MLP(nn.Module):
	"""
	Two layer MLP for MNIST benchmarks.
	"""
	def __init__(self, hiddens, config):
		super(MLP, self).__init__()
		self.W1 = nn.Linear(784, hiddens)
		self.relu = nn.ReLU(inplace=True)
		self.dropout_1 = nn.Dropout(p=config['dropout'])
		self.W2 = nn.Linear(hiddens, hiddens)
		self.dropout_2 = nn.Dropout(p=config['dropout'])
		self.W3 = nn.Linear(hiddens, 10)

	def forward(self, x, task_id=None):
		x = x.view(-1, 784)
		out = self.W1(x)
		out = self.relu(out)
		out = self.dropout_1(out)
		out = self.W2(out)
		out = self.relu(out)
		out = self.dropout_2(out)
		out = self.W3(out)
		return out


def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, config={}):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(in_planes, planes, stride)
		self.conv2 = conv3x3(planes, planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
						  stride=stride, bias=False),
			)
		self.IC1 = nn.Sequential(
			nn.BatchNorm2d(planes),
			nn.Dropout(p=config['dropout'])
		)

		self.IC2 = nn.Sequential(
			nn.BatchNorm2d(planes),
			nn.Dropout(p=config['dropout'])
		)

	def forward(self, x):
		out = self.conv1(x)
		out = relu(out)
		out = self.IC1(out)

		out += self.shortcut(x)
		out = relu(out)
		out = self.IC2(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes, nf, config={}):
		super(ResNet, self).__init__()
		self.in_planes = nf

		self.conv1 = conv3x3(3, nf * 1)
		self.bn1 = nn.BatchNorm2d(nf * 1)
		self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, config=config)
		self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, config=config)
		self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, config=config)
		self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, config=config)
		self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride, config):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, config=config))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x, task_id):
		bsz = x.size(0)
		out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		t = task_id
		offset1 = int((t-1) * 5)
		offset2 = int(t * 5)
		if offset1 > 0:
			out[:, :offset1].data.fill_(-10e10)
		if offset2 < 100:
			out[:, offset2:100].data.fill_(-10e10)
		return out


def ResNet18(nclasses=100, nf=20, config={}):
	net = ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, config=config)
	return net



class LeNet(nn.Module):
	def __init__(self, dropout, out_dim=10, in_channel=1, img_sz=32, hidden_dim=500):
		super(LeNet, self).__init__()
		feat_map_sz = img_sz//4
		self.n_feat = 50 * feat_map_sz * feat_map_sz
		self.hidden_dim = hidden_dim

		self.linear = nn.Sequential(
			nn.Conv2d(in_channel, 20, 5, padding=2),
			nn.BatchNorm2d(20),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(20, 50, 5, padding=2),
			nn.BatchNorm2d(50),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2),
			nn.Flatten(),
			nn.Dropout(p=dropout),
			nn.Linear(self.n_feat, hidden_dim),
			# TODO : Check why bug when using batchnorm with OGD+
			# nn.BatchNorm1d(hidden_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(p=dropout),
		)

		self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

	def features(self, x):
		return self.linear(x)

	def logits(self, x):
		x = self.last(x)
		return x

	def forward(self, x, task_id):
		x = self.features(x)
		out = self.logits(x)

		t = task_id
		offset1 = int((t - 1) * 5)
		offset2 = int(t * 5)
		if offset1 > 0:
			out[:, :offset1].data.fill_(-10e10)
		if offset2 < 100:
			out[:, offset2:100].data.fill_(-10e10)
		return out


def LeNetCIFAR100(hidden_dim, dropout, out_dim=100):  # LeNet with color input
	return LeNet(out_dim=out_dim, in_channel=3, img_sz=32, hidden_dim=hidden_dim, dropout=dropout)