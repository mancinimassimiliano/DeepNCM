'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




class NCM_classifier(nn.Module):
	
	# Initialize the classifier
	def __init__(self, features, classes, alpha=0.9):
		super(NCM_classifier, self).__init__()
		self.means=nn.Parameter(torch.zeros(classes,features),requires_grad=False)				# Class Means
		self.running_means=nn.Parameter(torch.zeros(classes,features),requires_grad=False)
		self.alpha=alpha			# Mean decay value
		self.features=features			# Input features
		self.classes=classes
		

	# Forward pass (x=features)
	def forward(self,x):
		means_reshaped=self.means.view(1,self.classes,self.features).expand(x.shape[0],self.classes,self.features)
		features_reshaped=x.view(-1,1,self.features).expand(x.shape[0],self.classes,self.features)
		diff=(features_reshaped-means_reshaped)**2
		cumulative_diff=diff.sum(dim=-1)

		return -cumulative_diff
			
	
	# Update centers (x=features, y=labels)
	def update_means(self,x,y):
		for i in torch.unique(y):				# For each label
			N,mean=self.compute_mean(x,y,i)	# Compute mean

			# If labels already in the set, just update holder, otherwise add it to the model
			if N==0:
				self.running_means.data[i,:]=self.means.data[i,:]
			else:
				self.running_means.data[i,:]=mean
		
		# Update means
		self.update()
	

	# Perform the update following the mean decay procedure
	def update(self):
		self.means.data=self.alpha*self.means.data+(1-self.alpha)*self.running_means



	# Compute mean by filtering the data of the same label
	def compute_mean(self,x,y,i):
		mask=(i==y).view(-1,1).float()
		mask=mask.cuda()
		N=mask.sum()
		if N==0:
			return N,0
		else:
			return N,(x.data*mask).sum(dim=0)/N



class ResNet_NCM(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_NCM, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = NCM_classifier(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #out = self.linear(out)
        return out

    def update_means(self, x,y):
        self.linear.update_means(x,y)

    def predict(self, x):
        out = self.linear(x)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet18_NCM():
    return ResNet_NCM(BasicBlock, [2,2,2,2])

def ResNet34_NCM(classes=10):
    return ResNet_NCM(BasicBlock, [3,4,6,3],num_classes=classes)

def ResNet34(classes=10):
    return ResNet(BasicBlock, [3,4,6,3],num_classes=classes)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()

