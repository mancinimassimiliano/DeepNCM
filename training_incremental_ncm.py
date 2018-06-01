'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import networks
import numpy as np
import argparse

import visdom

from utils import progress_bar


EPOCHS=250
LR=0.1

vis = visdom.Visdom()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset',  type=int, default=10, help='choose dataset')
parser.add_argument('--before',  type=int, default=0, help='update_position')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset==10:
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
elif args.dataset==100:
	trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

	testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = networks.ResNet34_iNCM()
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


lr=LR
# Training
def train(epoch,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs= inputs.to(device)
        optimizer.zero_grad()
	net.prepare(targets)
        outputs = net.forward(inputs)
	if args.before==0:
		net.update_means(outputs,targets)
	prediction=net.predict(outputs)
	if args.before==1:   
                net.update_means(outputs,targets)
	targets_converted=net.linear.convert_labels(targets).to(outputs.device)
        loss = criterion(prediction, targets_converted)
	if args.before==2:   
                net.update_means(outputs,targets)
        loss.backward()
	
        optimizer.step()

        train_loss += loss.item()
        _, predicted = prediction.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets_converted).sum().item()
	if batch_idx%500==0:
        	progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return (train_loss/(batch_idx+1)), 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs= inputs.to(device)
            outputs = net.forward(inputs)
	    outputs=net.predict(outputs)
	    targets_converted=net.linear.convert_labels(targets).to(outputs.device)
            loss = criterion(outputs, targets_converted)
	
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets_converted).sum().item()
	    if batch_idx%100==0:
		progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    return acc


def loop(epochs=200,dataset_name='cifar'+str(args.dataset)):
	vis.env ='incremental deep ncm' + dataset_name+str(args.before)
	model_name='DEEP NCM incremental: '+str(args.before)
	iters=[]
	losses_training=[]
	accuracy_training=[]
	accuracies_test=[]
	lr=LR
	optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, momentum=0.9, weight_decay=5e-4)
	#for name,param in net.named_parameters():
	#	if param.requires_grad:
	#		print(name)
	for epoch in range(start_epoch, start_epoch+epochs):

		if epoch%50==0 and epoch>50:
			lr=lr*0.1
			optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=5e-4)
			print('LR now is ' + str(lr))

		# Perform 1 training epoch
		loss_epoch, acc_epoch = train(epoch,optimizer)

		# Validate the model
		result = test(epoch)

		# Update lists (for visualization purposes)
		accuracies_test.append(result)
		accuracy_training.append(acc_epoch)
		losses_training.append(loss_epoch)
		iters.append(epoch)
	

		# Print results
		vis.line(
				X=np.array(iters),
				Y=np.array(losses_training),
		 		opts={
		        		'title': ' Training Loss ' ,
		        		'xlabel': 'epochs',
		        		'ylabel': 'loss'},
		    			name='Training Loss ',
		    		win=10)
		vis.line(
		    		X=np.array(iters),
		    		Y=np.array(accuracy_training),
		    		opts={
		        		'title': ' Training Accuracy ',
		        		'xlabel': 'epochs',
		        		'ylabel': 'accuracy'},
		    			name='Training Accuracy ',
		    		win=11)
		vis.line(
		    		X=np.array(iters),
		    		Y=np.array(accuracies_test),
		    		opts={
		        		'title': ' Accuracy ',
		        		'xlabel': 'epochs',
		        		'ylabel': 'accuracy'},
		    			name='Validation Accuracy ',
		    		win=12)


loop(epochs=EPOCHS)
