## -*- coding: utf-8 -*-
from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import math
import datetime
import time
import itertools
import code
import argparse
import random
import torch.utils.data as data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import scipy.misc as sm
from torch.autograd import Variable
import torchvision as vision
import scipy.ndimage as nd

import torch.nn.functional as F

import json

import math
import torch.utils.data as data
import networks
from PIL import Image
from PIL import ImageChops
import os.path
from tqdm import tqdm
from IPython import embed
from scipy.spatial import distance

torch.cuda.manual_seed(1)

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str,  help='dataset depth folder: JHUIT or split0 split1 etc')
parser.add_argument('--lr', type=float, default=0.001, help='starting learning rate.')
parser.add_argument('--wd', type=float, default=0.0001, help='weight_decay, default to 0')
parser.add_argument('--nepoch', type=int, default=90, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '' ,  help='model path')
parser.add_argument('--test_epochs', type=int, default=10, help='run testing phase every test_epochs ')
parser.add_argument('--lr_gamma', type=float, default=0.1, help='learning rate decrease factor.Suggested 1.0 for Adam, 0.1 per SGD/Nesterov')

opt = parser.parse_args()

ts = time.time()
timestamp_folder = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y %H:%M')

root_weights = timestamp_folder+'/weights/'
root_data = timestamp_folder+'/data_acc_loss/'

print(timestamp_folder)
print(root_weights)
print(root_data)

if (os.path.exists(root_weights) or os.path.exists(root_data)):
       print('Directory Exists')

if not os.path.exists(root_weights):
       os.makedirs(root_weights)

if not os.path.exists(root_data):
       os.makedirs(root_data)


def imageNetloader(batchsize):
    my_transform = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    imagenet_data = dset.ImageFolder('./ILSVRC2012_img_train/', transform=my_transform)
    my_dataloader = data.DataLoader(imagenet_data, batch_size = batchsize, shuffle = True, num_workers = 1)
    return my_dataloader

def imageNet_val_loader(batchsize):

    my_transform = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    imagenet_val_data = dset.ImageFolder('./ILSVRC2012_img_val/', transform=my_transform)
    my_val_dataloader = data.Dataloader(imagenet_val_data, batch_size = batchsize, shuffle = False, num_workers = 1)
    return my_val_dataloader

#def update_labels(labels,classes,images):
#    for i,value in enumerate(labels):
#        if not value in classes:
#            classes = np.append(classes,value)
#            classes.astype('int')
#            mean[value,:] = images[i,:] #.data.cpu().numpy()
#        else:
#            if mean[value,:].sum()==0:
#                mean[value,:] = images[i,:] #.data.cpu().numpy()
#            else:
#                mean[value,:] = (mean[value,:]+images[i,:])/2.0
#    return classes, mean

#def compute_distances(num_classes, batch_size, mean, images):
#    dst = torch.zeros((num_classes,))
#    dst = dst.cuda()
#    pred = torch.zeros((batch_size,))
#    dst_tot = torch.zeros((batch_size,num_classes),requires_grad=True)
#    dst_tot = dst_tot.cuda()
#    for count,im in enumerate(images):
#        for i in range(mean.shape[0]):
#            dst[i] = torch.sum((im-mean[i,:])**2) 
#        dst_tot[count,:] = dst
#        pred[count] = torch.argmin(dst_tot)

#    return pred, dst_tot

def update_labels(labels,visited,images,alpha=0.1):
    for i,value in enumerate(labels):
        if not value in visited:
	    visited.append(value)
	    mask=(labels==value).float()
	    N=mask.sum()
            mask = mask.view(-1,1)
            if mean[value,:].sum()==0:
                mean[value,:] = (images*mask).sum(dim=0)/N
            else:
                mean[value,:] = (images*mask).sum(dim=0)/N*alpha + (1-alpha)*mean[value,:] 
    return visited, mean


def compute_distances(num_classes, batch_size, mean, images,labels):
    print(images.shape)
    pred = torch.zeros((batch_size,1)).cuda()
    means_reshaped=mean.view(1,num_classes,-1).expand(batch_size,num_classes,mean.shape[1])
    features_reshaped=images.view(batch_size,1,-1).expand(batch_size,num_classes,mean.shape[1])
    print(features_reshaped.shape)

    diff=(features_reshaped-means_reshaped)**2
    diff = diff.sum(dim=-1)**0.5
    for i,value in enumerate(diff):
       pred[i] = torch.argmin(diff[i])
    return diff , pred


#def compute_distances2(num_classes, batch_size, mean, images):
#    dst_tot = torch.zeros((batch_size,num_classes),requires_grad=True)
#    for b in range(batch_size):
#        for i in range(mean.shape[0]):
#            dst_tot[b,i] =torch.sum((images[b,:]-mean[i,:])**2)
#
#    return dst_tot

#    for i in range(mean.shape[0]):
#         dst_tot = torch.cat((dst_tot,torch.sum((images-mean[i,:])**2)),1)
    #dst_tot[:,i] = torch.sum((images-mean[i,:])**2)
#    _,pred = dst_tot.min(1)
    #ritorna la matrice delle distanze delle immagini da tutti i centri e le labels predette


def train(epoch, lr):

    global prediction
    global distances
    global distances2
    global parameters
    global optimizer
    global correct_train
    global total_train
    global visited

    correct_train = 0.0
    total_train = 0.0
    model_0.train()

    my_dataloader = imageNetloader(Batch_size)

    for i, (images,labels) in enumerate(my_dataloader):

        images = images.cuda()
        labels = labels.cuda()

        #Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model_0(images)

        # calcolo medie
        visited, mean = update_labels(labels,visited,outputs.data)
        # calcolo distanze
        distances, prediction = compute_distances(mean.shape[0], images.shape[0], mean,outputs, labels)
        #definizione loss
        loss = criterion(distances,labels)
        if math.isnan(loss)==True:
            print(i)
        loss.backward()
        optimizer.step()

        correct_train += prediction[:,0].eq(labels.float()).sum()
        correct_train = correct_train.float().item()
#        print('correct train:',correct_train) 

        total_train += labels.size(0)
#        print('total train:',total_train)

        accuracy_train = 1.*correct_train/ total_train  #.item()/total_train

#        print('accuracy train:',accuracy_train)
        acc.write(str(accuracy_train) + '\n')
        loss_f.write(str(loss) + '\n')

        if (i+1) % 10 == 0:
            print("Epoch [%d/%d], Iter [%d]  Loss: %.6f Accuracy: %.6f"
                    %(epoch+1, num_epoch, i+1, loss, accuracy_train))
            res = "Epoch ["+str(epoch+1)+'/'+str(num_epoch)+" Iter "+str(i+1)+" Loss: "+str(loss.data)+" Accuracy: "+str(accuracy_train)
            resume_file.write(res+'\n') 



    #Save the model
    filesave = root_weights+'epoch'+str(epoch)+'.pkl'
    torch.save(model_0.state_dict(), filesave)

def test(epoch):
    global correct_0
    global total_0
    global num_epoch
    correct = 0.0
    total = 0.0
    model_0.eval() #; testdataloader.init_epoch()

    my_val_dataloader = imageNet_val_loader(Batch_size)

    for i, (images,labels) in enumerate(my_val_dataloader):
        images =  images.cuda() #trasforma in variabili$
        labels = images.cuda()

        #Forward + Backward + Optimize
        outputs = model_0(images)
        loss = criterion(outputs, labels)
        predicted = outputs.data.max(1)[1]
        correct += predicted.eq(labels.data).cpu().sum()
        total += labels.size(0)
        loss_f_test.write(str(loss.data[0]) + '\n')

    accuracy = 1. * correct.item()/total

    print ("Test %d  Accuracy: %.6f"
            %(epoch, accuracy))

    acc_test.write(str(accuracy) + '\n')
    acc_test.flush()


def main():

    global Batch_size
    global model_0
    global optimizer
    global criterion
    global acc
    global loss_f
    global acc_test
    global loss_f_test
    global num_epoch
    global resume_file
    global mean
    global visited

    Batch_size = 128

    model_0 = networks.ResNet_(ResNet.BasicBlock, [3,4,6,3]).cuda()
    #model_0 = ResNet.resnet32().cuda()
    #model_0 = torchvision.models.resnet34().cuda()

    mean = torch.zeros((1000,1000))
    mean = mean.cuda()
    visited = []

    acc = open('%s/acc.txt' %(root_data), 'w+')
    loss_f = open('%s/loss.txt' %(root_data), 'w+')
    acc_test  = open('%s/acc_test.txt' %(root_data), 'w+')
    loss_f_test = open('%s/loss_test.txt' %(root_data), 'w+')
    filename = timestamp_folder+'/results.txt'
    resume_file = open(filename, "w+")

    correct_train = 0
    total_train = 0
    correct = 0
    total = 0

    # Loss and Optimizer
    #s = nn.Softmax()
    criterion = nn.CrossEntropyLoss().cuda()
    #criterion = nn.MSELoss().cuda()
    lr = opt.lr

    for param in model_0.parameters():
        param.requires_grad = True

    parameters = itertools.ifilter(lambda p: p.requires_grad, model_0.parameters())

    print("Using Nesterov optimizer")
    optimizer = torch.optim.SGD(params = parameters, lr=lr, weight_decay=opt.wd,momentum=0.9,nesterov=True)

    num_epoch = opt.nepoch

    if opt.model:
        model_0.load_state_dict(torch.load(opt.model))
        print ("LOADING MODEL SNAPSHOT")


    #training/testing loop
    for epoch in range(num_epoch):   #fa il test ogni tot epoche
        train(epoch,lr)
        if (math.fmod(epoch+1,opt.test_epochs) == 0):
            print ("epoch: %d" % epoch)
            test(epoch)

        #reducing learning rate every num_epoch/2 epochs:
        #if (math.fmod(epoch+1,num_epoch/3) == 0):
        #    lr = lr * opt.lr_gamma # lr_gamma usually is 0.1
        #    print ("new learning rate: %.6f" % lr)

        lr = opt.lr_gamma * (0.1 ** (epoch // 30))
        print ("new learning rate: %.6f" % lr)


    acc.close()
    loss_f.close()
    acc_test.close()
    loss_f_test.close()
    resume_file.close()

if __name__ == '__main__':
    main()
