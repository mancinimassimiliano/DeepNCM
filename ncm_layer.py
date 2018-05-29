import numpy as np
import torch
import torch.nn as nn


class NCM_classifier(nn.Module):
	
	# Initialize the classifier
	def __init__(self, features, classes, alpha=0.95):
		super(NCM_classifier, self).__init__()
		self.means=nn.Parameter(torch.zeros(classes,features),requires_grad=False)				# Class Means
		self.labels={}				# Class Labels, to convert the order of labels to the actual labels of the dataset
		self.alpha=alpha			# Mean decay value
		self.features=features			# Input features
		self.classes=classes
		

	# Forward pass (x=features)
	def forward(self,x):
		means_reshaped=self.means.view(1,self.classes,self.features).expand(x.shape[0],self.classes,self.features)
		features_reshaped=x.view(-1,1,self.features).expand(x.shape[0],self.classes,self.features)
		diff=(features_reshaped-means_reshaped)**2

		return -diff.sum(dim=-1)**0.5
			
	
	# Update centers (x=features, y=labels)
	def update_means(self,x,y):
		holder = torch.zeros_like(self.means)	# Init mean holders
		holder.data=self.means.data		# Copy data for easy update
		for i in torch.unique(y):				# For each label
			N,mean=self.compute_mean(x,y,i)	# Compute mean

			# If labels already in the set, just update holder, otherwise add it to the model
			if N==0:
				holder.data[i,:]=self.means.data[i,:]
			else:
				holder.data[i,:]=mean
		
		# Update means
		self.update(holder)
	

	# Perform the update following the mean decay procedure
	def update(self,holder):
		self.means.data=self.alpha*self.means.data+(1-self.alpha)*holder



	# Compute mean by filtering the data of the same label
	def compute_mean(self,x,y,i):
		mask=(i==y).view(-1,1).float()
		mask=mask.cuda()
		N=mask.sum()
		if N==0:
			return N,0
		else:
			return N,(x.data*mask).sum(dim=0)/N






