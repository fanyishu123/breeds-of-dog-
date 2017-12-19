#!/u/yifan/anaconda3/bin/python

import torch
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#Hyper Parameters

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001



class DogSet(Dataset):


	def __init__(self, csv_file, root_dir, transform = None, trainsize = 10000, train = True):
		if train :
			self.label_frame = pd.read_csv(csv_file)[:trainsize]
		else:
			self.label_frame = pd.read_csv(csv_file)[trainsize:]
		self.root_dir = root_dir
		self.transform = transform
		set_breed = set(self.label_frame.breed)
		self.breeds={breed : i for i, breed in enumerate(set_breed)}

	def __len__(self):
		return len(self.label_frame)


	def __getitem__(self, idx):
		
		image_name = os.path.join(self.root_dir, self.label_frame.id[idx])+'.jpg'
		image = io.imread(image_name)
		breed = self.label_frame.breed[idx]
		label = self.breeds[breed]

		if self.transform:
			image = self.transform(image)
		return image, label



		
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(nn.Conv2d(3,16,5,stride =1,padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 =torch.nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out=nn.Linear(32*75*75, 120)


    def forward(self, x):
        x =self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        out = self.out(x)
        return output, x


train_dog =  DogSet(csv_file = 'labels.csv', 
		root_dir='./train', 
		transform = transforms.Compose(
			[transforms.ToPILImage(),
			transforms.Resize(400),
			transforms.RandomResizedCrop(300),
			transforms.ToTensor()
			]),
		train = True,
		trainsize = 8000
		)
		

test_dog = Dogset(csv_file = 'labels.csv',
                root_dir = './train',
                transform = transforms.Compose(
                [   transforms.ToPILImage(),
                    transforms.Resize(400),
                    transforms.RandomResizedCrop(300),
                    transforms.ToTensor()
                    ]),
                train = False,
                trainsize = 8000
                )
train_loader = DataLoader(dataset =train_dog, batch_size=BATCH_SIZE, shuffle = True)
test_loader = DataLoader(dataset = test_dog, batch_size = 3000, shuffle = True)
cnn = CNN()
print (cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
loss_func =nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader)
        b_x = Variable(x)
        b_y=  Variable(y)
        optimizer.zero_grad()
        output = cnn(b_x)
        loss = loss_func(output, b_y)

        loss.backward()
        optimizer.step()
        
        if step%50==0:
            for (test_x, test_y) in test_loader:
                optimizer.zero_grad()
                test_out, _ = cnn(Variable(test_x))
                loss = loss_func(test_out,Variable(test_y))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

# print (len(train_dog))
#train_data = dog_dataset[0:8000]
#test_data = dog_dataset[8000:]




