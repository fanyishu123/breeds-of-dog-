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
from torch.autograd import Variable


torch.manual_seed(1) 
#Hyper Parameters

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001


dog_pd = pd.read_csv('labels.csv')

dog_pd_label = pd.read_csv('sample_submission.csv')

labels = dog_pd_label.columns[1:]

label_to_ix = {breed: i for i, breed in enumerate(labels)}


class DogSet(Dataset):


    def __init__(self, dog_pd, root_dir, label_to_ix, transform = None, trainsize = 10000, train = True):
        if train :
            self.label_frame = dog_pd[:trainsize]
            self.idx_start = 0 
        else:
            self.label_frame = dog_pd[trainsize: trainsize+1000]
            self.idx_start = trainsize
        self.root_dir = root_dir
        self.transform = transform
        self.breeds=label_to_ix

    def __len__(self):
        return len(self.label_frame)


    def __getitem__(self, idx):
        idx = self.idx_start + idx
        image_name = os.path.join(self.root_dir, self.label_frame['id'][idx])+'.jpg'
        image = io.imread(image_name)
        breed = self.label_frame['breed'][idx]
        label_ix = self.breeds[breed]


        if self.transform:
            image = self.transform(image)
        return image, label_ix, breed




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
        return out


train_dog =  DogSet(label_to_ix = label_to_ix, dog_pd = dog_pd, 
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
        

test_dog = DogSet(label_to_ix = label_to_ix, dog_pd = dog_pd,
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
                
                
train_loader = DataLoader(dataset =train_dog, batch_size=BATCH_SIZE, shuffle = True,num_workers=4)
test_loader = DataLoader(dataset = test_dog, batch_size =BATCH_SIZE, shuffle = False, num_workers=4)

cnn = CNN()
print (cnn)



# x0, y0, breed0 =test_dog[0]
# print ('x0 size:',x0.size())
# x0_ = torch.unsqueeze(x0, 0)
# x0_out= cnn(Variable(x0_))

# print ('x0_out:',torch.max(x0_out, 1)[1].data.numpy())
# print ('y0:', y0, breed0)

#test_dog_1 = test_dog[0:10]


optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
loss_func =nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x,y, breed) in enumerate(train_loader):
        b_x = Variable(x)
        b_y=  Variable(y)
        optimizer.zero_grad()
        output = cnn(b_x)
        loss = loss_func(output, b_y)

        loss.backward()
        optimizer.step()
        
        if step%10==0:
            for i, (test_x, test_y, test_breed) in enumerate(test_loader):
                optimizer.zero_grad()
                test_out= cnn(Variable(test_x))
                loss = loss_func(test_out,Variable(test_y))
                pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
                accuracy = sum(pred_y == test_y) / float(test_y.size(0))
                print('Epoch: ', epoch, '|Step:', step, '| train loss: %.4f' % loss.data[0], '|test accuracy: %.2f' % accuracy)
                if i== 0:
                   break
        if step == 50:
            break
        

# print (len(train_dog))
# train_data = dog_dataset[0:8000]
# test_data = dog_dataset[8000:]




