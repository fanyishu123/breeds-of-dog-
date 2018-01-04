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
import torchvision.models as models
import torchvision
#torch.manual_seed(1) 
#Hyper Parameters

EPOCH = 50
BATCH_SIZE = 50
LR = 0.001


dog_pd = pd.read_csv('labels.csv')

dog_pd_label = pd.read_csv('sample_submission.csv')

labels = dog_pd_label.columns[1:]
label_list = []

for label in labels: 
    label_list.append(label)

label_list.sort()

label_to_ix = {breed: i for i, breed in enumerate(label_list)}

print (label_list)

print (type(label_to_ix))

read_model_file =  'train_b70e100_onlyStandDog.pkl'
save_model_file =  'train_b70e100_onlyStandDog_dogused_2.pkl'
def restore_model(param_file):
    model = torchvision.models.resnet152(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)
    model.load_state_dict(torch.load(param_file))
    return model

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)

class DogSet(Dataset):


    def __init__(self, dog_pd, root_dir, label_to_ix, transform = None, trainsize = 10000, train = True):
        if train :
            self.label_frame = dog_pd[:trainsize]
            self.idx_start = 0 
        else:
            self.label_frame = dog_pd[trainsize: trainsize+2000]
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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
train_dog =  DogSet(label_to_ix = label_to_ix, dog_pd = dog_pd, 
        root_dir='./train', 
        transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize 
            ]),
        train = True,
        trainsize = 8000
        )
        

test_dog = DogSet(label_to_ix = label_to_ix, dog_pd = dog_pd,
                root_dir = './train', 
                transform = transforms.Compose(
                [   transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                    ]),
                train = False,
                trainsize = 8000
                )
                
print("test_dog set:", len(test_dog))

train_loader = DataLoader(dataset =train_dog, batch_size=BATCH_SIZE, shuffle = True,num_workers=4)
test_loader = DataLoader(dataset = test_dog, batch_size =50, shuffle = False, num_workers=4)

# cnn = CNN()
# cnn.cuda()
# print (cnn)



# x0, y0, breed0 =test_dog[0]
# print ('x0 size:',x0.size())
# x0_ = torch.unsqueeze(x0, 0)
# x0_out= cnn(Variable(x0_))

# print ('x0_out:',torch.max(x0_out, 1)[1].data.numpy())
# print ('y0:', y0, breed0)

#test_dog_1 = test_dog[0:10]

#model_conv = torchvision.models.resnet152(pretrained=True)


model_conv = restore_model(read_model_file)

# for param in model_conv.parameters():
    # param.requires_grad = False

#print (model_conv)

#num_ftrs = model_conv.fc.in_features
#model_conv.fc = nn.Linear(num_ftrs, 120)

model_conv.cuda()

#optimizer = torch.optim.Adam(model_conv.fc.parameters(), lr = LR)
optimizer = torch.optim.Adam(model_conv.parameters(), lr = LR)
loss_func =nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    running_loss = 0.0
    running_accuracy = 0.0 
    for step, (x,y, breed) in enumerate(train_loader):
        b_x = Variable(x.cuda())
        b_y=  Variable(y.cuda())
        optimizer.zero_grad()
        output= model_conv(b_x)
        loss = loss_func(output, b_y)

        loss.backward()
        optimizer.step()
        pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
        accuracy = sum(pred_y == y.cuda()) / float(y.size(0))
        running_loss += loss.data[0]
        running_accuracy += accuracy
        print('Epoch: ', epoch, '|Step:', step, '| train loss: %.4f' % loss.data[0], '|train accuracy: %.2f' % accuracy)
        # if step%50==0:
            # for i, (test_x, test_y, test_breed) in enumerate(test_loader):
                # optimizer.zero_grad()
                # test_out= cnn(Variable(test_x.cuda()))
                # test_loss = loss_func(test_out,Variable(test_y.cuda()))
                # pred_y = torch.max(test_out, 1)[1].cuda().data.squeeze()
                # test_y = test_y.cuda()
                # accuracy = sum(pred_y == test_y) / float(test_y.size(0))
                # print('Epoch: ', epoch, '|Step:', step, '| train loss: %.4f' % test_loss.data[0], '|test accuracy: %.2f' % accuracy)
                # if i== 0:
                   # break
        #if step == 50:
        #    break
    epoch_loss = running_loss / (step+1)
    epoch_acc = running_accuracy / (step+1)
    print('Epoch: {:.4f} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
    
save_model(model_conv, save_model_file)

test_loss = 0.0
for i, (test_x, test_y, test_breed) in enumerate(test_loader):
    optimizer.zero_grad()
    test_out= model_conv(Variable(test_x.cuda()))
    loss = loss_func(test_out,Variable(test_y.cuda()))
    pred_y = torch.max(test_out, 1)[1].cuda().data.squeeze()
    test_y = test_y.cuda()
    test_loss +=loss.data[0]
    accuracy += sum(pred_y == test_y)    
print ('test loss: {:.4f} |test accuracy: {:.2f}' .format(test_loss/(i+1), accuracy/(len(test_dog))))
# print (len(train_dog))
# train_data = dog_dataset[0:8000]
# test_data = dog_dataset[8000:]




