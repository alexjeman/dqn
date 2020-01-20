import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torch import  nn
from torch.autograd import Variable
from torch import  optim
from torchvision import transforms
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3),  # 16, 26 ,26
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True))


        self.layer2 = nn.Sequential(nn.Conv2d(16, 32,kernel_size=3),  # 32, 24, 24
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2,stride=2)) # 32, 12,12     (24-2) /2 +1


        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 128, 4,4

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#train_set = mnist.MNIST('./data',train=True)
#test_set = mnist.MNIST('./data',train=False)


data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])

train_set = mnist.MNIST('./data',train=True,transform=data_tf,download=True)
test_set = mnist.MNIST('./data',train=False,transform=data_tf,download=True)


train_data = DataLoader(train_set,batch_size=64,shuffle=True)
test_data = DataLoader(test_set,batch_size=128,shuffle=False)


net = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),1e-1)

nums_epoch = 20

losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(nums_epoch):
    train_loss = 0
    train_acc = 0
    net = net.train()
    for img, label in train_data:
        img = img.to(device)
        label = label.to(device)

        out = net(img)
        optimizer.zero_grad()
        loss = criterion(out, label)
        print(loss)
        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc

        losses.append(train_loss / len(train_data))
        acces.append(train_acc / len(train_data))

        eval_loss = 0
        eval_acc = 0
        for img, label in test_data:
            # img = img.reshape(img.size(0),-1)
            img = img.to(device)
            label = label.to(device)
            out = net(img)

            loss = criterion(out, label)

            eval_loss += loss.item()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            eval_acc += acc
        eval_losses.append(eval_loss / len(test_data))
        eval_acces.append(eval_acc / len(test_data))
        print('Epoch {} Train Loss {} Train  Accuracy {} Teat Loss {} Test Accuracy {}'.format(
            epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
            eval_acc / len(test_data)))
