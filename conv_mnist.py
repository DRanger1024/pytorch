import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
import  torch.nn.functional as F
from torchvision import datasets,transforms



# 数据集的下载和装载
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
root = './data'
train_data = datasets.MNIST(root,train=True,download=True,transform=transform)
test_data =datasets.MNIST(root,train=False,download=True,transform=transform)
train_loader =Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=2)
test_loader = Data.DataLoader(dataset=test_data,batch_size=64,num_workers=2)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,5,1,2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.out = nn.Linear(16*7*7,10)
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return output
# 网络构建
if __name__ == '__main__':
    cnn =CNN()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.02)
    loss_fun = torch.nn.CrossEntropyLoss()
    for epoch in range(15):
        running_loss = 0
        correct = 0
        for step,(train_x,train_y) in enumerate(train_loader):
            out= cnn(train_x)
            _, pred = torch.max(out.data, 1)
            correct += (pred == train_y).sum()
            loss = loss_fun(out,train_y)
            running_loss +=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%100 == 0:
                print(step)
        print('epoch', epoch, 'loss', running_loss / 60000, 'correct', float(correct.numpy() / 60000)*100)

    correct = 0
    for  i,(data_x,data_y)in enumerate(test_loader):
        out=cnn(data_x)
        _,pred =torch.max(out.data,1)
        correct += (pred==data_y).sum()
    print('the accurary of test is :',float(correct.numpy()/10000)* 100)

