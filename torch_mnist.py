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
#print(train_loader.dataset)
#print(test_loader.dataset)


# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.L1=nn.Linear(784,300)
        self.L2 = nn.Linear(300,100)
        self.L3= nn.Linear(100,50)
        self.out = nn.Linear(50,10)

    def forward(self, x):
        x=x.view(-1,28*28)
        x=F.relu(self.L1(x))
        x=F.relu(self.L2(x))
        x=F.relu(self.L3(x))
        x=self.out(x)
        return x
# 第二步完成

if __name__ == '__main__':
    #net =torch.load('net.pkl')
    net=Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    # 定义损失函数
    loss_fun = torch.nn.CrossEntropyLoss()
    # 进行训练，对训练集拆分并训练两次
    for epoch in range(15):
        running_loss = 0
        correct = 0
        for step, (data_x, data_y) in enumerate(train_loader):
            out = net(data_x)
            _, pred = torch.max(out.data, 1)
            correct += (pred == data_y).sum()
            loss = loss_fun(out, data_y)
            running_loss += loss.data.item()
            # 进行更新的操作
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(step)
        print('epoch', epoch, 'loss', running_loss / 60000, 'correct', float(correct.numpy() / 60000) * 100)
    #第三部分结束     end
    correct =0
    #对测试集进行测试
    for  i,(data_x,data_y)in enumerate(test_loader):
        out=net(data_x)
        _,pred =torch.max(out.data,1)
        correct += (pred==data_y).sum()
    print('the accurary of test is :',float(correct.numpy()/10000)* 100)
    torch.save(net,'net.pkl')








