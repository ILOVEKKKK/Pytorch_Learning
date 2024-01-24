import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
writer=SummaryWriter("logs")

dataset=torchvision.datasets.CIFAR10("../torchvision数据集/dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)
data_loader=DataLoader(dataset,batch_size=1)#加载数据集


class my_net(nn.Module):
    def __init__(self):
        super(my_net,self).__init__()
        self.conv1=nn.Conv2d(3,32,5,1,padding=2)
        self.max_pool1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(32,32,5,padding=2)
        self.max_pool2=nn.MaxPool2d(2)
        self.conv3=nn.Conv2d(32,64,5,padding=2)
        self.max_pool3=nn.MaxPool2d(2)
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(1024,64)
        self.linear2=nn.Linear(64,10)
        self.model1=nn.Sequential(#简化代码结构
            self.conv1,
            self.max_pool1,
            self.conv2,
            self.max_pool2,
            self.conv3,
            self.max_pool3,
            self.flatten,
            self.linear1,
            self.linear2
        )

    def forward(self,input):
        output=self.model1(input)
        return output

net=my_net()
loss1=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.01)

for epoch in range(20):#训练迭代
    running_loss=0.0#每次迭代的损失函数是在整体数据集上所有loss的和，所以每次训练迭代前要先清零
    for data in data_loader:
        imgs,labels=data
        outputs=net(imgs)
        result_loss=loss1(outputs,labels)
        optimizer.zero_grad()
        result_loss.backward()#进行梯度反向传播以供后面一步进行修正
        optimizer.step()#开始执行优化器进行优化
        running_loss=result_loss+running_loss
    print(running_loss)

