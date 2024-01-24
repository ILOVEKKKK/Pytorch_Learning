import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
writer=SummaryWriter("logs")

dataset=torchvision.datasets.CIFAR10("../torchvision数据集/dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)
data_loader=DataLoader(dataset,batch_size=1)
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

loss=nn.CrossEntropyLoss()
net=my_net()
step=0

for data in data_loader:
    imgs,labels=data
    outputs=net(imgs)
    result_loss=loss(outputs,labels)
    result_loss.backward()
    print(result_loss)



writer.close()

#
# inputs=torch.tensor([1,2,3],dtype=torch.float32)
# targets=torch.tensor([1,2,5],dtype=torch.float32)
#
# loss1=nn.L1Loss(reduction='mean')#L1范数误差
# loss2=nn.MSELoss(reduction='mean')#均方误差
# loss3=nn.CrossEntropyLoss()#交叉熵损失函数，用于评价分类问题

