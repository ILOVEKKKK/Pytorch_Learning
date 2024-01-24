import torch
from torch import  nn
import torch.nn.functional as F
class Model1(nn.Module):#定义模型的层级结构与运算

    def __init__(self):
        super(Model1,self).__init__()

    def forward(self,input):
        output=input+1
        return output

net=Model1()#要先实例化网络对象
x=torch.tensor(1)#网络中用tensor数据进行运算
x=net(x)
print(x)
