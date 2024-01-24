import torch
import torchvision
from PIL import Image
from model import *

# data_set=torchvision.datasets.CIFAR10(r"E:\P_project\pytorch_learn\torchvision数据集\dataset",train=False,download=False)
# print("test classifier")
image=Image.open("./pic/ship.png")
image=image.convert("RGB")
data_transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])

image=data_transform(image)

model=torch.load("model_test.pth",map_location="cpu")#gpu上训练的模型在没有gpu显卡的机器上加载运行要加上map_location="cpu"指令

image=torch.reshape(image,(1,3,32,32))
model.eval()#开始验证前最好进行调用
with torch.no_grad():
    output=model(image)
    print(output.argmax(1))
    # print(output)

# print(image.shape)
# print(image)

