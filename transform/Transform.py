from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
img_path= "../dataset/train/ants/0013035.jpg"
img=Image.open(img_path)

writer=SummaryWriter("../logs")

tensor_trans=transforms.ToTensor()#实例化一个ToTensor()对象
tensor_image=tensor_trans(img)
writer.add_image("img",tensor_image)
writer.close()


