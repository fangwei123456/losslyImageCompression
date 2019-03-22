from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

class encodeNet(nn.Module):
    def __init__(self):
        super(encodeNet, self).__init__()
        # 输入为n*512*512*3

        self.conv1 = nn.Conv2d(3, 128, 63)
        self.conv2 = nn.Conv2d(128, 128, 31)
        self.conv3 = nn.Conv2d(128, 128, 15)
        self.conv4 = nn.Conv2d(128, 128, 7)

    def forward(self, x):
        positiveX = x
        negativeX = -x

        positiveX = F.max_pool2d(F.leaky_relu(self.conv1(positiveX)), (2, 2))
        positiveX = F.max_pool2d(F.leaky_relu(self.conv2(positiveX)), (2, 2))
        positiveX = F.max_pool2d(F.leaky_relu(self.conv3(positiveX)), (2, 2))
        positiveX = F.max_pool2d(F.leaky_relu(self.conv4(positiveX)), (2, 2))

        negativeX = F.max_pool2d(F.leaky_relu(self.conv1(negativeX)), (2, 2))
        negativeX = F.max_pool2d(F.leaky_relu(self.conv2(negativeX)), (2, 2))
        negativeX = F.max_pool2d(F.leaky_relu(self.conv3(negativeX)), (2, 2))
        negativeX = F.max_pool2d(F.leaky_relu(self.conv4(negativeX)), (2, 2))

        x = torch.abs(positiveX) + torch.abs(negativeX)

        return x


# 读取所有图片 存入trainData
imgNum = 25
imgData = numpy.empty([imgNum,512,512,3])

for i in range(imgNum):
    img = Image.open('./512bmp/' + str(i) + '.bmp')
    imgData[i] = numpy.asarray(img).astype(float)

imgData = imgData / 255 - 0.5 # 归一化到[-0.5,0.5]
imgData = imgData.transpose(0,3,1,2) # 转换为n*3*512*512
trainData = torch.from_numpy(imgData).float()
print(trainData.shape,trainData.dtype)





