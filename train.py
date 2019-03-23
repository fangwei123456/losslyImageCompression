from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim

class quantize(torch.autograd.Function): # 量化函数
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        output = torch.round(input) # 量化

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # input = ctx.saved_tensors[0]
        grad_input = None
        if(ctx.needs_input_grad[0]):
            grad_input = grad_output
        return grad_input


class encodeNet(nn.Module):
    def __init__(self):
        super(encodeNet, self).__init__()
        # 输入为n*512*512*3

        self.conv0 = nn.Conv2d(3, 128, 63)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(128, 128, 31)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 15)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 7)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 100, 3)
        self.bn4 = nn.BatchNorm2d(100)

        self.convA = nn.Conv2d(128, 100, 1)
        self.bnA = nn.BatchNorm2d(100)

    def forward(self, x):

        x = F.leaky_relu(self.bn0(self.conv0(x)))
        x = F.interpolate(x, scale_factor=0.6, mode='bilinear') # n*128*270*270

        xAvg = self.bnA(self.convA(x))
        xAvg = F.avg_pool2d(xAvg, 3, 1) # n*100*268*268
        xAvg = F.interpolate(xAvg, (24,24), mode='bilinear') # n*100*24*24

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=0.6, mode='bilinear')  # n*128*144*144

        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, scale_factor=0.6, mode='bilinear')  # n*128*78*78

        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.interpolate(x, scale_factor=0.6, mode='bilinear')  # n*128*33*43

        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.interpolate(x, scale_factor=0.6, mode='bilinear')  # n*100*24*24

        x = x + xAvg

        x = (input + 0.5) * 255

        return x















# 读取所有图片 存入trainData
imgNum = 25
imgData = numpy.empty([imgNum,512,512,3])

for i in range(imgNum):
    img = Image.open('./512bmp/' + str(i) + '.bmp')
    imgData[i] = numpy.asarray(img).astype(float)

imgData = imgData / 255  - 0.5 # 归一化到[-0.5,0.5]
imgData = imgData.transpose(0,3,1,2) # 转换为n*3*512*512
trainData = torch.from_numpy(imgData).float()
print(trainData.shape,trainData.dtype)

encoder = encodeNet()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(encoder.parameters(), lr=0.001, momentum=0.9)
running_loss = 0.0

for i in range(10):
    optimizer.zero_grad()
    outputs = encoder(trainData)
    loss = criterion(outputs, trainData)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print(running_loss)





