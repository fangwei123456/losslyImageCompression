from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim
import sys
import os

class Quantize(torch.autograd.Function): # 量化函数
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        output = torch.round(input) # 量化

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # input = ctx.saved_tensors[0]
        grad_input = grad_output
        return grad_input


class EncodeNet(nn.Module):
    def __init__(self):
        super(EncodeNet, self).__init__()

        self.conv0 = nn.Conv2d(1, 256, 1)

        self.conv1 = nn.Conv2d(256, 256, 9)

        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)


    def forward(self, x):

        # n*1*256*256 -> n*256*256*256
        x = self.conv0(x)

        # n*256*256*256 -> n*256*240*240
        xA = F.interpolate(x, (240, 240), mode='bilinear')
        xA = self.bn256(xA)

        x = F.leaky_relu(self.conv1(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn256(x)

        x = F.leaky_relu(self.conv1(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn256(x)

        x = x + xA

        return x



class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()


        self.tconv0 = nn.ConvTranspose2d(256, 1, 1)

        self.tconv1 = nn.ConvTranspose2d(256, 256, 9)

        self.bn256 = nn.BatchNorm2d(256)


    def forward(self, x):

        # n*256*240*240 -> n*256*256*256
        xA = F.interpolate(x, (256, 256), mode='bilinear')
        xA = self.bn256(xA)

        x = F.leaky_relu(self.tconv1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn256(x)

        x = F.leaky_relu(self.tconv1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn256(x)

        x = x + xA

        # n*256*256*256 -> n*1*256*256
        x = self.tconv0(x)

        return x


class cNet(nn.Module):
    def __init__(self):
        super(cNet, self).__init__()
        self.enc = EncodeNet()
        self.dec = DecodeNet()

    def forward(self, x):
        y = self.enc(x)
        return self.dec(y)




'''
argv:
1: 使用哪个显卡
2: 为0则重新开始训练 否则读取之前的模型
3: 为0则使用L1Loss 否则使用MSELoss
4: 学习率 Adam默认是1e-3
5: 训练次数
'''
'''
imgNum = os.listdir('/home/nvidia/文档/dataSet/256bmp').__len__()
imgData = numpy.empty([imgNum,1,256,256])

for i in range(imgNum):
    img = Image.open('/home/nvidia/文档/dataSet/256bmp/' + str(i) + '.bmp').convert('L')
    imgData[i] = numpy.asarray(img).astype(float).reshape([1,256,256])


imgData = imgData / 255 # 归一化到[0,1]

net = EncodeNet().cuda()
trainData = torch.from_numpy(imgData[0:1]).float().cuda()
net(trainData)

exit(0)
'''
if(len(sys.argv)==1):
    print('1: 使用哪个显卡\n'
          '2: 为0则重新开始训练 否则读取之前的模型\n'
          '3: 为0则使用L1Loss 否则使用MSELoss\n'
          '4: 学习率 Adam默认是1e-3\n'
          '5: 训练次数')
    exit(0)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡
imgNum = os.listdir('./256bmp').__len__()
imgData = numpy.empty([imgNum,1,256,256])

for i in range(imgNum):
    img = Image.open('./256bmp/' + str(i) + '.bmp').convert('L')
    imgData[i] = numpy.asarray(img).astype(float).reshape([1,256,256])




if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    net = cNet().cuda()
    print('create new model')
else:
    net = torch.load('./models/5.pkl').cuda()
    print('read ./models/5.pkl')

print(net)
if(sys.argv[3]=='0'):
    criterion = nn.L1Loss()
else:
    criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=float(sys.argv[4]))
posL = 0
posR = 0
minLoss = torch.zeros(1).cuda()

for i in range(int(sys.argv[5])):
    posR = min(imgNum, posL+3)
    trainData = torch.from_numpy(imgData[posL:posR]).float().cuda()
    posL = posR
    if(posL==imgNum):
        posL = 0

    optimizer.zero_grad()
    output = net(trainData)
    loss = criterion(output,trainData)
    loss.backward()
    optimizer.step()
    if(i==0):
        minLoss = loss
    else:
        if(loss<minLoss): # 保存最小loss对应的模型
            minLoss = loss
            torch.save(net, './models/5.pkl')
            print('save ./models/5.pkl')
    print(i)
    print(loss, minLoss)










