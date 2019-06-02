from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim
import sys
import os
import pytorch_ssim

class BasicalResNet(nn.Module):
    def __init__(self, channels):
        super(BasicalResNet, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv6 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv7 = nn.Conv2d(channels, channels, 3, padding=1)

        self.bn0 = nn.BatchNorm2d(channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(channels)
        self.bn5 = nn.BatchNorm2d(channels)
        self.bn6 = nn.BatchNorm2d(channels)
        self.bn7 = nn.BatchNorm2d(channels)

        self.bnA0 = nn.BatchNorm2d(channels)
        self.bnA1 = nn.BatchNorm2d(channels)
        self.bnA2 = nn.BatchNorm2d(channels)
        self.bnA3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        xA = self.bnA0(x)

        x = F.leaky_relu(self.conv0(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn0(x)

        x = F.leaky_relu(self.conv1(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn1(x)

        x = x + xA

        xA = self.bnA1(x)

        x = F.leaky_relu(self.conv2(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn2(x)

        x = F.leaky_relu(self.conv3(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn3(x)

        xA = self.bnA2(x)

        x = F.leaky_relu(self.conv4(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn4(x)

        x = F.leaky_relu(self.conv5(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn5(x)

        xA = self.bnA3(x)

        x = F.leaky_relu(self.conv6(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn6(x)

        x = F.leaky_relu(self.conv7(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn7(x)

        x = x + xA

        return x

class BasicalTResNet(nn.Module):
    def __init__(self, channels):
        super(BasicalTResNet, self).__init__()
        self.tconv0 = nn.ConvTranspose2d(channels, channels, 3, padding=1)
        self.tconv1 = nn.ConvTranspose2d(channels, channels, 3, padding=1)
        self.tconv2 = nn.ConvTranspose2d(channels, channels, 3, padding=1)
        self.tconv3 = nn.ConvTranspose2d(channels, channels, 3, padding=1)
        self.tconv4 = nn.ConvTranspose2d(channels, channels, 3, padding=1)
        self.tconv5 = nn.ConvTranspose2d(channels, channels, 3, padding=1)
        self.tconv6 = nn.ConvTranspose2d(channels, channels, 3, padding=1)
        self.tconv7 = nn.ConvTranspose2d(channels, channels, 3, padding=1)

        self.bn0 = nn.BatchNorm2d(channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(channels)
        self.bn5 = nn.BatchNorm2d(channels)
        self.bn6 = nn.BatchNorm2d(channels)
        self.bn7 = nn.BatchNorm2d(channels)

        self.bnA0 = nn.BatchNorm2d(channels)
        self.bnA1 = nn.BatchNorm2d(channels)
        self.bnA2 = nn.BatchNorm2d(channels)
        self.bnA3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        xA = self.bnA0(x)

        x = F.leaky_relu(self.tconv0(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn0(x)

        x = F.leaky_relu(self.tconv1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn1(x)

        x = x + xA

        xA = self.bnA1(x)

        x = F.leaky_relu(self.tconv2(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn2(x)

        x = F.leaky_relu(self.tconv3(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn3(x)

        xA = self.bnA2(x)

        x = F.leaky_relu(self.tconv4(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn4(x)

        x = F.leaky_relu(self.tconv5(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn5(x)

        xA = self.bnA3(x)

        x = F.leaky_relu(self.tconv6(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn6(x)

        x = F.leaky_relu(self.tconv7(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn7(x)

        x = x + xA

        return x

class Quantize(torch.autograd.Function): # 量化函数
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input) # 量化

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output # 把量化器的导数当做1

def quantize(input):
    return Quantize.apply(input)

class EncodeNet(nn.Module):
    def __init__(self):
        super(EncodeNet, self).__init__()

        self.conv0 = nn.Conv2d(1, 128, 1)
        self.conv0_0 = BasicalResNet(128)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv0_0(x)
        return x


class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()
        self.tconv0_0 = BasicalTResNet(128)
        self.tconv0 = nn.ConvTranspose2d(128, 1, 1)




    def forward(self, x):
        x = self.tconv0_0(x)
        x = self.tconv0(x)

        return x


class cNet(nn.Module):
    def __init__(self):
        super(cNet, self).__init__()
        self.enc = EncodeNet()
        self.dec = DecodeNet()

    def forward(self, x):
        y = self.enc(x)
        qy = quantize(y)
        return self.dec(qy)





'''
argv:
1: 使用哪个显卡
2: 为0则重新开始训练 否则读取之前的模型
3: 学习率 Adam默认是1e-3
4: 训练次数
5: 保存的模型标号

'''
if(len(sys.argv)!=6):
    print('1: 使用哪个显卡\n'
          '2: 为0则重新开始训练 否则读取之前的模型\n'
          '3: 学习率 Adam默认是1e-3\n'
          '4: 训练次数\n'
          '5: 保存的模型标号')
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
    net = torch.load('./models/' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/' + sys.argv[5] + '.pkl')

print(net)


criterion = pytorch_ssim.SSIM()
optimizer = torch.optim.Adam(net.parameters(), lr=float(sys.argv[3]))
batchSize = 2 # 一次读取?张图片进行训练
imgData = torch.from_numpy(imgData).float().cuda()
trainData = torch.empty([batchSize, 1, 256, 256]).float().cuda()
for i in range(int(sys.argv[4])):

    readSeq = torch.randperm(imgNum) # 生成读取的随机序列
    maxLossOfTrainData = -torch.ones(1).cuda()
    j = 0

    while(1):
        if(j==imgNum):
            break
        k = 0
        while(1):
            trainData[k] = imgData[readSeq[j]]
            k = k + 1
            j = j + 1
            if(k==batchSize or j==imgNum):
                break

        optimizer.zero_grad()
        output = net(trainData)

        loss = -criterion(output, trainData) # 极小化损失 所以要加个负号
        if(loss>maxLossOfTrainData):
            maxLossOfTrainData = loss # 保存所有训练样本中的最大损失

        loss.backward()
        optimizer.step()

    if (i == 0):
        minLoss = loss
    else:
        if (loss < minLoss):  # 保存最小loss对应的模型
            minLoss = loss
            torch.save(net, './models/' + sys.argv[5] + '.pkl')
            print('save ./models/' + sys.argv[5] + '.pkl')

    print(sys.argv)
    print(i)
    print(loss)
    print(minLoss)











