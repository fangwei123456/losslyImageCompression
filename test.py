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

        self.conv0 = nn.Conv2d(1, 128, 1)

        self.conv1 = nn.Conv2d(128, 128, 9)

        self.bn128 = nn.BatchNorm2d(128)


    def forward(self, x):

        # n*1*256*256 -> n*128*256*256
        x = self.conv0(x)

        # n*128*256*256 -> n*128*240*240
        xA = F.interpolate(x, (240, 240), mode='bilinear')
        xA = self.bn128(xA)

        x = F.leaky_relu(self.conv1(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn128(x)

        x = F.leaky_relu(self.conv1(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn128(x)

        x = x + xA

        return x



class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        self.tconv0 = nn.ConvTranspose2d(128, 1, 1)

        self.tconv1 = nn.ConvTranspose2d(128, 128, 9)

        self.bn128 = nn.BatchNorm2d(128)


    def forward(self, x):

        # n*128*240*240 -> n*128*256*256
        xA = F.interpolate(x, (256, 256), mode='bilinear')
        xA = self.bn128(xA)

        x = F.leaky_relu(self.tconv1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128(x)

        x = F.leaky_relu(self.tconv1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128(x)

        x = x + xA

        # n*128*256*256 -> n*1*256*256
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

class expDecayLoss(nn.Module):
    def __init__(self):
        super(expDecayLoss, self).__init__()
    def forward(self, x, y, lam):
        # lam介于[0,1]
        mse = torch.pow(x - y, 2);
        decay = torch.pow(torch.ones_like(mse)/2, lam) # 按1/2的lam次衰减
        mse = torch.mean(mse.mul(decay))
        return mse



torch.cuda.set_device(int(sys.argv[1]))
net = torch.load('./models/7.pkl').cuda()
imgNum = os.listdir('./256bmp').__len__()
j = 0
for i in range(0,imgNum,8):
    print(i)
    img = Image.open('./256bmp/' + str(i) + '.bmp').convert('L')
    imgData = numpy.asarray(img).astype(float)
    imgData = imgData.reshape((1,1,256,256))
    trainData = torch.from_numpy(imgData).float().cuda()
    output = net(trainData)
    newImgData = output.cpu().detach().numpy().reshape([256,256])
    newImgData[newImgData < 0] = 0
    newImgData[newImgData > 255] = 255
    newImgData = newImgData.astype(numpy.uint8)
    img = Image.fromarray(newImgData)
    img.save('./newImg/' + str(j) + '.bmp')
    j = j + 1
