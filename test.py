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
        # 输入为n*1*256*256
        # 输出为n*64*64*64

        self.conv0 = nn.Conv2d(1, 128, 7)

        self.conv1 = nn.Conv2d(128, 128, 7)
        self.conv2 = nn.Conv2d(128, 128, 3)

        self.conv3 = nn.Conv2d(128, 64, 3)

        self.bn128 = nn.BatchNorm2d(128)


    def forward(self, x):

        x = F.leaky_relu(self.bn128(self.conv0(x))) # n*128*256*256

        xA = F.interpolate(x, (128, 128), mode='bilinear')  # n*128*128*128
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.interpolate(x, (128, 128), mode='bilinear') # n*128*128*128
        x = x + xA

        xA = F.interpolate(x, (64, 64), mode='bilinear')  # n*128*64*64
        x = F.leaky_relu(self.bn128(self.conv2(x)))
        x = F.leaky_relu(self.bn128(self.conv2(x)))
        x = F.leaky_relu(self.bn128(self.conv2(x)))
        x = F.interpolate(x, (64, 64), mode='bilinear') # n*128*64*64
        x = x + xA

        x = F.interpolate(F.leaky_relu(self.conv3(x)), (64, 64), mode='bilinear') # n*64*64*64




        return x

class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        # 输入为n*64*64*64
        # 输出为n*1*256*256

        self.conv0 = nn.Conv2d(64, 128, 3)

        self.conv1 = nn.Conv2d(128, 128, 7)
        self.conv2 = nn.Conv2d(128, 128, 3)

        self.conv3 = nn.Conv2d(128, 3, 7)

        self.bn128 = nn.BatchNorm2d(128)


    def forward(self, x):

        x = F.leaky_relu(self.bn128(self.conv0(x))) # n*128*64*64

        xA = F.interpolate(x, (128,128), mode='bilinear') # n*128*128*128
        x = F.leaky_relu(self.bn128(self.conv2(x)))
        x = F.leaky_relu(self.bn128(self.conv2(x)))
        x = F.leaky_relu(self.bn128(self.conv2(x)))
        x = F.interpolate(x, (128,128), mode='bilinear') # n*128*128*128
        x = x + xA

        xA = F.interpolate(x, (256, 256), mode='bilinear')  # n*128*256*256
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.interpolate(x, (256, 256), mode='bilinear') # n*128*256*256
        x = x + xA

        x = F.interpolate(F.leaky_relu(self.conv3(x)), (256, 256), mode='bilinear') # n*1*256*256

        return x


class cNet(nn.Module):
    def __init__(self):
        super(cNet, self).__init__()
        self.enc = EncodeNet()
        self.dec = DecodeNet()

    def forward(self, x):
        y = self.enc(x)
        return self.dec(y)




torch.cuda.set_device(int(sys.argv[1]))
net = torch.load('./models/3.pkl').cuda()

imgNum = os.listdir('./256bmp').__len__()
j = 0
for i in range(0,imgNum,8):
    print(i)
    img = Image.open('./256bmp/' + str(i) + '.bmp').convert('L')
    imgData = numpy.asarray(img).astype(float)
    imgData = imgData / 255  # 归一化到[0, 1]
    imgData = imgData.reshape((1,1,256,256))
    trainData = torch.from_numpy(imgData).float().cuda()
    output = net(trainData)
    newImgData = output.cpu().detach().numpy().transpose(0, 2, 3, 1)
    newImgData = newImgData * 255
    newImgData[newImgData < 0] = 0
    newImgData[newImgData > 255] = 255
    newImgData = newImgData.astype(numpy.uint8)
    img = Image.fromarray(newImgData[0])
    img.save('./newImg/' + str(j) + '.bmp')
    j = j + 1
