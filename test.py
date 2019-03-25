from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim


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
        # 输入为n*3*512*512
        # 输出为n*100*32*32

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
        x = F.interpolate(x, (400,400), mode='bilinear')

        xAvg = self.bnA(self.convA(x))
        xAvg = F.avg_pool2d(xAvg, 3, 1)
        xAvg = F.interpolate(xAvg, (32,32), mode='bilinear')

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, (256,256), mode='bilinear')

        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, (128,128), mode='bilinear')

        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.interpolate(x, (64,64), mode='bilinear')

        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.interpolate(x, (32,32), mode='bilinear')


        return x

class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()
        # 输入为n*100*32*32
        # 输出为n*3*512*512

        self.conv0 = nn.Conv2d(100, 128, 3)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(128, 128, 7)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 15)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 31)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 3, 63)
        self.bn4 = nn.BatchNorm2d(3)

        self.convA = nn.Conv2d(128, 3, 1)
        self.bnA = nn.BatchNorm2d(3)

    def forward(self, x):
        x = F.leaky_relu(self.bn0(self.conv0(x)))
        x = F.interpolate(x, (64,64), mode='bilinear')

        xAvg = self.bnA(self.convA(x))
        xAvg = F.avg_pool2d(xAvg, 3, 1)
        xAvg = F.interpolate(xAvg, (512,512), mode='bilinear')

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, (128,128), mode='bilinear')

        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, (256,256), mode='bilinear')

        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.interpolate(x, (400,400), mode='bilinear')

        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.interpolate(x, (512,512), mode='bilinear')

        return x

class cNet(nn.Module):
    def __init__(self):
        super(cNet, self).__init__()
        self.enc = EncodeNet()
        self.dec = DecodeNet()

    def forward(self, x):
        y = self.enc(x)
        return self.dec(y)



torch.cuda.set_device(11)
net = torch.load('./models/1.pkl').cuda()

imgNum = 26

for i in range(imgNum):
    print(i)
    img = Image.open('./512bmp/' + str(i) + '.bmp')
    imgData = numpy.asarray(img).astype(float)
    imgData = imgData / 255  # 归一化到[0, 1]
    imgData.resize((1,512,512,3))
    imgData = imgData.transpose(0, 3, 1, 2)  # 转换为n*3*512*512
    trainData = torch.from_numpy(imgData).float().cuda()
    output = net(trainData)
    newImgData = output.cpu().detach().numpy().transpose(0, 2, 3, 1)  # 转换为n*512*512*3
    newImgData = (newImgData + 0.5) * 255
    newImgData[newImgData < 0] = 0
    newImgData[newImgData > 255] = 255
    newImgData = newImgData.astype(numpy.uint8)
    img = Image.fromarray(newImgData[0])
    img.save('./newImg/' + str(i) + '.bmp')
