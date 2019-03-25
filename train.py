from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim
import sys
import os
import time

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
        # 输出为n*64*64*64

        self.conv0 = nn.Conv2d(3, 128, 7)

        self.conv1 = nn.Conv2d(128, 128, 7)
        self.conv2 = nn.Conv2d(128, 128, 3)

        self.conv3 = nn.Conv2d(128, 64, 3)

        self.bn128 = nn.BatchNorm2d(128)


    def forward(self, x):

        x = F.leaky_relu(self.bn128(self.conv0(x))) # n*128*512*512

        xA = F.interpolate(x, (256, 256), mode='bilinear')  # n*128*256*256
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.interpolate(x, (256, 256), mode='bilinear') # n*128*256*256
        x = x + xA

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
        # 输出为n*3*512*512

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

        xA = F.interpolate(x, (512, 512), mode='bilinear')  # n*128*512*512
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.leaky_relu(self.bn128(self.conv1(x)))
        x = F.interpolate(x, (512, 512), mode='bilinear') # n*128*512*512
        x = x + xA

        x = F.interpolate(F.leaky_relu(self.conv3(x)), (512, 512), mode='bilinear') # n*3*512*512

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
3: 学习率 Adam默认是1e-3
'''
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡
imgNum = os.listdir('./512bmp').__len__()
imgData = numpy.empty([imgNum,512,512,3])

for i in range(imgNum):
    img = Image.open('./512bmp/' + str(i) + '.bmp')
    imgData[i] = numpy.asarray(img).astype(float)

imgData = imgData / 255 # 归一化到[0,1]
imgData = imgData.transpose(0,3,1,2) # 转换为n*3*512*512

if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    net = cNet().cuda()
    print('create new model')
else:
    net = torch.load('./models/2.pkl').cuda()
    print('read ./models/2.pkl')

print(net)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=float(sys.argv[3]))
posL = 0
posR = 0
minLoss = torch.zeros(1).cuda()

for i in range(1000):
    tStart = time.time()
    posR = min(imgNum, posL+2)
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
            torch.save(net, './models/2.pkl')
            print('save ./models/2.pkl')
    print(i,time.time() - tStart,loss)
'''
newImgData = output.cpu().detach().numpy().transpose(0,2,3,1) # 转换为n*512*512*3
newImgData = newImgData*255
newImgData[newImgData<0] = 0
newImgData[newImgData>255] = 255
newImgData = newImgData.astype(numpy.uint8)
for i in range(imgNum):

    img = Image.fromarray(newImgData[i])
    img.save('./newImg/' + str(i) + '.bmp')

'''










