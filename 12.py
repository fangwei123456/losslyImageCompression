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

        self.conv1_0 = nn.Conv2d(128, 128, 3)

        self.bn128_0 = nn.BatchNorm2d(128)
        self.bn128_1 = nn.BatchNorm2d(128)



    def forward(self, x):

        # n*1*256*256 -> n*128*256*256
        x = self.conv0(x)

        # n*128*256*256 -> n*128*254*254
        xA = F.interpolate(x, (254, 254), mode='bilinear')
        xA = self.bn128_0(xA)

        x = F.leaky_relu(self.conv1_0(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn128_1(x)


        x = x + xA

        return x



class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        self.tconv0 = nn.ConvTranspose2d(128, 1, 1)

        self.tconv1_0 = nn.ConvTranspose2d(128, 128, 3)

        self.bn128_0 = nn.BatchNorm2d(128)
        self.bn128_1 = nn.BatchNorm2d(128)


    def forward(self, x):

        # n*128*254*254 -> n*128*256*256
        xA = F.interpolate(x, (256, 256), mode='bilinear')
        xA = self.bn128_0(xA)

        x = F.leaky_relu(self.tconv1_0(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_1(x)


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
        decay = torch.pow(torch.ones_like(mse)/16, lam) # 按1/16的lam次衰减
        mse = torch.mean(mse.mul(decay))
        return mse






'''
argv:
1: 使用哪个显卡
2: 为0则重新开始训练 否则读取之前的模型
3: 为0则使用L1Loss 为1使用自定义loss
4: 学习率 Adam默认是1e-3
5: 训练次数
6: 保存的模型标号
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
if(len(sys.argv)!=7):
    print('1: 使用哪个显卡\n'
          '2: 为0则重新开始训练 否则读取之前的模型\n'
          '3: 为0则使用MSELoss 为1使用自定义loss\n'
          '4: 学习率 Adam默认是1e-3\n'
          '5: 训练次数\n'
          '6: 保存的模型标号')
    exit(0)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡
imgNum = os.listdir('./256bmp').__len__()
imgData = numpy.empty([imgNum,1,256,256])
laplacianData = numpy.empty([imgNum,1,256,256])
for i in range(imgNum):
    img = Image.open('./256bmp/' + str(i) + '.bmp').convert('L')
    imgData[i] = numpy.asarray(img).astype(float).reshape([1,256,256])
    if(sys.argv[3] == '1'):
        img = Image.open('./256bmpLaplacian/' + str(i) + '.bmp').convert('L')
        laplacianData[i] = numpy.asarray(img).astype(float).reshape([1,256,256])

if(sys.argv[3] == '1'):
    laplacianData = laplacianData / 255 # 归一化到[0,1]



if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    net = cNet().cuda()
    print('create new model')
else:
    net = torch.load('./models/' + sys.argv[6] + '.pkl').cuda()
    print('read ./models/' + sys.argv[6] + '.pkl')

print(net)
if(sys.argv[3]=='0'):
    criterion = nn.MSELoss()
elif(sys.argv[3]=='1'):
    criterion = expDecayLoss()
    mseCriterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=float(sys.argv[4]))
posL = 0
posR = 0
minMseLoss = torch.zeros(1).cuda()


for i in range(int(sys.argv[5])):
    posR = min(imgNum, posL+3)
    trainData = torch.from_numpy(imgData[posL:posR]).float().cuda()
    if(sys.argv[3] == '1'):
        laplacianMat = torch.from_numpy(laplacianData[posL:posR]).float().cuda()

    posL = posR
    if(posL==imgNum):
        posL = 0

    optimizer.zero_grad()
    output = net(trainData)
    # laplacianMat需要归一化到[0,1]
    # laplacianMat中 灰度变化剧烈（边缘）的地方 值也大
    # 这些地方的像素值可以适当衰减 因为人眼其实难以察觉
    # 因此这些地方的loss可以适当减小
    if(sys.argv[3] == '0'):
        loss = criterion(output,trainData)
    elif(sys.argv[3] == '1'):
        loss = criterion(output, trainData, laplacianMat)
        mseLoss = mseCriterion(output, trainData)
    loss.backward()
    optimizer.step()
    if(i==0):
        minMseLoss = mseLoss
    else:
        if(mseLoss<minMseLoss): # 保存最小loss对应的模型
            minMseLoss = mseLoss
            torch.save(net, './models/' + sys.argv[6] + '.pkl')
            print('save ./models/' + sys.argv[6] + '.pkl')
    print(sys.argv)
    print(i)
    print('loss=',loss)
    print('mseLoss=',mseLoss)
    print('minMseLoss=',minMseLoss)










