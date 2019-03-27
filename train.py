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
        self.bn64 = nn.BatchNorm2d(64)




    def forward(self, x):
        x = self.conv0(x)
        x = F.leaky_relu(x / (torch.norm(x) + 1e-3)) # n*128*256*256

        xA = F.interpolate(x, (128, 128), mode='bilinear')  # n*128*128*128
        x = self.conv1(x)
        x = F.leaky_relu(x / (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = self.conv1(x)
        x = F.leaky_relu(x / (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = self.conv1(x)
        x = F.leaky_relu(x / (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = F.interpolate(x, (128, 128), mode='bilinear') # n*128*128*128
        x = x + xA

        xA = F.interpolate(x, (64, 64), mode='bilinear')  # n*128*64*64
        x = self.conv2(x)
        x = F.leaky_relu(x / (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = self.conv2(x)
        x = F.leaky_relu(x / (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = self.conv2(x)
        x = F.leaky_relu(x / (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = F.interpolate(x, (64, 64), mode='bilinear') # n*128*64*64
        x = x + xA

        x = self.conv3(x)
        x = F.leaky_relu(x / (torch.norm(x) + 1e-3))
        x = self.bn64(x)
        x = F.interpolate(x, (64, 64), mode='bilinear') # n*64*64*64

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
        self.bn3 = nn.BatchNorm2d(3)


    def forward(self, x):
        x = self.conv0(x)
        x = F.leaky_relu(x * (torch.norm(x) + 1e-3)) # n*128*64*64

        xA = F.interpolate(x, (128,128), mode='bilinear') # n*128*128*128
        x = self.conv2(x)
        x = F.leaky_relu(x * (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = self.conv2(x)
        x = F.leaky_relu(x * (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = self.conv2(x)
        x = F.leaky_relu(x * (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = F.interpolate(x, (128,128), mode='bilinear') # n*128*128*128
        x = x + xA

        xA = F.interpolate(x, (256, 256), mode='bilinear')  # n*128*256*256
        x = self.conv1(x)
        x = F.leaky_relu(x * (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = self.conv1(x)
        x = F.leaky_relu(x * (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = self.conv1(x)
        x = F.leaky_relu(x * (torch.norm(x) + 1e-3))
        x = self.bn128(x)
        x = F.interpolate(x, (256, 256), mode='bilinear') # n*128*256*256
        x = x + xA

        x = self.conv3(x)
        x = F.leaky_relu(x / (torch.norm(x) + 1e-3))
        x = self.bn3(x)
        x = F.interpolate(x, (256, 256), mode='bilinear') # n*3*256*256
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


imgData = imgData / 255 # 归一化到[0,1]


if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    net = cNet().cuda()
    print('create new model')
else:
    net = torch.load('./models/3.pkl').cuda()
    print('read ./models/3.pkl')

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
    posR = min(imgNum, posL+8)
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
            torch.save(net, './models/3.pkl')
            print('save ./models/3.pkl')
    print(i)
    print(loss, minLoss)
'''
newImgData = output.cpu().detach().numpy().transpose(0,2,3,1)
newImgData = newImgData*255
newImgData[newImgData<0] = 0
newImgData[newImgData>255] = 255
newImgData = newImgData.astype(numpy.uint8)
for i in range(imgNum):

    img = Image.fromarray(newImgData[i])
    img.save('./newImg/' + str(i) + '.bmp')

'''










