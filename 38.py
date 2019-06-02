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
import pytorch_gdn
class Quantize(torch.autograd.Function): # 量化函数
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        output = torch.round(input) # 量化

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # input = ctx.saved_tensors[0]
        return grad_output # 把量化器的导数当做1

def quantize(input):
    return Quantize.apply(input)

class EncodeNet(nn.Module):
    def __init__(self, device):
        super(EncodeNet, self).__init__()

        self.conv0 = nn.Conv2d(1, 128, 9, padding=4)

        self.conv_down_0 = nn.Conv2d(128, 128, 4, 4)  # 降采样

        self.gdn128_0 = pytorch_gdn.GDN(128, device)

        self.conv1 = nn.Conv2d(128, 128, 5, padding=2)

        self.conv_down_1 = nn.Conv2d(128, 128, 2, 2)  # 降采样

        self.gdn128_1 = pytorch_gdn.GDN(128, device)

        self.conv2 = nn.Conv2d(128, 128, 5, padding=2)

        self.conv_down_2 = nn.Conv2d(128, 128, 2, 2)  # 降采样

        self.gdn128_2 = pytorch_gdn.GDN(128, device)

    def forward(self, x):

        x = self.conv0(x)

        x = self.conv_down_0(x)

        x = self.gdn128_0(x)

        x = self.conv1(x)

        x = self.conv_down_1(x)

        x = self.gdn128_1(x)

        x = self.conv2(x)

        x = self.conv_down_2(x)

        x = self.gdn128_2(x)

        return x

class DecodeNet(nn.Module):
    def __init__(self, device):
        super(DecodeNet, self).__init__()

        self.tconv0 = nn.ConvTranspose2d(128, 1, 9, padding=4)

        self.tconv_up_0 = nn.ConvTranspose2d(128, 128, 4, 4)  # 升采样

        self.igdn128_0 = pytorch_gdn.GDN(128, device, True)

        self.tconv1 = nn.ConvTranspose2d(128, 128, 5, padding=2)

        self.tconv_up_1 = nn.ConvTranspose2d(128, 128, 2, 2)  # 升采样

        self.igdn128_1 = pytorch_gdn.GDN(128, device, True)

        self.tconv2 = nn.ConvTranspose2d(128, 128, 5, padding=2)

        self.tconv_up_2 = nn.ConvTranspose2d(128, 128, 2, 2)  # 升采样

        self.igdn128_2 = pytorch_gdn.GDN(128, device, True)

    def forward(self, x):

        x = self.igdn128_2(x)

        x = self.tconv_up_2(x)

        x = self.tconv2(x)

        x = self.igdn128_1(x)

        x = self.tconv_up_1(x)

        x = self.tconv1(x)

        x = self.igdn128_0(x)

        x = self.tconv_up_0(x)

        x = self.tconv0(x)

        return x

class cNet(nn.Module):
    def __init__(self, device):
        super(cNet, self).__init__()
        self.enc = EncodeNet(device)
        self.dec = DecodeNet(device)

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
if(len(sys.argv)!=6):
    print('1: 使用哪个显卡\n'
          '2: 为0则重新开始训练 否则读取之前的模型\n'
          '3: 学习率 Adam默认是1e-3\n'
          '4: 训练次数\n'
          '5: 保存的模型标号')
    exit(0)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡
device = torch.device('cuda')

imgNum = os.listdir('./256bmp').__len__()
imgData = numpy.empty([imgNum,1,256,256])

for i in range(imgNum):
    img = Image.open('./256bmp/' + str(i) + '.bmp').convert('L')
    imgData[i] = numpy.asarray(img).astype(float).reshape([1,256,256])






if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    net = cNet(device).cuda()
    print('create new model')
else:
    net = torch.load('./models/' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/' + sys.argv[5] + '.pkl')

print(net)


criterion = pytorch_ssim.SSIM()
optimizer = torch.optim.Adam(net.parameters(), lr=float(sys.argv[3]))
batchSize = 16 # 一次读取?张图片进行训练
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
        if (loss > maxLossOfTrainData):
            maxLossOfTrainData = loss  # 保存所有训练样本中的最大损失

        loss.backward()
        optimizer.step()

    if (i == 0):
        minLoss = maxLossOfTrainData
    else:
        if (maxLossOfTrainData < minLoss):  # 保存最小loss对应的模型
            minLoss = maxLossOfTrainData
            torch.save(net, './models/' + sys.argv[5] + '.pkl')
            print('save ./models/' + sys.argv[5] + '.pkl')

    print(sys.argv)
    print(i)
    print(loss)
    print(minLoss)











