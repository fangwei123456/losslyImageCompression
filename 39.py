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

# 导入信息熵损失
from torch.utils.cpp_extension import load
entropy_loss_cuda = load(
    'entropy_loss_cuda', ['./pytorch_entropy_loss/entropy_loss_cuda.cpp', './pytorch_entropy_loss/entropy_loss_cuda_kernel.cu'], verbose=True)
help(entropy_loss_cuda)
from pytorch_entropy_loss.entropy_loss import EL

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

        self.conv1_0 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_1 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_3 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_4 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_5 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_6 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_7 = nn.Conv2d(128, 128, 5, padding=2)

        self.conv1 = nn.Conv2d(128, 64, 1) # 降通道数
        self.conv2 = nn.Conv2d(64, 32, 1) # 降通道数

        self.conv2_0 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_1 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_3 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_4 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_5 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_6 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_7 = nn.Conv2d(64, 64, 5, padding=2)

        self.conv3_0 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3_3 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3_4 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3_5 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3_6 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3_7 = nn.Conv2d(32, 32, 5, padding=2)

        self.conv_down_0 = nn.Conv2d(128, 128, 2, 2) # 降采样

        self.bn128_0 = nn.BatchNorm2d(128)
        self.bn128_1 = nn.BatchNorm2d(128)
        self.bn128_2 = nn.BatchNorm2d(128)
        self.bn128_3 = nn.BatchNorm2d(128)
        self.bn128_4 = nn.BatchNorm2d(128)
        self.bn128_5 = nn.BatchNorm2d(128)
        self.bn128_6 = nn.BatchNorm2d(128)
        self.bn128_7 = nn.BatchNorm2d(128)

        self.bn64_0 = nn.BatchNorm2d(64)
        self.bn64_1 = nn.BatchNorm2d(64)
        self.bn64_2 = nn.BatchNorm2d(64)
        self.bn64_3 = nn.BatchNorm2d(64)
        self.bn64_4 = nn.BatchNorm2d(64)
        self.bn64_5 = nn.BatchNorm2d(64)
        self.bn64_6 = nn.BatchNorm2d(64)
        self.bn64_7 = nn.BatchNorm2d(64)

        self.bn32_0 = nn.BatchNorm2d(32)
        self.bn32_1 = nn.BatchNorm2d(32)
        self.bn32_2 = nn.BatchNorm2d(32)
        self.bn32_3 = nn.BatchNorm2d(32)
        self.bn32_4 = nn.BatchNorm2d(32)
        self.bn32_5 = nn.BatchNorm2d(32)
        self.bn32_6 = nn.BatchNorm2d(32)
        self.bn32_7 = nn.BatchNorm2d(32)

        self.bn128_A_0 = nn.BatchNorm2d(128)
        self.bn128_A_1 = nn.BatchNorm2d(128)
        self.bn128_A_2 = nn.BatchNorm2d(128)
        self.bn128_A_3 = nn.BatchNorm2d(128)

        self.bn64_A_0 = nn.BatchNorm2d(64)
        self.bn64_A_1 = nn.BatchNorm2d(64)
        self.bn64_A_2 = nn.BatchNorm2d(64)
        self.bn64_A_3 = nn.BatchNorm2d(64)

        self.bn32_A_0 = nn.BatchNorm2d(32)
        self.bn32_A_1 = nn.BatchNorm2d(32)
        self.bn32_A_2 = nn.BatchNorm2d(32)
        self.bn32_A_3 = nn.BatchNorm2d(32)



    def forward(self, x):

        # n*1*256*256 -> n*128*256*256
        x = self.conv0(x)

        xA = self.bn128_A_0(x)

        x = F.leaky_relu(self.conv1_0(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn128_0(x)

        x = F.leaky_relu(self.conv1_1(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn128_1(x)

        x = x + xA

        xA = self.bn128_A_1(x)

        x = F.leaky_relu(self.conv1_2(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn128_2(x)

        x = F.leaky_relu(self.conv1_3(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn128_3(x)

        x = x + xA

        # n*128*256*256 -> n*128*128*128
        x = self.conv_down_0(x)

        xA = self.bn128_A_2(x)

        x = F.leaky_relu(self.conv1_4(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn128_4(x)

        x = F.leaky_relu(self.conv1_5(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn128_5(x)

        x = x + xA

        xA = self.bn128_A_3(x)

        x = F.leaky_relu(self.conv1_6(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn128_6(x)

        x = F.leaky_relu(self.conv1_7(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn128_7(x)

        x = x + xA

        # n*128*128*128 -> n*64*128*128
        x = self.conv1(x)

        xA = self.bn64_A_0(x)

        x = F.leaky_relu(self.conv2_0(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_0(x)

        x = F.leaky_relu(self.conv2_1(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_1(x)

        x = x + xA

        xA = self.bn64_A_1(x)

        x = F.leaky_relu(self.conv2_2(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_2(x)

        x = F.leaky_relu(self.conv2_3(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_3(x)

        x = x + xA

        xA = self.bn64_A_2(x)

        x = F.leaky_relu(self.conv2_4(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_4(x)

        x = F.leaky_relu(self.conv2_5(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_5(x)

        x = x + xA

        xA = self.bn64_A_3(x)

        x = F.leaky_relu(self.conv2_6(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_6(x)

        x = F.leaky_relu(self.conv2_7(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_7(x)

        x = x + xA

        # n*64*128*128 -> n*32*128*128
        x = self.conv2(x)

        xA = self.bn32_A_0(x)

        x = F.leaky_relu(self.conv3_0(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_0(x)

        x = F.leaky_relu(self.conv3_1(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_1(x)

        x = x + xA

        xA = self.bn32_A_1(x)

        x = F.leaky_relu(self.conv3_2(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_2(x)

        x = F.leaky_relu(self.conv3_3(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_3(x)

        x = x + xA

        xA = self.bn32_A_2(x)

        x = F.leaky_relu(self.conv3_4(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_4(x)

        x = F.leaky_relu(self.conv3_5(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_5(x)

        x = x + xA

        xA = self.bn32_A_3(x)

        x = F.leaky_relu(self.conv3_6(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_6(x)

        x = F.leaky_relu(self.conv3_7(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_7(x)

        x = x + xA

        return x # n*32*32*128

class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        self.tconv0 = nn.ConvTranspose2d(128, 1, 1)
        self.tconv1 = nn.ConvTranspose2d(64, 128, 1)
        self.tconv2 = nn.ConvTranspose2d(32, 64, 1)

        self.tconv1_0 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_1 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_2 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_3 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_4 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_5 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_6 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_7 = nn.ConvTranspose2d(128, 128, 5, padding=2)

        self.tconv2_0 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv2_1 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv2_2 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv2_3 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv2_4 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv2_5 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv2_6 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv2_7 = nn.ConvTranspose2d(64, 64, 5, padding=2)

        self.tconv3_0 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv3_1 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv3_2 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv3_3 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv3_4 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv3_5 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv3_6 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv3_7 = nn.ConvTranspose2d(32, 32, 5, padding=2)

        self.tconv_up_0 = nn.ConvTranspose2d(128, 128, 2, 2) # 升采样

        self.bn128_0 = nn.BatchNorm2d(128)
        self.bn128_1 = nn.BatchNorm2d(128)
        self.bn128_2 = nn.BatchNorm2d(128)
        self.bn128_3 = nn.BatchNorm2d(128)
        self.bn128_4 = nn.BatchNorm2d(128)
        self.bn128_5 = nn.BatchNorm2d(128)
        self.bn128_6 = nn.BatchNorm2d(128)
        self.bn128_7 = nn.BatchNorm2d(128)

        self.bn64_0 = nn.BatchNorm2d(64)
        self.bn64_1 = nn.BatchNorm2d(64)
        self.bn64_2 = nn.BatchNorm2d(64)
        self.bn64_3 = nn.BatchNorm2d(64)
        self.bn64_4 = nn.BatchNorm2d(64)
        self.bn64_5 = nn.BatchNorm2d(64)
        self.bn64_6 = nn.BatchNorm2d(64)
        self.bn64_7 = nn.BatchNorm2d(64)

        self.bn32_0 = nn.BatchNorm2d(32)
        self.bn32_1 = nn.BatchNorm2d(32)
        self.bn32_2 = nn.BatchNorm2d(32)
        self.bn32_3 = nn.BatchNorm2d(32)
        self.bn32_4 = nn.BatchNorm2d(32)
        self.bn32_5 = nn.BatchNorm2d(32)
        self.bn32_6 = nn.BatchNorm2d(32)
        self.bn32_7 = nn.BatchNorm2d(32)

        self.bn128_A_0 = nn.BatchNorm2d(128)
        self.bn128_A_1 = nn.BatchNorm2d(128)
        self.bn128_A_2 = nn.BatchNorm2d(128)
        self.bn128_A_3 = nn.BatchNorm2d(128)

        self.bn64_A_0 = nn.BatchNorm2d(64)
        self.bn64_A_1 = nn.BatchNorm2d(64)
        self.bn64_A_2 = nn.BatchNorm2d(64)
        self.bn64_A_3 = nn.BatchNorm2d(64)

        self.bn32_A_0 = nn.BatchNorm2d(32)
        self.bn32_A_1 = nn.BatchNorm2d(32)
        self.bn32_A_2 = nn.BatchNorm2d(32)
        self.bn32_A_3 = nn.BatchNorm2d(32)




    def forward(self, x):

        xA = self.bn32_A_0(x)

        x = F.leaky_relu(self.tconv3_0(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_0(x)

        x = F.leaky_relu(self.tconv3_1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_1(x)

        x = x + xA

        xA = self.bn32_A_1(x)

        x = F.leaky_relu(self.tconv3_2(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_2(x)

        x = F.leaky_relu(self.tconv3_3(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_3(x)

        x = x + xA

        xA = self.bn32_A_2(x)

        x = F.leaky_relu(self.tconv3_4(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_4(x)

        x = F.leaky_relu(self.tconv3_5(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_5(x)

        x = x + xA

        xA = self.bn32_A_3(x)

        x = F.leaky_relu(self.tconv3_6(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_6(x)

        x = F.leaky_relu(self.tconv3_7(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_7(x)

        x = x + xA

        # n*32*128*128 -> n*64*128*128
        x = self.tconv2(x)

        xA = self.bn64_A_0(x)

        x = F.leaky_relu(self.tconv2_0(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_0(x)

        x = F.leaky_relu(self.tconv2_1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_1(x)

        x = x + xA

        xA = self.bn64_A_1(x)

        x = F.leaky_relu(self.tconv2_2(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_2(x)

        x = F.leaky_relu(self.tconv2_3(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_3(x)

        x = x + xA

        xA = self.bn64_A_2(x)

        x = F.leaky_relu(self.tconv2_4(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_4(x)

        x = F.leaky_relu(self.tconv2_5(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_5(x)

        x = x + xA

        xA = self.bn64_A_3(x)

        x = F.leaky_relu(self.tconv2_6(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_6(x)

        x = F.leaky_relu(self.tconv2_7(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_7(x)

        x = x + xA

        # n*64*128*128 -> n*128*128*128
        x = self.tconv1(x)

        xA = self.bn128_A_0(x)

        x = F.leaky_relu(self.tconv1_0(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_0(x)

        x = F.leaky_relu(self.tconv1_1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_1(x)

        x = x + xA

        xA = self.bn128_A_1(x)

        x = F.leaky_relu(self.tconv1_2(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_2(x)

        x = F.leaky_relu(self.tconv1_3(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_3(x)

        x = x + xA

        # n*128*128*128 -> n*128*256*256
        x = self.tconv_up_0(x)

        xA = self.bn128_A_2(x)

        x = F.leaky_relu(self.tconv1_4(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_4(x)

        x = F.leaky_relu(self.tconv1_5(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_5(x)

        x = x + xA

        xA = self.bn128_A_3(x)

        x = F.leaky_relu(self.tconv1_6(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_6(x)

        x = F.leaky_relu(self.tconv1_7(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_7(x)

        x = x + xA

        # n*128*256*256 -> n*1*256*256
        x = self.tconv0(x)

        return x









'''
argv:
1: 使用哪个显卡
2: 为0则重新开始训练 否则读取之前的模型
3: 学习率 Adam默认是1e-3
4: 训练次数
5: 保存的模型标号
6: λ 训练目标是最小化loss = -λ*SSIM + EL
   增大λ 则训练目标向质量方向偏移
'''

if(len(sys.argv)!=6):
    print('1: 使用哪个显卡\n'
          '2: 为0则重新开始训练 否则读取之前的模型\n'
          '3: 学习率 Adam默认是1e-3\n'
          '4: 训练次数\n'
          '5: 保存的模型标号\n'
          '6: lambda')
    exit(0)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡
imgNum = os.listdir('./256bmp').__len__()
imgData = numpy.empty([imgNum,1,256,256])
laplacianData = numpy.empty([imgNum,1,256,256])
for i in range(imgNum):
    img = Image.open('./256bmp/' + str(i) + '.bmp').convert('L')
    imgData[i] = numpy.asarray(img).astype(float).reshape([1,256,256])

    img = Image.open('./256bmpLaplacian/' + str(i) + '.bmp').convert('L')
    laplacianData[i] = numpy.asarray(img).astype(float).reshape([1,256,256])


laplacianData = laplacianData / 255 # 归一化到[0,1]



if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    encNet = EncodeNet().cuda()
    decNet = DecodeNet().cuda()
    print('create new model')
else:
    encNet = torch.load('./models/encNet_' + sys.argv[5] + '.pkl').cuda()
    decNet = torch.load('./models/decNet_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/' + sys.argv[5] + '.pkl')

print(encNet)
print(decNet)


SSIMLoss = pytorch_ssim.SSIM()
ELoss = EL.apply
ssimLambda =  float(sys.argv[6])

optimizer = torch.optim.Adam([{'params':encNet.parameters()},{'params':decNet.parameters()}], lr=float(sys.argv[3]))

batchSize = 4 # 一次读取?张图片进行训练
imgData = torch.from_numpy(imgData).float().cuda()
trainData = torch.empty([batchSize, 1, 256, 256]).float().cuda()



for i in range(int(sys.argv[4])):

    readSeq = torch.randperm(imgNum) # 生成读取的随机序列

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
        encData = encNet(trainData)
        qEncData = quantize(encData)
        decData = decNet(qEncData)

        currentSL = SSIMLoss(decData, trainData)
        currentEL = ELoss(encData)
        loss = -ssimLambda *currentSL + currentEL
        print(currentSL,currentEL,loss)
        print(i,j,k)

        if(j==0):
            maxLossOfTrainData = loss
        else:
            if(loss>maxLossOfTrainData):
                maxLossOfTrainData = loss # 保存所有训练样本中的最大损失

        loss.backward()
        optimizer.step()

    if (i == 0):
        minLoss = loss
    else:
        if (loss < minLoss):  # 保存最小loss对应的模型
            minLoss = loss
            torch.save(encNet, './models/encNet_' + sys.argv[5] + '.pkl')
            torch.save(decNet, './models/decNet_' + sys.argv[5] + '.pkl')
            print('save ./models/' + sys.argv[5] + '.pkl')

    print(sys.argv)
    print(i)
    print(loss)
    print(minLoss)











