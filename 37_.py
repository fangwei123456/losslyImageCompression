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

class EncodeNet_1_128_256_256(nn.Module):
    def __init__(self):
        super(EncodeNet_1_128_256_256, self).__init__()

        self.conv0 = nn.Conv2d(1, 128, 1)

        self.conv1_0 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_1 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_3 = nn.Conv2d(128, 128, 5, padding=2)

        self.bn128_0 = nn.BatchNorm2d(128)
        self.bn128_1 = nn.BatchNorm2d(128)
        self.bn128_2 = nn.BatchNorm2d(128)
        self.bn128_3 = nn.BatchNorm2d(128)

        self.bn128_A_0 = nn.BatchNorm2d(128)
        self.bn128_A_1 = nn.BatchNorm2d(128)


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

        return x

class DecodeNet_128_1_256_256(nn.Module):
    def __init__(self):
        super(DecodeNet_128_1_256_256, self).__init__()

        self.tconv0 = nn.ConvTranspose2d(128, 1, 1)

        self.tconv1_0 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_1 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_2 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_3 = nn.ConvTranspose2d(128, 128, 5, padding=2)

        self.bn128_0 = nn.BatchNorm2d(128)
        self.bn128_1 = nn.BatchNorm2d(128)
        self.bn128_2 = nn.BatchNorm2d(128)
        self.bn128_3 = nn.BatchNorm2d(128)

        self.bn128_A_0 = nn.BatchNorm2d(128)
        self.bn128_A_1 = nn.BatchNorm2d(128)


    def forward(self, x):

        xA = self.bn128_A_1(x)

        x = F.leaky_relu(self.tconv1_3(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_3(x)

        x = F.leaky_relu(self.tconv1_2(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_2(x)

        x = x + xA

        xA = self.bn128_A_0(x)

        x = F.leaky_relu(self.tconv1_1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_1(x)

        x = F.leaky_relu(self.tconv1_0(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_0(x)

        x = x + xA

        # n*128*256*256 -> n*1*256*256
        x = self.tconv0(x)

        return x

class EncodeNet_128_128_256_128(nn.Module):
    def __init__(self):
        super(EncodeNet_128_128_256_128, self).__init__()

        self.conv0 = nn.Conv2d(128, 128, 2, 2)

        self.conv1_0 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_1 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_3 = nn.Conv2d(128, 128, 5, padding=2)

        self.bn128_0 = nn.BatchNorm2d(128)
        self.bn128_1 = nn.BatchNorm2d(128)
        self.bn128_2 = nn.BatchNorm2d(128)
        self.bn128_3 = nn.BatchNorm2d(128)

        self.bn128_A_0 = nn.BatchNorm2d(128)
        self.bn128_A_1 = nn.BatchNorm2d(128)


    def forward(self, x):

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

        return x

class DecodeNet_128_128_128_256(nn.Module):
    def __init__(self):
        super(DecodeNet_128_128_128_256, self).__init__()

        self.tconv0 = nn.ConvTranspose2d(128, 128, 2, 2)

        self.tconv1_0 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_1 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_2 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv1_3 = nn.ConvTranspose2d(128, 128, 5, padding=2)

        self.bn128_0 = nn.BatchNorm2d(128)
        self.bn128_1 = nn.BatchNorm2d(128)
        self.bn128_2 = nn.BatchNorm2d(128)
        self.bn128_3 = nn.BatchNorm2d(128)

        self.bn128_A_0 = nn.BatchNorm2d(128)
        self.bn128_A_1 = nn.BatchNorm2d(128)


    def forward(self, x):

        xA = self.bn128_A_1(x)

        x = F.leaky_relu(self.tconv1_3(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_3(x)

        x = F.leaky_relu(self.tconv1_2(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_2(x)

        x = x + xA

        xA = self.bn128_A_0(x)

        x = F.leaky_relu(self.tconv1_1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_1(x)

        x = F.leaky_relu(self.tconv1_0(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn128_0(x)

        x = x + xA

        x = self.tconv0(x)

        return x

class EncodeNet_128_64_128_128(nn.Module):
    def __init__(self):
        super(EncodeNet_128_64_128_128, self).__init__()

        self.conv0 = nn.Conv2d(128, 64, 1)

        self.conv1_0 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv1_1 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv1_2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv1_3 = nn.Conv2d(64, 64, 5, padding=2)

        self.bn64_0 = nn.BatchNorm2d(64)
        self.bn64_1 = nn.BatchNorm2d(64)
        self.bn64_2 = nn.BatchNorm2d(64)
        self.bn64_3 = nn.BatchNorm2d(64)

        self.bn64_A_0 = nn.BatchNorm2d(64)
        self.bn64_A_1 = nn.BatchNorm2d(64)


    def forward(self, x):

        x = self.conv0(x)

        xA = self.bn64_A_0(x)

        x = F.leaky_relu(self.conv1_0(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_0(x)

        x = F.leaky_relu(self.conv1_1(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_1(x)

        x = x + xA

        xA = self.bn64_A_1(x)

        x = F.leaky_relu(self.conv1_2(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_2(x)

        x = F.leaky_relu(self.conv1_3(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn64_3(x)

        x = x + xA

        return x

class DecodeNet_64_128_128_128(nn.Module):
    def __init__(self):
        super(DecodeNet_64_128_128_128, self).__init__()

        self.tconv0 = nn.ConvTranspose2d(64, 128, 1)

        self.tconv1_0 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv1_1 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv1_2 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv1_3 = nn.ConvTranspose2d(64, 64, 5, padding=2)

        self.bn64_0 = nn.BatchNorm2d(64)
        self.bn64_1 = nn.BatchNorm2d(64)
        self.bn64_2 = nn.BatchNorm2d(64)
        self.bn64_3 = nn.BatchNorm2d(64)

        self.bn64_A_0 = nn.BatchNorm2d(64)
        self.bn64_A_1 = nn.BatchNorm2d(64)


    def forward(self, x):

        xA = self.bn64_A_1(x)

        x = F.leaky_relu(self.tconv1_3(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_3(x)

        x = F.leaky_relu(self.tconv1_2(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_2(x)

        x = x + xA

        xA = self.bn64_A_0(x)

        x = F.leaky_relu(self.tconv1_1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_1(x)

        x = F.leaky_relu(self.tconv1_0(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn64_0(x)

        x = x + xA

        x = self.tconv0(x)

        return x

class EncodeNet_64_32_128_128(nn.Module):
    def __init__(self):
        super(EncodeNet_64_32_128_128, self).__init__()

        self.conv0 = nn.Conv2d(64, 32, 1)

        self.conv1_0 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv1_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv1_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv1_3 = nn.Conv2d(32, 32, 5, padding=2)

        self.bn32_0 = nn.BatchNorm2d(32)
        self.bn32_1 = nn.BatchNorm2d(32)
        self.bn32_2 = nn.BatchNorm2d(32)
        self.bn32_3 = nn.BatchNorm2d(32)

        self.bn32_A_0 = nn.BatchNorm2d(32)
        self.bn32_A_1 = nn.BatchNorm2d(32)


    def forward(self, x):

        x = self.conv0(x)

        xA = self.bn32_A_0(x)

        x = F.leaky_relu(self.conv1_0(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_0(x)

        x = F.leaky_relu(self.conv1_1(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_1(x)

        x = x + xA

        xA = self.bn32_A_1(x)

        x = F.leaky_relu(self.conv1_2(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_2(x)

        x = F.leaky_relu(self.conv1_3(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn32_3(x)

        x = x + xA

        return x

class DecodeNet_32_64_128_128(nn.Module):
    def __init__(self):
        super(DecodeNet_32_64_128_128, self).__init__()

        self.tconv0 = nn.ConvTranspose2d(32, 64, 1)

        self.tconv1_0 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv1_1 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv1_2 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv1_3 = nn.ConvTranspose2d(32, 32, 5, padding=2)

        self.bn32_0 = nn.BatchNorm2d(32)
        self.bn32_1 = nn.BatchNorm2d(32)
        self.bn32_2 = nn.BatchNorm2d(32)
        self.bn32_3 = nn.BatchNorm2d(32)

        self.bn32_A_0 = nn.BatchNorm2d(32)
        self.bn32_A_1 = nn.BatchNorm2d(32)


    def forward(self, x):

        xA = self.bn32_A_1(x)

        x = F.leaky_relu(self.tconv1_3(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_3(x)

        x = F.leaky_relu(self.tconv1_2(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_2(x)

        x = x + xA

        xA = self.bn32_A_0(x)

        x = F.leaky_relu(self.tconv1_1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_1(x)

        x = F.leaky_relu(self.tconv1_0(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn32_0(x)

        x = x + xA

        x = self.tconv0(x)

        return x

class EncodeNet_32_16_128_128(nn.Module):
    def __init__(self):
        super(EncodeNet_32_16_128_128, self).__init__()

        self.conv0 = nn.Conv2d(32, 16, 1)

        self.conv1_0 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv1_1 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv1_2 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv1_3 = nn.Conv2d(16, 16, 5, padding=2)

        self.bn16_0 = nn.BatchNorm2d(16)
        self.bn16_1 = nn.BatchNorm2d(16)
        self.bn16_2 = nn.BatchNorm2d(16)
        self.bn16_3 = nn.BatchNorm2d(16)

        self.bn16_A_0 = nn.BatchNorm2d(16)
        self.bn16_A_1 = nn.BatchNorm2d(16)


    def forward(self, x):

        x = self.conv0(x)

        xA = self.bn16_A_0(x)

        x = F.leaky_relu(self.conv1_0(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn16_0(x)

        x = F.leaky_relu(self.conv1_1(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn16_1(x)

        x = x + xA

        xA = self.bn16_A_1(x)

        x = F.leaky_relu(self.conv1_2(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn16_2(x)

        x = F.leaky_relu(self.conv1_3(x))
        x = x / (torch.norm(x) + 1e-6)
        x = self.bn16_3(x)

        x = x + xA

        return x

class DecodeNet_16_32_128_128(nn.Module):
    def __init__(self):
        super(DecodeNet_16_32_128_128, self).__init__()

        self.tconv0 = nn.ConvTranspose2d(16, 32, 1)

        self.tconv1_0 = nn.ConvTranspose2d(16, 16, 5, padding=2)
        self.tconv1_1 = nn.ConvTranspose2d(16, 16, 5, padding=2)
        self.tconv1_2 = nn.ConvTranspose2d(16, 16, 5, padding=2)
        self.tconv1_3 = nn.ConvTranspose2d(16, 16, 5, padding=2)

        self.bn16_0 = nn.BatchNorm2d(16)
        self.bn16_1 = nn.BatchNorm2d(16)
        self.bn16_2 = nn.BatchNorm2d(16)
        self.bn16_3 = nn.BatchNorm2d(16)

        self.bn16_A_0 = nn.BatchNorm2d(16)
        self.bn16_A_1 = nn.BatchNorm2d(16)


    def forward(self, x):

        xA = self.bn16_A_1(x)

        x = F.leaky_relu(self.tconv1_3(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn16_3(x)

        x = F.leaky_relu(self.tconv1_2(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn16_2(x)

        x = x + xA

        xA = self.bn16_A_0(x)

        x = F.leaky_relu(self.tconv1_1(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn16_1(x)

        x = F.leaky_relu(self.tconv1_0(x))
        x = x * (torch.norm(x) + 1e-6)
        x = self.bn16_0(x)

        x = x + xA

        x = self.tconv0(x)

        return x
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

    '''
    encodeNet0 = EncodeNet_1_128_256_256().cuda()
    decodeNet0 = DecodeNet_128_1_256_256().cuda()
    
    encodeNet1 = EncodeNet_128_128_256_128().cuda()
    decodeNet1 = DecodeNet_128_128_128_256().cuda()

    encodeNet2 = EncodeNet_128_64_128_128().cuda()
    decodeNet2 = DecodeNet_64_128_128_128().cuda()

    encodeNet3 = EncodeNet_64_32_128_128().cuda()
    decodeNet3 = DecodeNet_32_64_128_128().cuda()
    
    encodeNet4 = EncodeNet_32_16_128_128().cuda()
    decodeNet4 = DecodeNet_16_32_128_128().cuda()
    '''

    print('create new model')
else:
    '''
    encodeNet0 = torch.load('./models/encodeNet0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/encodeNet0_' + sys.argv[5] + '.pkl')
    decodeNet0 = torch.load('./models/decodeNet0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/decodeNet0_' + sys.argv[5] + '.pkl')

    encodeNet1 = torch.load('./models/encodeNet1_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/encodeNet1_' + sys.argv[5] + '.pkl')
    decodeNet1 = torch.load('./models/decodeNet1_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/decodeNet1_' + sys.argv[5] + '.pkl')

    encodeNet2 = torch.load('./models/encodeNet2_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/encodeNet2_' + sys.argv[5] + '.pkl')
    decodeNet2 = torch.load('./models/decodeNet2_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/decodeNet2_' + sys.argv[5] + '.pkl')

    encodeNet3 = torch.load('./models/encodeNet3_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/encodeNet3_' + sys.argv[5] + '.pkl')
    decodeNet3 = torch.load('./models/decodeNet3_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/decodeNet3_' + sys.argv[5] + '.pkl')

    encodeNet4 = torch.load('./models/encodeNet4_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/encodeNet4_' + sys.argv[5] + '.pkl')
    decodeNet4 = torch.load('./models/decodeNet4_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/decodeNet4_' + sys.argv[5] + '.pkl')
    '''

    encodeNet0 = torch.load('./models/encodeNet0_0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/encodeNet0_0_' + sys.argv[5] + '.pkl')
    decodeNet0 = torch.load('./models/decodeNet0_0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/decodeNet0_0_' + sys.argv[5] + '.pkl')

    encodeNet1 = torch.load('./models/encodeNet1_0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/encodeNet1_0_' + sys.argv[5] + '.pkl')
    decodeNet1 = torch.load('./models/decodeNet1_0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/decodeNet1_0_' + sys.argv[5] + '.pkl')

    encodeNet2 = torch.load('./models/encodeNet2_0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/encodeNet2_0_' + sys.argv[5] + '.pkl')
    decodeNet2 = torch.load('./models/decodeNet2_0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/decodeNet2_0_' + sys.argv[5] + '.pkl')

    encodeNet3 = torch.load('./models/encodeNet3_0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/encodeNet3_0_' + sys.argv[5] + '.pkl')
    decodeNet3 = torch.load('./models/decodeNet3_0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/decodeNet3_0_' + sys.argv[5] + '.pkl')

    encodeNet4 = torch.load('./models/encodeNet4_0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/encodeNet4_0_' + sys.argv[5] + '.pkl')
    decodeNet4 = torch.load('./models/decodeNet4_0_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/decodeNet4_0_' + sys.argv[5] + '.pkl')


criterion = pytorch_ssim.SSIM()
optimizer = torch.optim.Adam([
    {'params':encodeNet0.parameters()},
    {'params':encodeNet1.parameters()},
    {'params':encodeNet2.parameters()},
    {'params':encodeNet3.parameters()},
    {'params':decodeNet3.parameters()},
    {'params':decodeNet2.parameters()},
    {'params':decodeNet1.parameters()},
    {'params':decodeNet0.parameters()},
    ], lr=float(sys.argv[3]))
batchSize = 6
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
        y0 = encodeNet0(trainData)
        y1 = encodeNet1(y0)
        y2 = encodeNet2(y1)
        y3 = encodeNet3(y2)
        yq = quantize(y3)
        x3 = decodeNet3(yq)
        x2 = decodeNet2(x3)
        x1 = decodeNet1(x2)
        x0 = decodeNet0(x1)

        loss = -criterion(x0, trainData)
        if(loss>maxLossOfTrainData):
            maxLossOfTrainData = loss # 保存所有训练样本中的最大损失

        loss.backward()
        optimizer.step()

    if (i == 0):
        minLoss = maxLossOfTrainData
    else:
        if (maxLossOfTrainData < minLoss):  # 保存最小loss对应的模型
            minLoss = maxLossOfTrainData

            torch.save(encodeNet0, './models/encodeNet0_0_' + sys.argv[5] + '.pkl')
            print('./models/encodeNet0_0_' + sys.argv[5] + '.pkl')
            torch.save(decodeNet0, './models/decodeNet0_0_' + sys.argv[5] + '.pkl')
            print('./models/decodeNet0_0_' + sys.argv[5] + '.pkl')
            torch.save(encodeNet1, './models/encodeNet1_0_' + sys.argv[5] + '.pkl')
            print('./models/encodeNet1_0_' + sys.argv[5] + '.pkl')
            torch.save(decodeNet1, './models/decodeNet1_0_' + sys.argv[5] + '.pkl')
            print('./models/decodeNet1_0_' + sys.argv[5] + '.pkl')
            torch.save(encodeNet2, './models/encodeNet2_0_' + sys.argv[5] + '.pkl')
            print('./models/encodeNet2_0_' + sys.argv[5] + '.pkl')
            torch.save(decodeNet2, './models/decodeNet2_0_' + sys.argv[5] + '.pkl')
            print('./models/decodeNet2_0_' + sys.argv[5] + '.pkl')
            torch.save(encodeNet3, './models/encodeNet3_0_' + sys.argv[5] + '.pkl')
            print('./models/encodeNet3_0_' + sys.argv[5] + '.pkl')
            torch.save(decodeNet3, './models/decodeNet3_0_' + sys.argv[5] + '.pkl')
            print('./models/decodeNet3_0_' + sys.argv[5] + '.pkl')
            torch.save(encodeNet4, './models/encodeNet4_0_' + sys.argv[5] + '.pkl')
            print('./models/encodeNet4_0_' + sys.argv[5] + '.pkl')
            torch.save(decodeNet4, './models/decodeNet4_0_' + sys.argv[5] + '.pkl')
            print('./models/decodeNet4_0_' + sys.argv[5] + '.pkl')

    print(sys.argv)
    print(i)
    print(maxLossOfTrainData)
    print(minLoss)











