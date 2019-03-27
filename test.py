from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim
import sys
import os
import train


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
