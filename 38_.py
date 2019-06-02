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

        self.conv0 = nn.Conv2d(1, 128, 1)

        self.conv1_0 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_1 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_3 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_4 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_5 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_6 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv1_7 = nn.Conv2d(128, 128, 5, padding=2)

        self.conv1 = nn.Conv2d(128, 64, 1)  # 降通道数
        self.conv2 = nn.Conv2d(64, 32, 1)  # 降通道数
        self.conv3 = nn.Conv2d(32, 16, 1)  # 降通道数
        self.conv4 = nn.Conv2d(16, 8, 1)  # 降通道数
        self.conv5 = nn.Conv2d(8, 4, 1)  # 降通道数

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

        self.conv4_0 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4_1 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4_2 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4_3 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4_4 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4_5 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4_6 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4_7 = nn.Conv2d(16, 16, 5, padding=2)

        self.conv5_0 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv5_1 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv5_2 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv5_3 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv5_4 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv5_5 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv5_6 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv5_7 = nn.Conv2d(8, 8, 5, padding=2)

        self.conv6_0 = nn.Conv2d(4, 4, 5, padding=2)
        self.conv6_1 = nn.Conv2d(4, 4, 5, padding=2)
        self.conv6_2 = nn.Conv2d(4, 4, 5, padding=2)
        self.conv6_3 = nn.Conv2d(4, 4, 5, padding=2)
        self.conv6_4 = nn.Conv2d(4, 4, 5, padding=2)
        self.conv6_5 = nn.Conv2d(4, 4, 5, padding=2)
        self.conv6_6 = nn.Conv2d(4, 4, 5, padding=2)
        self.conv6_7 = nn.Conv2d(4, 4, 5, padding=2)

        self.conv_down_0 = nn.Conv2d(128, 128, 2, 2)  # 降采样

        self.gdn128_0 = pytorch_gdn.GDN(128, device)
        self.gdn128_1 = pytorch_gdn.GDN(128, device)
        self.gdn128_2 = pytorch_gdn.GDN(128, device)
        self.gdn128_3 = pytorch_gdn.GDN(128, device)
        self.gdn128_4 = pytorch_gdn.GDN(128, device)
        self.gdn128_5 = pytorch_gdn.GDN(128, device)
        self.gdn128_6 = pytorch_gdn.GDN(128, device)
        self.gdn128_7 = pytorch_gdn.GDN(128, device)

        self.gdn64_0 = pytorch_gdn.GDN(64, device)
        self.gdn64_1 = pytorch_gdn.GDN(64, device)
        self.gdn64_2 = pytorch_gdn.GDN(64, device)
        self.gdn64_3 = pytorch_gdn.GDN(64, device)
        self.gdn64_4 = pytorch_gdn.GDN(64, device)
        self.gdn64_5 = pytorch_gdn.GDN(64, device)
        self.gdn64_6 = pytorch_gdn.GDN(64, device)
        self.gdn64_7 = pytorch_gdn.GDN(64, device)

        self.gdn32_0 = pytorch_gdn.GDN(32, device)
        self.gdn32_1 = pytorch_gdn.GDN(32, device)
        self.gdn32_2 = pytorch_gdn.GDN(32, device)
        self.gdn32_3 = pytorch_gdn.GDN(32, device)
        self.gdn32_4 = pytorch_gdn.GDN(32, device)
        self.gdn32_5 = pytorch_gdn.GDN(32, device)
        self.gdn32_6 = pytorch_gdn.GDN(32, device)
        self.gdn32_7 = pytorch_gdn.GDN(32, device)

        self.gdn16_0 = pytorch_gdn.GDN(16, device)
        self.gdn16_1 = pytorch_gdn.GDN(16, device)
        self.gdn16_2 = pytorch_gdn.GDN(16, device)
        self.gdn16_3 = pytorch_gdn.GDN(16, device)
        self.gdn16_4 = pytorch_gdn.GDN(16, device)
        self.gdn16_5 = pytorch_gdn.GDN(16, device)
        self.gdn16_6 = pytorch_gdn.GDN(16, device)
        self.gdn16_7 = pytorch_gdn.GDN(16, device)

        self.gdn8_0 = pytorch_gdn.GDN(8, device)
        self.gdn8_1 = pytorch_gdn.GDN(8, device)
        self.gdn8_2 = pytorch_gdn.GDN(8, device)
        self.gdn8_3 = pytorch_gdn.GDN(8, device)
        self.gdn8_4 = pytorch_gdn.GDN(8, device)
        self.gdn8_5 = pytorch_gdn.GDN(8, device)
        self.gdn8_6 = pytorch_gdn.GDN(8, device)
        self.gdn8_7 = pytorch_gdn.GDN(8, device)

        self.gdn4_0 = pytorch_gdn.GDN(4, device)
        self.gdn4_1 = pytorch_gdn.GDN(4, device)
        self.gdn4_2 = pytorch_gdn.GDN(4, device)
        self.gdn4_3 = pytorch_gdn.GDN(4, device)
        self.gdn4_4 = pytorch_gdn.GDN(4, device)
        self.gdn4_5 = pytorch_gdn.GDN(4, device)
        self.gdn4_6 = pytorch_gdn.GDN(4, device)
        self.gdn4_7 = pytorch_gdn.GDN(4, device)

        self.gdn128_A_0 = pytorch_gdn.GDN(128, device)
        self.gdn128_A_1 = pytorch_gdn.GDN(128, device)
        self.gdn128_A_2 = pytorch_gdn.GDN(128, device)
        self.gdn128_A_3 = pytorch_gdn.GDN(128, device)

        self.gdn64_A_0 = pytorch_gdn.GDN(64, device)
        self.gdn64_A_1 = pytorch_gdn.GDN(64, device)
        self.gdn64_A_2 = pytorch_gdn.GDN(64, device)
        self.gdn64_A_3 = pytorch_gdn.GDN(64, device)

        self.gdn32_A_0 = pytorch_gdn.GDN(32, device)
        self.gdn32_A_1 = pytorch_gdn.GDN(32, device)
        self.gdn32_A_2 = pytorch_gdn.GDN(32, device)
        self.gdn32_A_3 = pytorch_gdn.GDN(32, device)

        self.gdn16_A_0 = pytorch_gdn.GDN(16, device)
        self.gdn16_A_1 = pytorch_gdn.GDN(16, device)
        self.gdn16_A_2 = pytorch_gdn.GDN(16, device)
        self.gdn16_A_3 = pytorch_gdn.GDN(16, device)

        self.gdn8_A_0 = pytorch_gdn.GDN(8, device)
        self.gdn8_A_1 = pytorch_gdn.GDN(8, device)
        self.gdn8_A_2 = pytorch_gdn.GDN(8, device)
        self.gdn8_A_3 = pytorch_gdn.GDN(8, device)

        self.gdn4_A_0 = pytorch_gdn.GDN(4, device)
        self.gdn4_A_1 = pytorch_gdn.GDN(4, device)
        self.gdn4_A_2 = pytorch_gdn.GDN(4, device)
        self.gdn4_A_3 = pytorch_gdn.GDN(4, device)

    def forward(self, x):
        # n*1*256*256 -> n*128*256*256
        x = self.conv0(x)

        xA = self.gdn128_A_0(x)

        x = F.leaky_relu(self.conv1_0(x))
        
        x = self.gdn128_0(x)

        x = F.leaky_relu(self.conv1_1(x))
        
        x = self.gdn128_1(x)

        x = x + xA

        xA = self.gdn128_A_1(x)

        x = F.leaky_relu(self.conv1_2(x))
        
        x = self.gdn128_2(x)

        x = F.leaky_relu(self.conv1_3(x))
        
        x = self.gdn128_3(x)

        x = x + xA

        # n*128*256*256 -> n*128*128*128
        x = self.conv_down_0(x)

        xA = self.gdn128_A_2(x)

        x = F.leaky_relu(self.conv1_4(x))
        
        x = self.gdn128_4(x)

        x = F.leaky_relu(self.conv1_5(x))
        
        x = self.gdn128_5(x)

        x = x + xA

        xA = self.gdn128_A_3(x)

        x = F.leaky_relu(self.conv1_6(x))
        
        x = self.gdn128_6(x)

        x = F.leaky_relu(self.conv1_7(x))
        
        x = self.gdn128_7(x)

        x = x + xA

        # n*128*128*128 -> n*64*128*128
        x = self.conv1(x)

        xA = self.gdn64_A_0(x)

        x = F.leaky_relu(self.conv2_0(x))
        
        x = self.gdn64_0(x)

        x = F.leaky_relu(self.conv2_1(x))
        
        x = self.gdn64_1(x)

        x = x + xA

        xA = self.gdn64_A_1(x)

        x = F.leaky_relu(self.conv2_2(x))
        
        x = self.gdn64_2(x)

        x = F.leaky_relu(self.conv2_3(x))
        
        x = self.gdn64_3(x)

        x = x + xA

        xA = self.gdn64_A_2(x)

        x = F.leaky_relu(self.conv2_4(x))
        
        x = self.gdn64_4(x)

        x = F.leaky_relu(self.conv2_5(x))
        
        x = self.gdn64_5(x)

        x = x + xA

        xA = self.gdn64_A_3(x)

        x = F.leaky_relu(self.conv2_6(x))
        
        x = self.gdn64_6(x)

        x = F.leaky_relu(self.conv2_7(x))
        
        x = self.gdn64_7(x)

        x = x + xA

        # n*64*128*128 -> n*32*128*128
        x = self.conv2(x)

        xA = self.gdn32_A_0(x)

        x = F.leaky_relu(self.conv3_0(x))
        
        x = self.gdn32_0(x)

        x = F.leaky_relu(self.conv3_1(x))
        
        x = self.gdn32_1(x)

        x = x + xA

        xA = self.gdn32_A_1(x)

        x = F.leaky_relu(self.conv3_2(x))
        
        x = self.gdn32_2(x)

        x = F.leaky_relu(self.conv3_3(x))
        
        x = self.gdn32_3(x)

        x = x + xA

        xA = self.gdn32_A_2(x)

        x = F.leaky_relu(self.conv3_4(x))
        
        x = self.gdn32_4(x)

        x = F.leaky_relu(self.conv3_5(x))
        
        x = self.gdn32_5(x)

        x = x + xA

        xA = self.gdn32_A_3(x)

        x = F.leaky_relu(self.conv3_6(x))
        
        x = self.gdn32_6(x)

        x = F.leaky_relu(self.conv3_7(x))
        
        x = self.gdn32_7(x)

        x = x + xA

        # n*32*128*128 -> n*16*128*128
        x = self.conv3(x)

        xA = self.gdn16_A_0(x)

        x = F.leaky_relu(self.conv4_0(x))
        
        x = self.gdn16_0(x)

        x = F.leaky_relu(self.conv4_1(x))
        
        x = self.gdn16_1(x)

        x = x + xA

        xA = self.gdn16_A_1(x)

        x = F.leaky_relu(self.conv4_2(x))
        
        x = self.gdn16_2(x)

        x = F.leaky_relu(self.conv4_3(x))
        
        x = self.gdn16_3(x)

        x = x + xA

        xA = self.gdn16_A_2(x)

        x = F.leaky_relu(self.conv4_4(x))
        
        x = self.gdn16_4(x)

        x = F.leaky_relu(self.conv4_5(x))
        
        x = self.gdn16_5(x)

        x = x + xA

        xA = self.gdn16_A_3(x)

        x = F.leaky_relu(self.conv4_6(x))
        
        x = self.gdn16_6(x)

        x = F.leaky_relu(self.conv4_7(x))
        
        x = self.gdn16_7(x)

        x = x + xA

        # n*16*128*128 -> n*8*128*128
        x = self.conv4(x)

        xA = self.gdn8_A_0(x)

        x = F.leaky_relu(self.conv5_0(x))
        
        x = self.gdn8_0(x)

        x = F.leaky_relu(self.conv5_1(x))
        
        x = self.gdn8_1(x)

        x = x + xA

        xA = self.gdn8_A_1(x)

        x = F.leaky_relu(self.conv5_2(x))
        
        x = self.gdn8_2(x)

        x = F.leaky_relu(self.conv5_3(x))
        
        x = self.gdn8_3(x)

        x = x + xA

        xA = self.gdn8_A_2(x)

        x = F.leaky_relu(self.conv5_4(x))
        
        x = self.gdn8_4(x)

        x = F.leaky_relu(self.conv5_5(x))
        
        x = self.gdn8_5(x)

        x = x + xA

        xA = self.gdn8_A_3(x)

        x = F.leaky_relu(self.conv5_6(x))
        
        x = self.gdn8_6(x)

        x = F.leaky_relu(self.conv5_7(x))
        
        x = self.gdn8_7(x)

        x = x + xA

        # n*8*128*128 -> n*4*128*128
        x = self.conv5(x)

        xA = self.gdn4_A_0(x)

        x = F.leaky_relu(self.conv6_0(x))
        
        x = self.gdn4_0(x)

        x = F.leaky_relu(self.conv6_1(x))
        
        x = self.gdn4_1(x)

        x = x + xA

        xA = self.gdn4_A_1(x)

        x = F.leaky_relu(self.conv6_2(x))
        
        x = self.gdn4_2(x)

        x = F.leaky_relu(self.conv6_3(x))
        
        x = self.gdn4_3(x)

        x = x + xA

        xA = self.gdn4_A_2(x)

        x = F.leaky_relu(self.conv6_4(x))
        
        x = self.gdn4_4(x)

        x = F.leaky_relu(self.conv6_5(x))
        
        x = self.gdn4_5(x)

        x = x + xA

        xA = self.gdn4_A_3(x)

        x = F.leaky_relu(self.conv6_6(x))
        
        x = self.gdn4_6(x)

        x = F.leaky_relu(self.conv6_7(x))
        
        x = self.gdn4_7(x)

        x = x + xA

        return x

class DecodeNet(nn.Module):
    def __init__(self, device):
        super(DecodeNet, self).__init__()

        self.tconv0 = nn.ConvTranspose2d(128, 1, 1)
        self.tconv1 = nn.ConvTranspose2d(64, 128, 1)
        self.tconv2 = nn.ConvTranspose2d(32, 64, 1)
        self.tconv3 = nn.ConvTranspose2d(16, 32, 1)
        self.tconv4 = nn.ConvTranspose2d(8, 16, 1)
        self.tconv5 = nn.ConvTranspose2d(4, 8, 1)

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

        self.tconv4_0 = nn.ConvTranspose2d(16, 16, 5, padding=2)
        self.tconv4_1 = nn.ConvTranspose2d(16, 16, 5, padding=2)
        self.tconv4_2 = nn.ConvTranspose2d(16, 16, 5, padding=2)
        self.tconv4_3 = nn.ConvTranspose2d(16, 16, 5, padding=2)
        self.tconv4_4 = nn.ConvTranspose2d(16, 16, 5, padding=2)
        self.tconv4_5 = nn.ConvTranspose2d(16, 16, 5, padding=2)
        self.tconv4_6 = nn.ConvTranspose2d(16, 16, 5, padding=2)
        self.tconv4_7 = nn.ConvTranspose2d(16, 16, 5, padding=2)

        self.tconv5_0 = nn.ConvTranspose2d(8, 8, 5, padding=2)
        self.tconv5_1 = nn.ConvTranspose2d(8, 8, 5, padding=2)
        self.tconv5_2 = nn.ConvTranspose2d(8, 8, 5, padding=2)
        self.tconv5_3 = nn.ConvTranspose2d(8, 8, 5, padding=2)
        self.tconv5_4 = nn.ConvTranspose2d(8, 8, 5, padding=2)
        self.tconv5_5 = nn.ConvTranspose2d(8, 8, 5, padding=2)
        self.tconv5_6 = nn.ConvTranspose2d(8, 8, 5, padding=2)
        self.tconv5_7 = nn.ConvTranspose2d(8, 8, 5, padding=2)

        self.tconv6_0 = nn.ConvTranspose2d(4, 4, 5, padding=2)
        self.tconv6_1 = nn.ConvTranspose2d(4, 4, 5, padding=2)
        self.tconv6_2 = nn.ConvTranspose2d(4, 4, 5, padding=2)
        self.tconv6_3 = nn.ConvTranspose2d(4, 4, 5, padding=2)
        self.tconv6_4 = nn.ConvTranspose2d(4, 4, 5, padding=2)
        self.tconv6_5 = nn.ConvTranspose2d(4, 4, 5, padding=2)
        self.tconv6_6 = nn.ConvTranspose2d(4, 4, 5, padding=2)
        self.tconv6_7 = nn.ConvTranspose2d(4, 4, 5, padding=2)

        self.tconv_up_0 = nn.ConvTranspose2d(128, 128, 2, 2) # 升采样

        self.igdn128_0 = pytorch_gdn.GDN(128, device, True)
        self.igdn128_1 = pytorch_gdn.GDN(128, device, True)
        self.igdn128_2 = pytorch_gdn.GDN(128, device, True)
        self.igdn128_3 = pytorch_gdn.GDN(128, device, True)
        self.igdn128_4 = pytorch_gdn.GDN(128, device, True)
        self.igdn128_5 = pytorch_gdn.GDN(128, device, True)
        self.igdn128_6 = pytorch_gdn.GDN(128, device, True)
        self.igdn128_7 = pytorch_gdn.GDN(128, device, True)

        self.igdn64_0 = pytorch_gdn.GDN(64, device, True)
        self.igdn64_1 = pytorch_gdn.GDN(64, device, True)
        self.igdn64_2 = pytorch_gdn.GDN(64, device, True)
        self.igdn64_3 = pytorch_gdn.GDN(64, device, True)
        self.igdn64_4 = pytorch_gdn.GDN(64, device, True)
        self.igdn64_5 = pytorch_gdn.GDN(64, device, True)
        self.igdn64_6 = pytorch_gdn.GDN(64, device, True)
        self.igdn64_7 = pytorch_gdn.GDN(64, device, True)

        self.igdn32_0 = pytorch_gdn.GDN(32, device, True)
        self.igdn32_1 = pytorch_gdn.GDN(32, device, True)
        self.igdn32_2 = pytorch_gdn.GDN(32, device, True)
        self.igdn32_3 = pytorch_gdn.GDN(32, device, True)
        self.igdn32_4 = pytorch_gdn.GDN(32, device, True)
        self.igdn32_5 = pytorch_gdn.GDN(32, device, True)
        self.igdn32_6 = pytorch_gdn.GDN(32, device, True)
        self.igdn32_7 = pytorch_gdn.GDN(32, device, True)

        self.igdn16_0 = pytorch_gdn.GDN(16, device, True)
        self.igdn16_1 = pytorch_gdn.GDN(16, device, True)
        self.igdn16_2 = pytorch_gdn.GDN(16, device, True)
        self.igdn16_3 = pytorch_gdn.GDN(16, device, True)
        self.igdn16_4 = pytorch_gdn.GDN(16, device, True)
        self.igdn16_5 = pytorch_gdn.GDN(16, device, True)
        self.igdn16_6 = pytorch_gdn.GDN(16, device, True)
        self.igdn16_7 = pytorch_gdn.GDN(16, device, True)

        self.igdn8_0 = pytorch_gdn.GDN(8, device, True)
        self.igdn8_1 = pytorch_gdn.GDN(8, device, True)
        self.igdn8_2 = pytorch_gdn.GDN(8, device, True)
        self.igdn8_3 = pytorch_gdn.GDN(8, device, True)
        self.igdn8_4 = pytorch_gdn.GDN(8, device, True)
        self.igdn8_5 = pytorch_gdn.GDN(8, device, True)
        self.igdn8_6 = pytorch_gdn.GDN(8, device, True)
        self.igdn8_7 = pytorch_gdn.GDN(8, device, True)

        self.igdn4_0 = pytorch_gdn.GDN(4, device, True)
        self.igdn4_1 = pytorch_gdn.GDN(4, device, True)
        self.igdn4_2 = pytorch_gdn.GDN(4, device, True)
        self.igdn4_3 = pytorch_gdn.GDN(4, device, True)
        self.igdn4_4 = pytorch_gdn.GDN(4, device, True)
        self.igdn4_5 = pytorch_gdn.GDN(4, device, True)
        self.igdn4_6 = pytorch_gdn.GDN(4, device, True)
        self.igdn4_7 = pytorch_gdn.GDN(4, device, True)


        self.igdn128_A_0 = pytorch_gdn.GDN(128, device, True)
        self.igdn128_A_1 = pytorch_gdn.GDN(128, device, True)
        self.igdn128_A_2 = pytorch_gdn.GDN(128, device, True)
        self.igdn128_A_3 = pytorch_gdn.GDN(128, device, True)

        self.igdn64_A_0 = pytorch_gdn.GDN(64, device, True)
        self.igdn64_A_1 = pytorch_gdn.GDN(64, device, True)
        self.igdn64_A_2 = pytorch_gdn.GDN(64, device, True)
        self.igdn64_A_3 = pytorch_gdn.GDN(64, device, True)

        self.igdn32_A_0 = pytorch_gdn.GDN(32, device, True)
        self.igdn32_A_1 = pytorch_gdn.GDN(32, device, True)
        self.igdn32_A_2 = pytorch_gdn.GDN(32, device, True)
        self.igdn32_A_3 = pytorch_gdn.GDN(32, device, True)

        self.igdn16_A_0 = pytorch_gdn.GDN(16, device, True)
        self.igdn16_A_1 = pytorch_gdn.GDN(16, device, True)
        self.igdn16_A_2 = pytorch_gdn.GDN(16, device, True)
        self.igdn16_A_3 = pytorch_gdn.GDN(16, device, True)

        self.igdn8_A_0 = pytorch_gdn.GDN(8, device, True)
        self.igdn8_A_1 = pytorch_gdn.GDN(8, device, True)
        self.igdn8_A_2 = pytorch_gdn.GDN(8, device, True)
        self.igdn8_A_3 = pytorch_gdn.GDN(8, device, True)

        self.igdn4_A_0 = pytorch_gdn.GDN(4, device, True)
        self.igdn4_A_1 = pytorch_gdn.GDN(4, device, True)
        self.igdn4_A_2 = pytorch_gdn.GDN(4, device, True)
        self.igdn4_A_3 = pytorch_gdn.GDN(4, device, True)




    def forward(self, x):
        xA = self.igdn4_A_0(x)

        x = F.leaky_relu(self.tconv6_0(x))
        
        x = self.igdn4_0(x)

        x = F.leaky_relu(self.tconv6_1(x))
        
        x = self.igdn4_1(x)

        x = x + xA

        xA = self.igdn4_A_1(x)

        x = F.leaky_relu(self.tconv6_2(x))
        
        x = self.igdn4_2(x)

        x = F.leaky_relu(self.tconv6_3(x))
        
        x = self.igdn4_3(x)

        x = x + xA

        xA = self.igdn4_A_2(x)

        x = F.leaky_relu(self.tconv6_4(x))
        
        x = self.igdn4_4(x)

        x = F.leaky_relu(self.tconv6_5(x))
        
        x = self.igdn4_5(x)

        x = x + xA

        xA = self.igdn4_A_3(x)

        x = F.leaky_relu(self.tconv6_6(x))
        
        x = self.igdn4_6(x)

        x = F.leaky_relu(self.tconv6_7(x))
        
        x = self.igdn4_7(x)

        x = x + xA

        # n*4*128*128 -> n*8*128*128
        x = self.tconv5(x)

        xA = self.igdn8_A_0(x)

        x = F.leaky_relu(self.tconv5_0(x))
        
        x = self.igdn8_0(x)

        x = F.leaky_relu(self.tconv5_1(x))
        
        x = self.igdn8_1(x)

        x = x + xA

        xA = self.igdn8_A_1(x)

        x = F.leaky_relu(self.tconv5_2(x))
        
        x = self.igdn8_2(x)

        x = F.leaky_relu(self.tconv5_3(x))
        
        x = self.igdn8_3(x)

        x = x + xA

        xA = self.igdn8_A_2(x)

        x = F.leaky_relu(self.tconv5_4(x))
        
        x = self.igdn8_4(x)

        x = F.leaky_relu(self.tconv5_5(x))
        
        x = self.igdn8_5(x)

        x = x + xA

        xA = self.igdn8_A_3(x)

        x = F.leaky_relu(self.tconv5_6(x))
        
        x = self.igdn8_6(x)

        x = F.leaky_relu(self.tconv5_7(x))
        
        x = self.igdn8_7(x)

        x = x + xA

        # n*8*128*128 -> n*16*128*128
        x = self.tconv4(x)

        xA = self.igdn16_A_0(x)

        x = F.leaky_relu(self.tconv4_0(x))
        
        x = self.igdn16_0(x)

        x = F.leaky_relu(self.tconv4_1(x))
        
        x = self.igdn16_1(x)

        x = x + xA

        xA = self.igdn16_A_1(x)

        x = F.leaky_relu(self.tconv4_2(x))
        
        x = self.igdn16_2(x)

        x = F.leaky_relu(self.tconv4_3(x))
        
        x = self.igdn16_3(x)

        x = x + xA

        xA = self.igdn16_A_2(x)

        x = F.leaky_relu(self.tconv4_4(x))
        
        x = self.igdn16_4(x)

        x = F.leaky_relu(self.tconv4_5(x))
        
        x = self.igdn16_5(x)

        x = x + xA

        xA = self.igdn16_A_3(x)

        x = F.leaky_relu(self.tconv4_6(x))
        
        x = self.igdn16_6(x)

        x = F.leaky_relu(self.tconv4_7(x))
        
        x = self.igdn16_7(x)

        x = x + xA

        # n*16*128*128 -> n*32*128*128
        x = self.tconv3(x)

        xA = self.igdn32_A_0(x)

        x = F.leaky_relu(self.tconv3_0(x))
        
        x = self.igdn32_0(x)

        x = F.leaky_relu(self.tconv3_1(x))
        
        x = self.igdn32_1(x)

        x = x + xA

        xA = self.igdn32_A_1(x)

        x = F.leaky_relu(self.tconv3_2(x))
        
        x = self.igdn32_2(x)

        x = F.leaky_relu(self.tconv3_3(x))
        
        x = self.igdn32_3(x)

        x = x + xA

        xA = self.igdn32_A_2(x)

        x = F.leaky_relu(self.tconv3_4(x))
        
        x = self.igdn32_4(x)

        x = F.leaky_relu(self.tconv3_5(x))
        
        x = self.igdn32_5(x)

        x = x + xA

        xA = self.igdn32_A_3(x)

        x = F.leaky_relu(self.tconv3_6(x))
        
        x = self.igdn32_6(x)

        x = F.leaky_relu(self.tconv3_7(x))
        
        x = self.igdn32_7(x)

        x = x + xA

        # n*32*128*128 -> n*64*128*128
        x = self.tconv2(x)

        xA = self.igdn64_A_0(x)

        x = F.leaky_relu(self.tconv2_0(x))
        
        x = self.igdn64_0(x)

        x = F.leaky_relu(self.tconv2_1(x))
        
        x = self.igdn64_1(x)

        x = x + xA

        xA = self.igdn64_A_1(x)

        x = F.leaky_relu(self.tconv2_2(x))
        
        x = self.igdn64_2(x)

        x = F.leaky_relu(self.tconv2_3(x))
        
        x = self.igdn64_3(x)

        x = x + xA

        xA = self.igdn64_A_2(x)

        x = F.leaky_relu(self.tconv2_4(x))
        
        x = self.igdn64_4(x)

        x = F.leaky_relu(self.tconv2_5(x))
        
        x = self.igdn64_5(x)

        x = x + xA

        xA = self.igdn64_A_3(x)

        x = F.leaky_relu(self.tconv2_6(x))
        
        x = self.igdn64_6(x)

        x = F.leaky_relu(self.tconv2_7(x))
        
        x = self.igdn64_7(x)

        x = x + xA

        # n*64*128*128 -> n*128*128*128
        x = self.tconv1(x)

        xA = self.igdn128_A_0(x)

        x = F.leaky_relu(self.tconv1_0(x))
        
        x = self.igdn128_0(x)

        x = F.leaky_relu(self.tconv1_1(x))
        
        x = self.igdn128_1(x)

        x = x + xA

        xA = self.igdn128_A_1(x)

        x = F.leaky_relu(self.tconv1_2(x))
        
        x = self.igdn128_2(x)

        x = F.leaky_relu(self.tconv1_3(x))
        
        x = self.igdn128_3(x)

        x = x + xA

        # n*128*128*128 -> n*128*256*256
        x = self.tconv_up_0(x)

        xA = self.igdn128_A_2(x)

        x = F.leaky_relu(self.tconv1_4(x))
        
        x = self.igdn128_4(x)

        x = F.leaky_relu(self.tconv1_5(x))
        
        x = self.igdn128_5(x)

        x = x + xA

        xA = self.igdn128_A_3(x)

        x = F.leaky_relu(self.tconv1_6(x))
        
        x = self.igdn128_6(x)

        x = F.leaky_relu(self.tconv1_7(x))
        
        x = self.igdn128_7(x)

        x = x + xA

        # n*128*256*256 -> n*1*256*256
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
batchSize = 3 # 一次读取?张图片进行训练
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











