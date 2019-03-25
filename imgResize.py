from PIL import Image
import os

imgDir = '/home/nvidia/PycharmProjects/losslyImageCompression/512bmp'

imgList = os.listdir(imgDir)
num = 0
for fileName in imgList:
    img = Image.open(imgDir + '/' + fileName)
    img = img.crop((0,0,512,512))
    img.save('/home/nvidia/文档/dataSet/512bmp/' + str(num) + '.bmp')
    num = num + 1

    nimg = img.transpose(Image.FLIP_LEFT_RIGHT)
    nimg.save('/home/nvidia/文档/dataSet/512bmp/' + str(num) + '.bmp')
    num = num + 1
    nimg = img.transpose(Image.FLIP_TOP_BOTTOM)
    nimg.save('/home/nvidia/文档/dataSet/512bmp/' + str(num) + '.bmp')
    num = num + 1
    nimg = img.transpose(Image.ROTATE_90)
    nimg.save('/home/nvidia/文档/dataSet/512bmp/' + str(num) + '.bmp')
    num = num + 1
    nimg = img.transpose(Image.ROTATE_180)
    nimg.save('/home/nvidia/文档/dataSet/512bmp/' + str(num) + '.bmp')
    num = num + 1
    nimg = img.transpose(Image.ROTATE_270)
    nimg.save('/home/nvidia/文档/dataSet/512bmp/' + str(num) + '.bmp')
    num = num + 1
    nimg = img.transpose(Image.TRANSPOSE)
    nimg.save('/home/nvidia/文档/dataSet/512bmp/' + str(num) + '.bmp')
    num = num + 1
    nimg = img.transpose(Image.TRANSVERSE)
    nimg.save('/home/nvidia/文档/dataSet/512bmp/' + str(num) + '.bmp')
    num = num + 1