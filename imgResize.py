from PIL import Image
import os

imgDir = '/home/nvidia/文档/dataSet/kodim24'

imgList = os.listdir(imgDir)
num = 0
for fileName in imgList:
    img = Image.open(imgDir + '/' + fileName)
    img = img.crop((0,0,512,512))
    img.save('/home/nvidia/文档/dataSet/512bmp/' + str(num) + '.bmp')
    num = num + 1