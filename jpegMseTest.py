from PIL import Image
import numpy

img1 = Image.open('/home/nvidia/桌面/0.bmp').convert('L')
imgData1 = numpy.asarray(img1).astype(float)
print(imgData1)

img2 = Image.open('/home/nvidia/文档/dataSet/256bmp/0.bmp').convert('L')
imgData2 = numpy.asarray(img2).astype(float)
print(imgData2)

lossData = (imgData1-imgData2)**2
print(lossData)
print(lossData.mean())
lossData = 255*(lossData - numpy.min(lossData))/(numpy.max(lossData) - numpy.min(lossData))
img = Image.fromarray(lossData.astype(numpy.uint8))
Image._show(img1)
Image._show(img2)
