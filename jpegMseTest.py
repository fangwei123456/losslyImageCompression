from PIL import Image
import numpy

img1 = Image.open('lena.bmp').convert('L')
imgData1 = numpy.asarray(img1).astype(int)

img2 = Image.open('lena.jpg').convert('L')
imgData2 = numpy.asarray(img2).astype(int)

lossData = numpy.abs(imgData1-imgData2)
print(lossData.mean())
lossData = 255*(lossData - numpy.min(lossData))/(numpy.max(lossData) - numpy.min(lossData))
img = Image.fromarray(lossData.astype(numpy.uint8))
Image._show(img)
Image._show(img1)