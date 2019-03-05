from PIL import Image
from scipy import fftpack
import numpy

#libjpeg::jcparam.c

std_luminance_quant_tbl = numpy.array([  16,  11,  10,  16,  24,  40,  51,  61,
  12,  12,  14,  19,  26,  58,  60,  55,
  14,  13,  16,  24,  40,  57,  69,  56,
  14,  17,  22,  29,  51,  87,  80,  62,
  18,  22,  37,  56,  68, 109, 103,  77,
  24,  35,  55,  64,  81, 104, 113,  92,
  49,  64,  78,  87, 103, 121, 120, 101,
  72,  92,  95,  98, 112, 100, 103,  99])
std_luminance_quant_tbl.reshape([8,8])

std_chrominance_quant_tbl = numpy.array( [  17,  18,  24,  47,  99,  99,  99,  99,
  18,  21,  26,  66,  99,  99,  99,  99,
  24,  26,  56,  99,  99,  99,  99,  99,
  47,  66,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99])
std_chrominance_quant_tbl.reshape([8,8])

zigZagOrder = numpy.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,
                           49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63])


numpy.set_printoptions(threshold=numpy.inf)

srcFileName = './8x8.bmp'
srcImage = Image.open(srcFileName)
srcImageWidth, srcImageHeight = srcImage.size
print('srcImageWidth = %d srcImageHeight = %d'%(srcImageWidth,srcImageHeight))
print('srcImage info:\n',srcImage)
srcImageMatrix = numpy.asarray(srcImage).reshape([srcImageWidth,srcImageHeight,3])
print('srcImageMatrix:\n')
for y in range(srcImageHeight):
    for x in range(srcImageWidth):
        print(srcImageMatrix[y][x],end='')
    print("\n",end='')

#转换为yuv
yuvImage = srcImage.convert('YCbCr')
print('yuvImage info:\n',yuvImage)
yuvImageMatrix = numpy.asarray(yuvImage).reshape([srcImageWidth,srcImageHeight,3])
print('yuvImageMatrix:\n')
for y in range(srcImageHeight):
    for x in range(srcImageWidth):
        print(yuvImageMatrix[y][x],end='')
    print("\n",end='')

#分离三个通道并中心化
yuvImageMatrix = yuvImageMatrix.astype(numpy.int16)
yuvImageMatrix = yuvImageMatrix - 128;

yImageMatrix = yuvImageMatrix[:,:,0];
uImageMatrix = yuvImageMatrix[:,:,1];
vImageMatrix = yuvImageMatrix[:,:,2];

print('yImageMatrix:\n',yImageMatrix)
print('uImageMatrix:\n',uImageMatrix)
print('vImageMatrix:\n',vImageMatrix)

yDctMatrix = fftpack.dct(yImageMatrix,norm='ortho');
uDctMatrix = fftpack.dct(yImageMatrix,norm='ortho');
vDctMatrix = fftpack.dct(yImageMatrix,norm='ortho');
print('yDctMatrix:\n',yDctMatrix)
print('uDctMatrix:\n',uDctMatrix)
print('vDctMatrix:\n',vDctMatrix)

#量化
#Although JPEG allows for the use of any quantization matrix, ISO has done extensive testing and developed a standard set of quantization values that cause impressive degrees of compression.

#设置品质因子
quality = 70

if(quality<=0):
    quality = 1

if(quality>100):
    quality = 100

if(quality<50):
    qualityScale = 5000 / quality
else:
    qualityScale = 200 - quality * 2

luminanceQuantTbl = numpy.array(numpy.floor((std_luminance_quant_tbl*qualityScale + 50)/100))
luminanceQuantTbl[luminanceQuantTbl==0] = 1
luminanceQuantTbl = luminanceQuantTbl.reshape([8,8])
print('luminanceQuantTbl:\n',luminanceQuantTbl)
chrominanceQuantTbl = numpy.array(numpy.floor((std_chrominance_quant_tbl*qualityScale + 50)/100))
chrominanceQuantTbl[chrominanceQuantTbl==0] = 1
chrominanceQuantTbl = chrominanceQuantTbl.reshape([8,8])
print('chrominanceQuantTbl:\n',chrominanceQuantTbl)

yQuantMatrix = numpy.round(yDctMatrix/luminanceQuantTbl)
uQuantMatrix = numpy.round(uDctMatrix/chrominanceQuantTbl)
vQuantMatrix = numpy.round(vDctMatrix/chrominanceQuantTbl)
print(yQuantMatrix)
print(uQuantMatrix)
print(vQuantMatrix)






