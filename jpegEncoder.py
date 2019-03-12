from PIL import Image
from scipy import fftpack
import numpy
from bitstream import BitStream
from numpy import *
import huffmanEncode


#http://home.elka.pw.edu.pl/~mmanowie/psap/neue/1%20JPEG%20Overview.htm
#libjpeg::jcparam.c


std_luminance_quant_tbl = numpy.array(
[ 16,  11,  10,  16,  24,  40,  51,  61,
  12,  12,  14,  19,  26,  58,  60,  55,
  14,  13,  16,  24,  40,  57,  69,  56,
  14,  17,  22,  29,  51,  87,  80,  62,
  18,  22,  37,  56,  68, 109, 103,  77,
  24,  35,  55,  64,  81, 104, 113,  92,
  49,  64,  78,  87, 103, 121, 120, 101,
  72,  92,  95,  98, 112, 100, 103,  99],dtype=int)
std_luminance_quant_tbl.reshape([8,8])

std_chrominance_quant_tbl = numpy.array(
[ 17,  18,  24,  47,  99,  99,  99,  99,
  18,  21,  26,  66,  99,  99,  99,  99,
  24,  26,  56,  99,  99,  99,  99,  99,
  47,  66,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99],dtype=int)
std_chrominance_quant_tbl.reshape([8,8])

zigzagOrder = numpy.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,
                           49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63])


def main():
    numpy.set_printoptions(threshold=numpy.inf)
    srcFileName = './1.bmp'
    srcImage = Image.open(srcFileName)
    srcImageWidth, srcImageHeight = srcImage.size
    print('srcImageWidth = %d srcImageHeight = %d' % (srcImageWidth, srcImageHeight))
    print('srcImage info:\n', srcImage)
    srcImageMatrix = numpy.asarray(srcImage)

    # print('srcImageMatrix:\n')
    # for y in range(srcImageHeight):
    #    for x in range(srcImageWidth):
    #        print(srcImageMatrix[y][x],end='')
    #    print("\n",end='')

    # 检测长宽是否为8的整数倍，不为则补全到整数
    imageWidth = srcImageWidth
    imageHeight = srcImageHeight
    if (srcImageWidth % 8 != 0):
        imageWidth = srcImageWidth // 8 * 8 + 8
    if (srcImageHeight % 8 != 0):
        imageHeight = srcImageHeight // 8 * 8 + 8

    print('added to: ', imageWidth, imageHeight)
    addedImageMatrix = numpy.zeros((imageHeight, imageWidth, 3), dtype=numpy.uint8)

    for y in range(srcImageHeight):
        for x in range(srcImageWidth):
            addedImageMatrix[y][x] = srcImageMatrix[y][x]

    # Image._show(Image.fromarray(addedImageMatrix))

    # 转换为yuv
    yuvImage = Image.fromarray(addedImageMatrix).convert('YCbCr')
    print('yuvImage info:\n', yuvImage)
    yuvImageMatrix = numpy.asarray(yuvImage)
    # print('yuvImageMatrix:\n')
    # for y in range(srcImageHeight):
    #    for x in range(srcImageWidth):
    #        print(yuvImageMatrix[y][x],end='')
    #    print("\n",end='')

    # 分离三个通道并中心化

    yuvImageMatrix = yuvImageMatrix.astype(numpy.int)
    yuvImageMatrix = yuvImageMatrix - 128

    yImageMatrix = yuvImageMatrix[:, :, 0]
    uImageMatrix = yuvImageMatrix[:, :, 1]
    vImageMatrix = yuvImageMatrix[:, :, 2]

    # print('yImageMatrix:\n',yImageMatrix)
    # print('uImageMatrix:\n',uImageMatrix)
    # print('vImageMatrix:\n',vImageMatrix)

    # 设置品质因子
    quality = 70

    if (quality <= 0):
        quality = 1

    if (quality > 100):
        quality = 100

    if (quality < 50):
        qualityScale = 5000 / quality
    else:
        qualityScale = 200 - quality * 2

    luminanceQuantTbl = numpy.array(numpy.floor((std_luminance_quant_tbl * qualityScale + 50) / 100))
    luminanceQuantTbl[luminanceQuantTbl == 0] = 1
    luminanceQuantTbl = luminanceQuantTbl.reshape([8, 8])
    print('luminanceQuantTbl:\n', luminanceQuantTbl)
    chrominanceQuantTbl = numpy.array(numpy.floor((std_chrominance_quant_tbl * qualityScale + 50) / 100))
    chrominanceQuantTbl[chrominanceQuantTbl == 0] = 1
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([8, 8])
    print('chrominanceQuantTbl:\n', chrominanceQuantTbl)

    # 分块
    blockSum = imageWidth // 8 * imageHeight // 8

    yDC = numpy.zeros((blockSum), dtype=int)
    uDC = numpy.zeros((blockSum), dtype=int)
    vDC = numpy.zeros((blockSum), dtype=int)
    # 保存所有块的亮度DC值
    # 但是需要注意，只有第0个是真正的DC值，后面的保存的都是和前者的差值
    print('blockSum = ', blockSum)



    sosBitStream = BitStream()

    blockNum = 0
    for y in range(0, imageHeight, 8):
        for x in range(0, imageWidth, 8):
            print(y, x, ' -> ', y + 8, x + 8)
            yDctMatrix = fftpack.dct(fftpack.dct(yImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            uDctMatrix = fftpack.dct(fftpack.dct(uImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            vDctMatrix = fftpack.dct(fftpack.dct(vImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            print('yDctMatrix:\n',yDctMatrix)
            print('uDctMatrix:\n',uDctMatrix)
            print('vDctMatrix:\n',vDctMatrix)

            # 量化
            # Although JPEG allows for the use of any quantization matrix, ISO has done extensive testing and developed a standard set of quantization values that cause impressive degrees of compression.

            yQuantMatrix = numpy.rint(yDctMatrix / luminanceQuantTbl)
            uQuantMatrix = numpy.rint(uDctMatrix / chrominanceQuantTbl)
            vQuantMatrix = numpy.rint(vDctMatrix / chrominanceQuantTbl)

            # print('yQuantMatrix:\n',yQuantMatrix)
            # print('uQuantMatrix:\n',uQuantMatrix)
            # print('vQuantMatrix:\n',vQuantMatrix)

            # z字形遍历，转化为一维数组
            yZCode = yQuantMatrix.reshape([64])[zigzagOrder]
            uZCode = uQuantMatrix.reshape([64])[zigzagOrder]
            vZCode = vQuantMatrix.reshape([64])[zigzagOrder]
            yZCode = yZCode.astype(numpy.int)
            uZCode = uZCode.astype(numpy.int)
            vZCode = vZCode.astype(numpy.int)

            if (blockNum == 0):
                yDC[blockNum] = yZCode[0]
                uDC[blockNum] = uZCode[0]
                vDC[blockNum] = vZCode[0]
            else:
                yDC[blockNum] = yZCode[0] - yDC[blockNum - 1]
                uDC[blockNum] = uZCode[0] - uDC[blockNum - 1]
                vDC[blockNum] = vZCode[0] - vDC[blockNum - 1]

            # huffman编码，可以参考https://www.impulseadventure.com/photo/jpeg-huffman-coding.html
            sosBitStream.write(huffmanEncode.encodeDCToBoolList(yDC[blockNum],1),bool)
            huffmanEncode.encodeACBlock(sosBitStream, yZCode[1:], 1)

            sosBitStream.write(huffmanEncode.encodeDCToBoolList(uDC[blockNum],0),bool)
            huffmanEncode.encodeACBlock(sosBitStream, uZCode[1:], 0)

            sosBitStream.write(huffmanEncode.encodeDCToBoolList(vDC[blockNum],0),bool)
            huffmanEncode.encodeACBlock(sosBitStream, vZCode[1:], 0)

            blockNum = blockNum + 1



    jpegFile = open('output.jpg', 'wb+')
    # 图像开始
    jpegFile.write(huffmanEncode.hexToBytes('FFD8FFE000104A46494600010200000100010000'))
    # y量化表
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004300'))
    jpegFile.write(bytes(std_luminance_quant_tbl.tolist()))
    # u/v量化表
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004301'))
    jpegFile.write(bytes(std_chrominance_quant_tbl.tolist()))
    # 帧图像开始
    jpegFile.write(huffmanEncode.hexToBytes('FFC0001108'))
    hHex = hex(srcImageHeight)[2:]  # 大于65536会出错，因为需要更高的位数。嫌麻烦，暂时不考虑
    while len(hHex) != 4:
        hHex = '0' + hHex

    jpegFile.write(huffmanEncode.hexToBytes(hHex))

    wHex = hex(srcImageWidth)[2:]  # 大于65536会出错，因为需要更高的位数。嫌麻烦，暂时不考虑
    while len(wHex) != 4:
        wHex = '0' + wHex

    jpegFile.write(huffmanEncode.hexToBytes(wHex))

    # yuv采样分别为11 11 11
    # 03    01 11 00    02 11 01    03 11 01
    # 1：1	01 11 00	02 11 01	03 11 01
    # 1：2	01 21 00	02 11 01	03 11 01
    # 1：4	01 22 00	02 11 01	03 11 01

    jpegFile.write(huffmanEncode.hexToBytes('03011100021101031101'))

    # huffman DC表0 y DC
    jpegFile.write(huffmanEncode.DCLuminanceTableToBytes())

    # huffman AC表0 y AC
    jpegFile.write(huffmanEncode.ACLuminanceTableToBytes())

    # huffman DC表1 u/v DC
    jpegFile.write(huffmanEncode.DCChrominanceTableToBytes())

    # huffman AC表1 u/v AC
    jpegFile.write(huffmanEncode.ACChrominanceTableToBytes())

    # SOS Start of Scan 编码后的数据
    # 数据是一块一块的写， yDC yAC EOB uDC uAC EOB vDC vAC EOB
    # 在encodeACBlock函数中，已经在AC后加入EOB了
    sosLength = sosBitStream.__len__()
    print(sosLength)
    filledNum = 8 - sosLength % 8 #凑不够8的倍数时需要补全
    if(filledNum!=0):
        sosBitStream.write(numpy.ones([filledNum]).tolist(),bool)
    print(sosBitStream.__len__())

    jpegFile.write(bytes([255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0])) # FF DA 00 0C 03 01 00 02 11 03 11 00 3F 00

    sosBytes = sosBitStream.read(bytes)
    for i in range(len(sosBytes)):
        jpegFile.write(bytes([sosBytes[i]]))
        if(sosBytes[i]==255):
            jpegFile.write(bytes([0])) # FF应该补全为FF00 避免混淆


    jpegFile.write(bytes([255,217])) # FF D9
    jpegFile.close()


if __name__ == '__main__':
    main()



