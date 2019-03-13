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
std_luminance_quant_tbl = std_luminance_quant_tbl.reshape([8,8])

std_chrominance_quant_tbl = numpy.array(
[ 17,  18,  24,  47,  99,  99,  99,  99,
  18,  21,  26,  66,  99,  99,  99,  99,
  24,  26,  56,  99,  99,  99,  99,  99,
  47,  66,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99],dtype=int)
std_chrominance_quant_tbl = std_chrominance_quant_tbl.reshape([8,8])

zigzagOrder = numpy.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,
                           49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63])


def main():
    numpy.set_printoptions(threshold=numpy.inf)
    srcFileName = './8x8.bmp'
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


    # 转换为yuv 分离三个通道并中心化
    # Image._show(Image.fromarray(addedImageMatrix))
    yImage,uImage,vImage = Image.fromarray(addedImageMatrix).convert('YCbCr').split()


    yImageMatrix = numpy.asarray(yImage)
    uImageMatrix = numpy.asarray(uImage)
    vImageMatrix = numpy.asarray(vImage)
    print(yImageMatrix)
    print(uImageMatrix)
    print(vImageMatrix)

    yImageMatrix = yImageMatrix - 127
    uImageMatrix = uImageMatrix - 127
    vImageMatrix = vImageMatrix - 127
    # 设置品质因子
    '''
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
    '''
    luminanceQuantTbl = numpy.array([
           2,2,2,   2,   3,   4,   5,   6,
           2,2,2,   2,   3,   4,   5,   6,
           2,2,2,   2,   4,   5,   7,   9,
           2,2,2,   4,   5,   7,   9,  12,
           3,3,4,   5,   8,  10,  12,  12,
           4,4,5,   7,  10,  12,  12,  12,
           5,5,7,   9,  12,  12,  12,  12,
           6,6,9,  12,  12,  12,  12,  12 ],dtype=int)
    luminanceQuantTbl = luminanceQuantTbl.reshape([8,8])

    chrominanceQuantTbl = numpy.array([
       3,   3,   5,   9,  13,  15,  15,  15,
       3,   4,   6,  11,  14,  12,  12,  12,
       5,   6,   9,  14,  12,  12,  12,  12,
       9,  11,  14,  12,  12,  12,  12,  12,
      13,  14,  12,  12,  12,  12,  12,  12,
      15,  12,  12,  12,  12,  12,  12,  12,
      15,  12,  12,  12,  12,  12,  12,  12,
      15,  12,  12,  12,  12,  12,  12,  12
    ],dtype=int)
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([8,8])
    # 分块
    blockSum = imageWidth // 8 * imageHeight // 8

    yDC = numpy.zeros([blockSum], dtype=int)
    uDC = numpy.zeros([blockSum], dtype=int)
    vDC = numpy.zeros([blockSum], dtype=int)
    dyDC = numpy.zeros([blockSum], dtype=int)
    duDC = numpy.zeros([blockSum], dtype=int)
    dvDC = numpy.zeros([blockSum], dtype=int)
    # 保存所有块的亮度DC值 DC差值
    print('blockSum = ', blockSum)



    sosBitStream = BitStream()

    blockNum = 0
    for y in range(0, imageHeight, 8):
        for x in range(0, imageWidth, 8):
            print(y, x, ' -> ', y + 8, x + 8)
            yDctMatrix = fftpack.dct(fftpack.dct(yImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            uDctMatrix = fftpack.dct(fftpack.dct(uImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            vDctMatrix = fftpack.dct(fftpack.dct(vImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            if(blockSum<=8):
                print('yDctMatrix:\n',yDctMatrix)
                print('uDctMatrix:\n',uDctMatrix)
                print('vDctMatrix:\n',vDctMatrix)

            # 量化
            # Although JPEG allows for the use of any quantization matrix, ISO has done extensive testing and developed a standard set of quantization values that cause impressive degrees of compression.

            yQuantMatrix = numpy.rint(yDctMatrix / luminanceQuantTbl)
            uQuantMatrix = numpy.rint(uDctMatrix / chrominanceQuantTbl)
            vQuantMatrix = numpy.rint(vDctMatrix / chrominanceQuantTbl)
            if(blockSum<=8):
                print('yQuantMatrix:\n',yQuantMatrix)
                print('uQuantMatrix:\n',uQuantMatrix)
                print('vQuantMatrix:\n',vQuantMatrix)

            # z字形遍历，转化为一维数组
            yZCode = yQuantMatrix.reshape([64])[zigzagOrder]
            uZCode = uQuantMatrix.reshape([64])[zigzagOrder]
            vZCode = vQuantMatrix.reshape([64])[zigzagOrder]
            yZCode = yZCode.astype(numpy.int)
            uZCode = uZCode.astype(numpy.int)
            vZCode = vZCode.astype(numpy.int)


            yDC[blockNum] = yZCode[0]
            uDC[blockNum] = uZCode[0]
            vDC[blockNum] = vZCode[0]

            if(blockNum==0):
                dyDC[blockNum] = yDC[blockNum]
                duDC[blockNum] = uDC[blockNum]
                dvDC[blockNum] = vDC[blockNum]
            else:
                dyDC[blockNum] = yDC[blockNum] - yDC[blockNum-1]
                duDC[blockNum] = uDC[blockNum] - uDC[blockNum-1]
                dvDC[blockNum] = vDC[blockNum] - vDC[blockNum-1]



            # huffman编码，可以参考https://www.impulseadventure.com/photo/jpeg-huffman-coding.html

            sosBitStream.write(huffmanEncode.encodeDCToBoolList(dyDC[blockNum],1, 0),bool)
            huffmanEncode.encodeACBlock(sosBitStream, yZCode[1:], 1, 0)

            sosBitStream.write(huffmanEncode.encodeDCToBoolList(duDC[blockNum],0, 0),bool)
            huffmanEncode.encodeACBlock(sosBitStream, uZCode[1:], 0, 0)

            sosBitStream.write(huffmanEncode.encodeDCToBoolList(dvDC[blockNum],0, 0),bool)
            huffmanEncode.encodeACBlock(sosBitStream, vZCode[1:], 0, 0)

            blockNum = blockNum + 1



    jpegFile = open('output.jpg', 'wb+')
    # 图像开始
    jpegFile.write(huffmanEncode.hexToBytes('FFD8FFE000104A46494600010100000100010000'))
    # y量化表
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004300'))
    luminanceQuantTbl = luminanceQuantTbl.reshape([64])
    jpegFile.write(bytes(luminanceQuantTbl.tolist()))
    # u/v量化表
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004301'))
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([64])
    jpegFile.write(bytes(chrominanceQuantTbl.tolist()))
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

    '''
    # huffman DC表0 y DC
    jpegFile.write(huffmanEncode.DCLuminanceTableToBytes())

    # huffman AC表0 y AC
    jpegFile.write(huffmanEncode.ACLuminanceTableToBytes())

    # huffman DC表1 u/v DC
    jpegFile.write(huffmanEncode.DCChrominanceTableToBytes())

    # huffman AC表1 u/v AC
    jpegFile.write(huffmanEncode.ACChrominanceTableToBytes())
    '''
    #huffman table
    jpegFile.write(huffmanEncode.hexToBytes('FFC401A20000000701010101010000000000000000040503020601000708090A0B0100020203010101010100000000000000010002030405060708090A0B1000020103030204020607030402060273010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA110002020102030505040506040803036D0100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F839495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))
    # SOS Start of Scan 编码后的数据
    # 数据是一块一块的写， yDC yAC EOB uDC uAC EOB vDC vAC EOB
    # 在encodeACBlock函数中，已经在AC后加入EOB了
    #TEST
    #sosBitStream = BitStream([1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,1,0,1,0,1,0,1],bool)
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



