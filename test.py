from bitstream import BitStream
import numpy
from numpy import *
import huffmanEncode
import jpegEncoder
from scipy import fftpack

#ACArray = numpy.array([12,10,1,-7,0,0,-4],dtype=int)
#ACArray = numpy.hstack((ACArray,numpy.zeros([56],dtype=int)))
#print(ACArray)
#bitStream = BitStream()
#huffmanEncode.encodeACBlock(bitStream,ACArray,1)
#print(bitStream)

#a = BitStream()
#a.write([0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1],bool)
#print(a.read(bytes))
#file = open('1.b','wb+')
#file.write(a.read(bytes))
#file.close()

#a = huffmanEncode.hexToBytes('FF005B0006')
#print(a)
#file = open('1.b','wb+')
#file.write(a)
#file.close()

#value = 0
#if (value <= 0):  # if value==0, codeList = [], (SIZE,VALUE)=(SIZE)=EOB
#    codeList = list(bin(value)[3:])
#    for i in range(len(codeList)):
#        if (codeList[i] == '0'):
#            codeList[i] = 1
#        else:
#            codeList[i] = 0
#
#print(codeList)
#a = BitStream()
#a.write([0],bool)
#a.write(codeList, bool)
#a.write([1],bool)
#print(a)


a = BitStream([1, 1, 1, 1, 0,1, 1, 0, 0, 1, 1, 0,1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
1, 0, 1, 0,0, 0,1, 1, 1, 1, 1, 1, 1, 0, 1, 0,1, 1, 1, 1, 1, 1, 1, 0, 1, 0,1, 1, 1, 1, 1, 1, 1, 0, 1, 0,0,0,0,0,1, 1, 1, 1, 1, 1, 1, 0, 1, 0,1, 1, 1, 1, 1, 1, 1, 0, 1, 0,1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
               0,0
               ],bool)
print(a,len(a))
a.write([1,1,1],bool)
print(a,len(a))
b = a.read(bytes)
for i in range(len(b)):
    print(hex(b[i]-0)[2:])
'''
1.jpg
F7 FA 28 A2 80 3F
11110111 11111010 00101000 10100010 10000000 00111111

1010 00101000 10100010 10000000 00111111


yDC
11110 = 07 size = 07
1111111 value = 127

yAC
OPENC


uDC


uAC


vDC


vAC



'''


