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

# https://cs.stanford.edu/people/eroberts/courses/soco/projects/data-compression/lossy/jpeg/dct.htm
a = numpy.array([140,144,147,140,140,155,179,175,144,152,140,147,140,148,167,179,152,155,136,167,163,162,152,172,168,145,156,160,152,155,136,160,162,148,156,148,140,136,147,162,147,167,140,155,155,140,136,162,136,156,123,167,162,144,140,147,148,155,136,155,152,147,147,136])
a.reshape([8,8])
a.astype(numpy.int)
#a = a - 128
print(fftpack.dct(fftpack.dct(a,norm='ortho').T,norm='ortho').T)


