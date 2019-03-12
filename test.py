from bitstream import BitStream
import numpy
from numpy import *
import huffmanEncode


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

huffmanEncode.writeDCTable0ToHexStr()