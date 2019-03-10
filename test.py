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

a = BitStream()
a.write([0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1],bool)
print(a.read(bytes))
#file = open('1.b','wb+')
#file.write(a.read(bytes))
#file.close()