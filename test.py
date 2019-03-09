from bitstream import BitStream
import numpy
from numpy import *
import huffmanEncode

ACArray = numpy.array([12,10,1,-7,0,0,-4],dtype=int)
ACArray = numpy.hstack((ACArray,numpy.zeros([56],dtype=int)))
print(ACArray)
print(huffmanEncode.encodeACBlock(ACArray,1))