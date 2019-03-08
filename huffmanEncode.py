from bitstream import BitStream
from numpy import *
#Image Compression: JPEG Multimedia Systems (Module 4 Lesson 1).pdf

sizeToCode = [
    [0,0],              #0
    [0,1,0],            #1
    [0,1,1],            #2
    [1,0,0],            #3
    [1,0,1],            #4
    [1,1,0],            #5
    [1,1,1,0],          #6
    [1,1,1,1,0],        #7
    [1,1,1,1,1,0],      #8
    [1,1,1,1,1,1,0],    #9
    [1,1,1,1,1,1,1,0],  #10
    [1,1,1,1,1,1,1,1,0],#11
]








#DC components are differentially coded as (SIZE,Value)

def encodeDC(DCArray):
    ret = BitStream()

    for value in DCArray:
        size = int(value).bit_length() # int(0).bit_length()=0
        ret.write(sizeToCode[size],bool)

        if(value<0):
            codeList = list(bin(value)[3:])
            for i in range(len(codeList)):
                if (codeList[i] == '0'):
                    codeList[i] = 1
                else:
                    codeList[i] = 0
        else:
            codeList = list(bin(value)[2:])
            for i in range(len(codeList)):
                if (codeList[i] == '0'):
                    codeList[i] = 0
                else:
                    codeList[i] = 1
        ret.write(codeList,bool)
    return ret










