from bitstream import BitStream
import numpy
from numpy import *
from collections import OrderedDict

#hexToBytes('F0') = 1111 1111 0000 0000(bytes)
def hexToBytes(hexStr):
    num = len(hexStr)//2
    ret = numpy.zeros([num],dtype=int)
    for i in range(num):
        ret[i] = int(hexStr[2*i:2*i+2],16)

    ret = ret.tolist()
    ret = bytes(ret)
    return ret

#Image Compression: JPEG Multimedia Systems (Module 4 Lesson 1).pdf

#The DC Hoffman coding table for luminance recommended by JPEG
DCLuminanceSizeToCode = [
    [0,0],              #0 EOB
    [0,1,0],            #1
    [0,1,1],            #2
    [1,0,0],            #3
    [1,0,1],            #4
    [1,1,0],            #5
    [1,1,1,0],          #6
    [1,1,1,1,0],        #7
    [1,1,1,1,1,0],      #8
    [1,1,1,1,1,1,0],    #9
    [1,1,1,1,1,1,1,0],  #10 0A
    [1,1,1,1,1,1,1,1,0] #11 0B
]
#将DC huffman Luminance 表保存成十六进制（字符串形式）
def DCLuminanceTableToBytes():

    #codeLength[0]保存比特数为1的数量
    codeLength = numpy.zeros([16],dtype=int)
    #category[0]保存比特为1的类别
    category = []
    for i in range(16):
        category.append([])

    for i in range(len(DCLuminanceSizeToCode)):
        #比特数为currentLength 对应的类别为i
        currentLength = len(DCLuminanceSizeToCode[i])
        codeLength[currentLength-1] = codeLength[currentLength-1] + 1
        category[currentLength-1].append(i)

    tableList = codeLength.tolist()
    for i in range(16):
        if(len(category[i])>0):
            category[i].sort()
            tableList.extend(category[i])
    #print(tableList)
    #>>[0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    bytesLength = len(tableList) + 3
    bytesLengthHex = hex(bytesLength)[2:]
    while len(bytesLengthHex) != 4:
        bytesLengthHex = '0' + bytesLengthHex
    headList = []
    headList.extend(([255,196,int(bytesLengthHex[0:2],16),int(bytesLengthHex[2:4],16),0])) # FF C4 00 1F 00
    ret = headList + tableList
    #print(ret)
    #>>[255, 196, 0, 31, 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    return bytes(ret)














#The DC Hoffman coding table for chrominance recommended by JPEG
DCChrominanceSizeToCode = [
    [0,0],                 #0 EOB
    [0,1],                 #1
    [1,0],                 #2
    [1,1,0],               #3
    [1,1,1,0],             #4
    [1,1,1,1,0],           #5
    [1,1,1,1,1,0],         #6
    [1,1,1,1,1,1,0],       #7
    [1,1,1,1,1,1,1,0],     #8
    [1,1,1,1,1,1,1,1,0],   #9
    [1,1,1,1,1,1,1,1,1,0], #10 0A
    [1,1,1,1,1,1,1,1,1,1,0]#11 0B
]

def DCChrominanceTableToBytes():

    #codeLength[0]保存比特数为1的数量
    codeLength = numpy.zeros([16],dtype=int)
    #category[0]保存比特为1的类别
    category = []
    for i in range(16):
        category.append([])

    for i in range(len(DCChrominanceSizeToCode)):
        #比特数为currentLength 对应的类别为i
        currentLength = len(DCChrominanceSizeToCode[i])
        codeLength[currentLength-1] = codeLength[currentLength-1] + 1
        category[currentLength-1].append(i)

    tableList = codeLength.tolist()
    for i in range(16):
        if(len(category[i])>0):
            category[i].sort()
            tableList.extend(category[i])
    #print(tableList)
    #>>[0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    bytesLength = len(tableList) + 3
    bytesLengthHex = hex(bytesLength)[2:]
    while len(bytesLengthHex) != 4:
        bytesLengthHex = '0' + bytesLengthHex
    headList = []
    headList.extend(([255,196,int(bytesLengthHex[0:2],16),int(bytesLengthHex[2:4],16),1])) # FF C4 00 B5 01
    ret = headList + tableList
    #print(ret)
    #>>[255, 196, 0, 31, 1, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    return bytes(ret)

ACLuminanceSizeToCode = {
'0/0':[1,0,1,0],#EOB

'0/1':[0,0],

'0/2':[0,1],

'0/3':[1,0,0],

'0/4':[1,0,1,1],

'0/5':[1,1,0,1,0],

'0/6':[1,1,1,1,0,0,0],

'0/7':[1,1,1,1,1,0,0,0],

'0/8':[1,1,1,1,1,1,0,1,1,0],

'0/9':[1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0],

'0/A':[1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1],

'1/1':[1,1,0,0],

'1/2':[1,1,0,1,1],

'1/3':[1,1,1,1,0,0,1],

'1/4':[1,1,1,1,1,0,1,1,0],

'1/5':[1,1,1,1,1,1,1,0,1,1,0],

'1/6':[1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0],

'1/7':[1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1],

'1/8':[1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0],

'1/9':[1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1],

'1/A':[1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0],

'2/1':[1,1,1,0,0],

'2/2':[1,1,1,1,1,0,0,1],

'2/3':[1,1,1,1,1,1,0,1,1,1],

'2/4':[1,1,1,1,1,1,1,1,0,1,0,0],

'2/5':[1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,1],

'2/6':[1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,0],

'2/7':[1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,1],

'2/8':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0],

'2/9':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,1],

'2/A':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0],

'3/1':[1,1,1,0,1,0],

'3/2':[1,1,1,1,1,0,1,1,1],

'3/3':[1,1,1,1,1,1,1,1,0,1,0,1],

'3/4':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1],

'3/5':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0],

'3/6':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,1],

'3/7':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,0],

'3/8':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1],

'3/9':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,0],

'3/A':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,1],

'4/1':[1,1,1,0,1,1],

'4/2':[1,1,1,1,1,1,1,0,0,0],

'4/3':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,1,0],

'4/4':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1],

'4/5':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0],

'4/6':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1],

'4/7':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,0],

'4/8':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1],

'4/9':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,0],

'4/A':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1],

'5/1':[1,1,1,1,0,1,0],

'5/2':[1,1,1,1,1,1,1,0,1,1,1],

'5/3':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0],

'5/4':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1],

'5/5':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0],

'5/6':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,1],

'5/7':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0],

'5/8':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,1],

'5/9':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,0],

'5/A':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,1],

'6/1':[1,1,1,1,0,1,1],

'6/2':[1,1,1,1,1,1,1,1,0,1,1,0],

'6/3':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0],

'6/4':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1],

'6/5':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0],

'6/6':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,1],

'6/7':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0],

'6/8':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1],

'6/9':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,0,0],

'6/A':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,0,1],

'7/1':[1,1,1,1,1,0,1,0],

'7/2':[1,1,1,1,1,1,1,1,0,1,1,1],

'7/3':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0],

'7/4':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1],

'7/5':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,0],

'7/6':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,1],

'7/7':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,0],

'7/8':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1],

'7/9':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,0],

'7/A':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1],

'8/1':[1,1,1,1,1,1,0,0,0],

'8/2':[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],

'8/3':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0],

'8/4':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1],

'8/5':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0],

'8/6':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1],

'8/7':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0],

'8/8':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1],

'8/9':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0],

'8/A':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1],

'9/1':[1,1,1,1,1,1,0,0,1],

'9/2':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0],

'9/3':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1],

'9/4':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],

'9/5':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1],

'9/6':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0],

'9/7':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1],

'9/8':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,0],

'9/9':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1],

'9/A':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0],

'A/1':[1,1,1,1,1,1,0,1,0],

'A/2':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1],

'A/3':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0],

'A/4':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1],

'A/5':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,0],

'A/6':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,1],

'A/7':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0],

'A/8':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1],

'A/9':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0],

'A/A':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1],

'B/1':[1,1,1,1,1,1,1,0,0,1],

'B/2':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0],

'B/3':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1],

'B/4':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,0],

'B/5':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1],

'B/6':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0],

'B/7':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1],

'B/8':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,0],

'B/9':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1],

'B/A':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,0],

'C/1':[1,1,1,1,1,1,1,0,1,0],

'C/2':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1],

'C/3':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0],

'C/4':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1],

'C/5':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0],

'C/6':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1],

'C/7':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0],

'C/8':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1],

'C/9':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],

'C/A':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1],

'D/1':[1,1,1,1,1,1,1,1,0,0,0],

'D/2':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0],

'D/3':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1],

'D/4':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0],

'D/5':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1],

'D/6':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0],

'D/7':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1],

'D/8':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0],

'D/9':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1],

'D/A':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0],

'E/1':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1],

'E/2':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0],

'E/3':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1],

'E/4':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0],

'E/5':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1],

'E/6':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],

'E/7':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1],

'E/8':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0],

'E/9':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],

'E/A':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0],

'F/0':[1,1,1,1,1,1,1,1,1,1,0,0,1],

'F/1':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1],

'F/2':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0],

'F/3':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1],

'F/4':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],

'F/5':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1],

'F/6':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0],

'F/7':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1],

'F/8':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],

'F/9':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],

'F/A':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
}

ACLuminanceSizeToCodeList = [
[1, 0, 1, 0] ,
[0, 0] ,
[0, 1] ,
[1, 0, 0] ,
[1, 0, 1, 1] ,
[1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1] ,
[1, 1, 0, 0] ,
[1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0] ,
[1, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0] ,
[1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1] ,
[1, 1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1] ,
[1, 1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1] ,
[1, 1, 1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
]

def ACLuminanceTableToBytes():

    #codeLength[0]保存比特数为1的数量
    codeLength = numpy.zeros([16],dtype=int)
    #category[0]保存比特为1的类别
    category = []
    for i in range(16):
        category.append([])


    for i in range(len(ACLuminanceSizeToCodeList)):
        #比特数为currentLength 对应的类别为i
        currentLength = len(ACLuminanceSizeToCodeList[i])
        codeLength[currentLength-1] = codeLength[currentLength-1] + 1
        category[currentLength-1].append(i)

    tableList = codeLength.tolist()
    for i in range(16):
        if(len(category[i])>0):
            category[i].sort()
            tableList.extend(category[i])
    #print(tableList)

    bytesLength = len(tableList) + 3
    bytesLengthHex = hex(bytesLength)[2:]
    while len(bytesLengthHex) != 4:
        bytesLengthHex = '0' + bytesLengthHex
    headList = []
    headList.extend(([255,196,int(bytesLengthHex[0:2],16),int(bytesLengthHex[2:4],16),16])) # FF C4 00 B5 10
    ret = headList + tableList
    #print(ret)

    return bytes(ret)

ACChrominanceToCode = {
'0/0':[0,0], #EOB

'0/1':[0,1],

'0/2':[1,0,0],

'0/3':[1,0,1,0],

'0/4':[1,1,0,0,0],

'0/5':[1,1,0,0,1],

'0/6':[1,1,1,0,0,0],

'0/7':[1,1,1,1,0,0,0],

'0/8':[1,1,1,1,1,0,1,0,0],

'0/9':[1,1,1,1,1,1,0,1,1,0],

'0/A':[1,1,1,1,1,1,1,1,0,1,0,0],

'1/1':[1,0,1,1],

'1/2':[1,1,1,0,0,1],

'1/3':[1,1,1,1,0,1,1,0],

'1/4':[1,1,1,1,1,0,1,0,1],

'1/5':[1,1,1,1,1,1,1,0,1,1,0],

'1/6':[1,1,1,1,1,1,1,1,0,1,0,1],

'1/7':[1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0],

'1/8':[1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,1],

'1/9':[1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,0],

'1/A':[1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,1],

'2/1':[1,1,0,1,0],

'2/2':[1,1,1,1,0,1,1,1],

'2/3':[1,1,1,1,1,1,0,1,1,1],

'2/4':[1,1,1,1,1,1,1,1,0,1,1,0],

'2/5':[1,1,1,1,1,1,1,1,1,0,0,0,0,1,0],

'2/6':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0],

'2/7':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,1],

'2/8':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0],

'2/9':[1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1],

'2/A':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0],

'3/1':[1,1,0,1,1],

'3/2':[1,1,1,1,1,0,0,0],

'3/3':[1,1,1,1,1,1,1,0,0,0],

'3/4':[1,1,1,1,1,1,1,1,0,1,1,1],

'3/5':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,1],

'3/6':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,0],

'3/7':[1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1],

'3/8':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,0],

'3/9':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,1],

'3/A':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,1,0],

'4/1':[1,1,1,0,1,0],

'4/2':[1,1,1,1,1,0,1,1,0],

'4/3':[1,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1],

'4/4':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0],

'4/5':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1],

'4/6':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,0],

'4/7':[1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1],

'4/8':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,0],

'4/9':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1],

'4/A':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0],

'5/1':[1,1,1,0,1,1],

'5/2':[1,1,1,1,1,1,1,0,0,1],

'5/3':[1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1],

'5/4':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0],

'5/5':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,1],

'5/6':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0],

'5/7':[1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,1],

'5/8':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,0],

'5/9':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,1],

'5/A':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0],

'6/1':[1,1,1,1,0,0,1],

'6/2':[1,1,1,1,1,1,1,0,1,1,1],

'6/3':[1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1],

'6/4':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0],

'6/5':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,1],

'6/6':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0],

'6/7':[1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1],

'6/8':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,0,0],

'6/9':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,0,1],

'6/A':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0],

'7/1':[1,1,1,1,0,1,0],

'7/2':[1,1,1,1,1,1,1,1,0,0,0],

'7/3':[1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1],

'7/4':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,0],

'7/5':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,1],

'7/6':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,0],

'7/7':[1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1],

'7/8':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,0],

'7/9':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1],

'7/A':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0],

'8/1':[1,1,1,1,1,0,0,1],

'8/2':[1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1],

'8/3':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0],

'8/4':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1],

'8/5':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0],

'8/6':[1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1],

'8/7':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0],

'8/8':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1],

'8/9':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0],

'8/A':[1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1],

'9/1':[1,1,1,1,1,0,1,1,1],

'9/2':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],

'9/3':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1],

'9/4':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0],

'9/5':[1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1],

'9/6':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,0],

'9/7':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1],

'9/8':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0],

'9/9':[1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1],

'9/A':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0],

'A/1':[1,1,1,1,1,1,0,0,0],

'A/2':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1],

'A/3':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,0],

'A/4':[1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,1],

'A/5':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0],

'A/6':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1],

'A/7':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0],

'A/8':[1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1],

'A/9':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0],

'A/A':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1],

'B/1':[1,1,1,1,1,1,0,0,1],

'B/2':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,0],

'B/3':[1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1],

'B/4':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0],

'B/5':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1],

'B/6':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,0],

'B/7':[1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1],

'B/8':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,0],

'B/9':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1],

'B/A':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0],

'C/1':[1,1,1,1,1,1,0,1,0],

'C/2':[1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1],

'C/3':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0],

'C/4':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1],

'C/5':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0],

'C/6':[1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1],

'C/7':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],

'C/8':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1],

'C/9':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0],

'C/A':[1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1],

'D/1':[1,1,1,1,1,1,1,1,0,0,1],

'D/2':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0],

'D/3':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1],

'D/4':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0],

'D/5':[1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1],

'D/6':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0],

'D/7':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1],

'D/8':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0],

'D/9':[1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1],

'D/A':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0],

'E/1':[1,1,1,1,1,1,1,1,1,0,0,0,0,0],

'E/2':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1],

'E/3':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0],

'E/4':[1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1],

'E/5':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],

'E/6':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1],

'E/7':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0],

'E/8':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],

'E/9':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0],

'E/A':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1],

'F/0':[1,1,1,1,1,1,1,0,1,0],

'F/1':[1,1,1,1,1,1,1,1,1,0,0,0,0,1,1],

'F/2':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0],

'F/3':[1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1],

'F/4':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],

'F/5':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1],

'F/6':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0],

'F/7':[1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1],

'F/8':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],

'F/9':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],

'F/A':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
}

ACChrominanceToCodeList = [
[0, 0] ,
[0, 1] ,
[1, 0, 0] ,
[1, 0, 1, 0] ,
[1, 1, 0, 0, 0] ,
[1, 1, 0, 0, 1] ,
[1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0] ,
[1, 0, 1, 1] ,
[1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1] ,
[1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0] ,
[1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0] ,
[1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0] ,
[1, 1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0] ,
[1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0] ,
[1, 1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1] ,
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
]

def ACChrominanceTableToBytes():

    #codeLength[0]保存比特数为1的数量
    codeLength = numpy.zeros([16],dtype=int)
    #category[0]保存比特为1的类别
    category = []
    for i in range(16):
        category.append([])


    for i in range(len(ACChrominanceToCodeList)):
        #比特数为currentLength 对应的类别为i
        currentLength = len(ACChrominanceToCodeList[i])
        codeLength[currentLength-1] = codeLength[currentLength-1] + 1
        category[currentLength-1].append(i)

    tableList = codeLength.tolist()
    for i in range(16):
        if(len(category[i])>0):
            category[i].sort()
            tableList.extend(category[i])
    #print(tableList)

    bytesLength = len(tableList) + 3
    bytesLengthHex = hex(bytesLength)[2:]
    while len(bytesLengthHex) != 4:
        bytesLengthHex = '0' + bytesLengthHex
    headList = []
    headList.extend(([255,196,int(bytesLengthHex[0:2],16),int(bytesLengthHex[2:4],16),17])) # FF C4 00 B5 11
    ret = headList + tableList
    #print(ret)

    return bytes(ret)


#DC components are differentially coded as (SIZE,Value)
def encodeDCToBoolList(value,isLuminance):
    boolList = []
    size = int(value).bit_length() # int(0).bit_length()=0
    if(isLuminance==1):
        boolList = boolList + DCLuminanceSizeToCode[size]
    else:
        boolList = boolList + DCChrominanceSizeToCode[size]
    if(value<=0): # if value==0, codeList = [], (SIZE,VALUE)=(SIZE)=EOB
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
    boolList = boolList + codeList
    return boolList

def encodeACBlock(bitStream,ACArray,isLuminance):

    i = 0
    maxI = numpy.size(ACArray)
    while 1:
        if(i==maxI):
            break
        run = 0
        while 1:
            if(ACArray[i]!=0 or i==maxI - 1 or run==15):
                break
            else:
                run = run + 1
                i = i + 1

        value = ACArray[i]

        if(value==0 and run!=15):
            break # Rest of the components are zeros therefore we simply put the EOB to signify this fact

        size = int(value).bit_length()

        runSizeStr = str.upper(str(hex(run))[2:]) + '/' + str.upper(str(hex(size))[2:])


        if (isLuminance == 1):
            bitStream.write(ACLuminanceSizeToCode[runSizeStr], bool)
        else:
            bitStream.write(ACChrominanceToCode[runSizeStr], bool)


        if(value<=0):# if value==0, codeList = [], (SIZE,VALUE)=(SIZE)=EOB
            codeList = list(bin(value)[3:])
            for k in range(len(codeList)):
                if (codeList[k] == '0'):
                    codeList[k] = 1
                else:
                    codeList[k] = 0
        else:
            codeList = list(bin(value)[2:])
            for k in range(len(codeList)):
                if (codeList[k] == '0'):
                    codeList[k] = 0
                else:
                    codeList[k] = 1
        bitStream.write(codeList, bool)
        i = i + 1

    if (isLuminance == 1):
        bitStream.write(ACLuminanceSizeToCode['0/0'], bool) # EOB
    else:
        bitStream.write(ACChrominanceToCode['0/0'], bool)


















