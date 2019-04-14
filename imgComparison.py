import cv2
import numpy

def getSSIM(x, y):
    uX = numpy.mean(x)
    uY = numpy.mean(y)
    sigmaX2 = numpy.var(x)
    sigmaY2 = numpy.var(x)
    c1 = (0.01*255)**2
    c2 = (0.03*255)**2
    return (2*uX*uY+c1)*(2*numpy.cov(x,y)+c2)/(uX**2+uY**2+c1)/(sigmaX2+sigmaY2+c2)


img1Path = '/home/nvidia/文档/dataSet/256bmp/64.bmp'
img2Path = '/home/nvidia/桌面/newImg/8.bmp'
img1Data = numpy.asarray(cv2.imread(img1Path, 0), dtype=float)
img2Data = numpy.asarray(cv2.imread(img2Path, 0), dtype=float)

errorData = img1Data - img2Data
print('MSE = ', numpy.mean(errorData**2))
print('SSIM = ', getSSIM(img1Data, img2Data))
