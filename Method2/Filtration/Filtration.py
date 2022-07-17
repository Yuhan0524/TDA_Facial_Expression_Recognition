import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

#to greyscale
def img_gray(path):
  img = cv2.imread(path)
  h,w = img.shape[:2] #获取图片的high和wide
  img_gray=np.zeros([h,w],img.dtype) #创建一张和当前图片大小一样的单通道图片
  for i in range(h):
      for j in range(w):
          m = img[i,j]
          img_gray[i,j] =int(m[0]*0.11+m[1]*0.59+m[2]*0.3) #将BGR坐标转换为gray坐标
  return img_gray

path = "/content/drive/MyDrive/data/NE.1.tiff"
im_ori = img_gray(path)

#change to 64*64 avoid noise
im = []
for i in range(0,256,4):
  for j in range(0,256,4):
    im.append(im_ori[i][j])
im = np.array(im).reshape(64,64)

#Binarization
from gtda.images import Binarizer
im_NE = im[None, :, :]
binarizer = Binarizer(threshold=0.4)

im_binarized = binarizer.fit_transform(im_NE)

#FILTRATION
#Height filtration
from gtda.images import HeightFiltration

height_filtration = HeightFiltration(direction = np.array([1,0]))
im_filtration_he = height_filtration.fit_transform(im_binarized)

height_filtration.plot(im_filtration_he, colorscale="jet")

#Radial Filtration
from gtda.images import RadialFiltration

radial_filtration = RadialFiltration(center=np.array([0, 6]))
im_filtration_ra = radial_filtration.fit_transform(im_binarized)

radial_filtration.plot(im_filtration_ra, colorscale="jet")

#density filtration
from gtda.images import DensityFiltration

density_filtration = DensityFiltration()
im_filtration_de = density_filtration.fit_transform(im_binarized)

density_filtration.plot(im_filtration_de, colorscale="jet")

#dilation filtration
from gtda.images import DilationFiltration

dilation_filtration = DilationFiltration()
im_filtration_di = dilation_filtration.fit_transform(im_binarized)

dilation_filtration.plot(im_filtration_di, colorscale="jet")

#erosion filtration
from gtda.images import ErosionFiltration

erosion_filtration = ErosionFiltration()
im_filtration_er = erosion_filtration.fit_transform(im_binarized)

erosion_filtration.plot(im_filtration_er, colorscale="jet")

#signed distance filtration
from gtda.images import SignedDistanceFiltration

signeddistance_filtration = SignedDistanceFiltration()
im_filtration_si = signeddistance_filtration.fit_transform(im_binarized)

signeddistance_filtration.plot(im_filtration_si, colorscale="jet")

