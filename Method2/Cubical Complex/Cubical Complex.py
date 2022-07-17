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

#Binarization - Open eye Close mouth
from gtda.images import Binarizer
im_NE = im[None, :, :]
binarizer = Binarizer(threshold=0.4)

im_binarized = binarizer.fit_transform(im_NE)
binarizer.plot(im_binarized,colorscale="blues")

#Binarization - Close eye Close mouth
im_binarized_cc = im_binarized
for i in range(64):
  for j in range(64):
    if 20 <=j<=44 and 30<=i<=33:
      im_binarized_cc[0][i][j] = True
    if i == 31 and 37<=j<=43:
        im_binarized_cc[0][i][j] = False
    if i == 31 and 21<=j<=28:
      im_binarized_cc[0][i][j] = False
binarizer.plot(im_binarized_cc,colorscale="blues")

#Binarization - Open eye Open mouth
im_binarized_oo = im_binarized
for i in range(64):
  for j in range(64):
        if 28<=j<=40 and 48<=i<=55:
            im_binarized_oo[0][i][j] = True
        if 32<=j<=35:
            im_binarized_oo[0][48][j] = False
            im_binarized_oo[0][53][j] = False
        if 30<=j<=37:
            im_binarized_oo[0][49][j] = False
            im_binarized_oo[0][52][j] = False
        if 29<=j<=38:
            im_binarized_oo[0][50][j] = False
            im_binarized_oo[0][51][j] = False

im_binarized_oo[0][49][28] = False
im_binarized_oo[0][49][29] = False

binarizer.plot(im_binarized_oo,colorscale="blues")



#Height filtration
from gtda.images import HeightFiltration

height_filtration = HeightFiltration(direction = np.array([1,0]))
im_filtration = height_filtration.fit_transform(im_binarized)

height_filtration.plot(im_filtration, colorscale="jet")

#Cubical Complex
from gtda.homology import CubicalPersistence

cubical_persistence = CubicalPersistence(n_jobs=-1)
im_cubical = cubical_persistence.fit_transform(im_filtration)

cubical_persistence.plot(im_cubical)

#Rescale
from gtda.diagrams import Scaler

scaler = Scaler()
im_scaled = scaler.fit_transform(im_cubical)
scaler.plot(im_scaled)