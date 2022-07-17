import dlib
import numpy as np
import cv2
import dlib
import gudhi as gd
import gudhi.representations
import matplotlib.pyplot as plt
import numpy.random as rd
import matplotlib.pyplot as plt
import random

#FACIAL LANDMARKS
def facial_landmark(data_path):
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('/content/drive/MyDrive/Colab_Notebook/shape_predictor_81_face_landmarks.dat')

  # cv2 read the picture
  img = cv2.imread(data_path)
  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  rects = detector(img_gray, 0)
  dataset=[]
  for i in range(len(rects)):
      landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
      for idx, point in enumerate(landmarks):
          pos =[point[0, 0], point[0, 1]]
          dataset.append(pos)
  return dataset[:68]


#GET DATA
whole_data = []
for i in range(1,11):
  path = '/content/drive/MyDrive/data/AN.'+str(i)+'.tiff'
  data = facial_landmark(path)
  whole_data.append(data)
for i in range(1,11):
  path = '/content/drive/MyDrive/data/DI.'+str(i)+'.tiff'
  data = facial_landmark(path)
  whole_data.append(data)
for i in range(1,11):
  path = '/content/drive/MyDrive/data/FE.'+str(i)+'.tiff'
  data = facial_landmark(path)
  whole_data.append(data)
for i in range(1,11):
  path = '/content/drive/MyDrive/data/HA.'+str(i)+'.tiff'
  data = facial_landmark(path)
  whole_data.append(data)
for i in range(1,11):
  path = '/content/drive/MyDrive/data/NE.'+str(i)+'.tiff'
  data = facial_landmark(path)
  whole_data.append(data)
for i in range(1,11):
  path = '/content/drive/MyDrive/data/SA.'+str(i)+'.tiff'
  data = facial_landmark(path)
  whole_data.append(data)
for i in range(1,11):
  path = '/content/drive/MyDrive/data/SU.'+str(i)+'.tiff'
  data = facial_landmark(path)
  whole_data.append(data)
whole_data=np.array(whole_data).reshape(70,68,2)
labels_basic = [0]*10+[1]*10+[2]*10+[3]*10+[4]*10+[5]*10+[6]*10
labels_basic = np.array(labels_basic)


#VR COMPLEX
from gtda.homology import VietorisRipsPersistence

# Track connected components, loops
homology_dimensions = [0, 1]

# Collapse edges to speed up H2 persistence calculation!
persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=6,
    collapse_edges=True,
)

diagrams_basic = persistence.fit_transform(whole_data)


#PIPELINE
from sklearn.pipeline import make_union
from gtda.diagrams import NumberOfPoints
from gtda.diagrams import Amplitude
from gtda.diagrams import PersistenceEntropy

# Select a variety of metrics to calculate amplitudes
metrics = [
    {"metric": metric}
    for metric in ["bottleneck", "wasserstein", "landscape", "persistence_image"]
]

# Concatenate to generate 3 + 3 + (4 x 3) = 18 topological features
feature_union = make_union(
    PersistenceEntropy(normalize=True),
    NumberOfPoints(n_jobs=-1),
    *[Amplitude(**metric, n_jobs=-1) for metric in metrics]
)

#RFC --
from sklearn.ensemble import RandomForestClassifier
from gtda.pipeline import Pipeline
pipe = Pipeline(
    [
        ("features", feature_union),
        ("rf", RandomForestClassifier(oob_score=True, random_state=22)),
    ]
)
pipe.fit(diagrams_basic, labels_basic)
print(pipe["rf"].oob_score_)
