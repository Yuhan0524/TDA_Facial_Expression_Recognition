# TDA_Facial_Expression_Recognition
Supplement for Odyssey Programme - Topological Data Analysis for Facial Expression Recognition

Acknowledgements:
----

- This is the supplement for my summer research program with the topic "Topological Data Analysis for Facial Expression Recognition"

- I would like to extend my gratitude for NTU SPMS, the Odyssey Programme for this wonderful experience, Prof Xia and Mr Wee for their guidance.

- The following picture is my research poster for this program


![images](https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/blob/master/poster_final.jpg)


Abstract
----
- We introduce 2 ways of using Topological Data Analysis(TDA) to generate topological features of 7 different facial expressions including angry, disgust, fear, happy, neutral, sad, and surprise. 

- We apply the TDA methods to JAFFE and FER2013 datasets and try to conduct a dimensionality reduction of features for machine learning through TDA. 

Motivation
----
- I did a presentation about PCA and eigenface for face recognition during last semster, so I'm quite interested in Computer Vision. 
  When considering for the project topic of Odyssey, I thought of CV topics directly.
  Because human face have topological features, TDA may be applicable for getting features.(Actually the result for my project is not good - I think some adjustment can be done - more than what I can handle currently anyway QAQ)

- With the impression of two references, I have some ideas for this project topic
  - https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/blob/master/References/TDA-FER-Visualization.pdf [1] A. Garin and G. Tauzin, "A Topological "Reading" Lesson: Classification of MNIST using TDA," 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA), 2019, pp. 1551-1556, doi: 10.1109/ICMLA.2019.00256.
  - https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/blob/master/References/TDA-MNIST.pdf [2] H. Elhamdadi, S. Canavan and P. Rosen, "AffectiveTDA: Using Topological Data Analysis to Improve Analysis and Explainability in Affective Computing," in IEEE Transactions on Visualization and Computer Graphics, vol. 28, no. 1, pp. 769-779, Jan. 2022, doi: 10.1109/TVCG.2021.3114784.
  
Main Idea
----
I conduct 2 methods. The main idea is the same and the main different is the input.
At first, I thought of the first method with an input of 68 facial landmarks. While, then I found that if the image containing hands/side face etc will strongly affect the results of landmark detection. So I tried the second method which directly deal with pixels input with the impression of paper https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/blob/master/References/TDA-MNIST.pdf. Finally, get a pipeline as a summary for the whole methods.

Dataset - https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/blob/master/dataset
----
(cz the dataset files are too large for github - plz go to mircrosoft drive to get that XD)

- JAFFE dataset - The database contains 213 images of 7 facial expressions (6 basic facial expressions + 1 neutral) posed by 10 Japanese female models. Each image has been rated on 6 emotion adjectives by 60 Japanese subjects. The database was planned and assembled by Michael Lyons, Miyuki Kamachi, and Jiro Gyoba. We thank Reiko Kubota for her help as a research assistant. The photos were taken at the Psychology Department in Kyushu University.

- FER2013 dataset .csv - Kaggle Competition https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge

- All Train and Private test images in FER2013, I change them from pixels to images - *Note: for train images, 0-13505 png, 13506-13963 jpg, 13964-14460 jpeg, 14461- png

Method 1 - https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/tree/master/Method1
----
- Input
  - the JAFFE dataset has better result for method 1
  - Using images (*not pixels) as input, use Dlib package to do facial landmark detection
    - tutorial - https://learnopencv.com/facial-landmark-detection/
 
```python
import numpy as np
import cv2
import dlib
import gudhi as gd
import gudhi.representations
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')

def get_data(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY
    rects = detector(img_gray, 0)
    dataset=[]
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
        for idx, point in enumerate(landmarks[:68]):
            pos =[point[0, 0], point[0, 1]]
            dataset.append(pos)
            cv2.circle(img, pos, 1, color=(0, 255, 0))
            font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1,cv2.LINE_AA)
        cv2.namedWindow("img", 2)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.imwrite("1_landmark.jpg",img)
    return dataset[:68]
path = "1.tiff"
dataset = get_data(path)
print(dataset)
```
With the coordinates of facial landmarks as output. I will be able to use it as input for VR complex

Vietoris-Rips Complex
----
Vietoris-Rips complex 𝑉𝑅(𝑋,𝜀) with parameter 𝜀, is a simplicial complex with vertex set 𝑋, in which {𝑥_0,𝑥_1,…,𝑥_𝑘} spans a 𝒌-simplex iff 𝑑(𝑥_𝑖,𝑥_𝑗)≤𝜀 for all 0≤𝑖,𝑗≤𝑘.

- Simple Introduction for TDA

  - Simply speaking, we use TDA to obtain the topological features of datas.

  - Imagine a point cloud, with each points as center, then draw a circle with diameter 𝜀 fro each center. As diameter increasing, the circles will also increase. If two circle overlap each other, we will connect their centers. As this going on, we find there will be simplex appear in the diagram.
![image](https://user-images.githubusercontent.com/102588357/179451525-d5b047f4-a002-4caa-b38e-042b262fc9b1.png)
[1] Yen, P.T., & Cheong, S.A. (2021). Using Topological Data Analysis (TDA) and Persistent Homology to Analyze the Stock Markets in Singapore and Taiwan. Frontiers in Physics.

  - What we want to do is that, as 𝜀 increasing, we get the Betti number of the diagram and use persistence barcode or diagram to visualize.

  - For example
  ![image](https://user-images.githubusercontent.com/102588357/179451698-a92a1bcb-ede0-40d8-8f56-655c170fc621.png)
  [2] C. -C. Wong and C. -M. Vong, &quot;Persistent Homology based Graph Convolution Network for Fine-grained 3D Shape Segmentation,&quot; 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 7078-7087, doi: 10.1109/ICCV48922.
  
  - The LHS is the persistence barcode, with the red one for 0-dimension Betti number and blue one for 1-dimension. Each bar starts at a time called 'birth time' meaning this x-dim appears at this time, and a 'death time' meaning the x-dim disappears here. Similarly, we can use birth time as x-axis and death time as y-axis to map the persistence barcode to persistence diagram. These are the main ideas of TDA.

Through a VR Complex DEMO - see https://github.com/Yuhan0524/TDA-Demo_RipsComplex

We can figure out the tendency of the changing topological features of the face through VR Complex
![image](https://user-images.githubusercontent.com/102588357/179450499-05bfb674-7c0a-4723-a031-cffc608a14b4.png)

Then we use Perisistence Barcode and Diagram to give summary for persistent homology of the topological features of the Vietoris-Rips filltartion for points.

Diagram - https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/blob/master/Method1/Vietoris-Rips%20Complex/VR_Persistent_Diagram_Gtda.py
```python
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams
data = [[198, 408], [202, 463], [212, 517], [226, 570], [248, 620], [280, 662], [320, 698], [364, 726], [411, 733],
        [459, 726], [503, 698], [543, 663], [575, 620], [595, 570], [607, 516], [614, 461], [616, 405], [229, 355],
        [255, 327], [294, 317], [334, 323], [370, 341], [442, 339], [479, 320], [519, 314], [558, 323], [584, 351],
        [406, 401], [407, 442], [407, 482], [407, 523], [370, 553], [389, 558], [408, 563], [428, 557], [446, 552],
        [270, 408], [295, 391], [327, 392], [353, 418], [324, 425], [292, 424], [461, 417], [486, 391], [518, 389],
        [543, 406], [522, 422], [489, 424], [339, 624], [366, 611], [391, 601], [408, 607], [425, 601], [451, 613],
        [480, 626], [451, 648], [426, 658], [408, 660], [389, 658], [365, 647], [353, 625], [391, 625], [408, 627],
        [425, 625], [466, 627], [425, 626], [408, 628], [390, 625]]
data = np.array(data).reshape(1,68,2)
VR_persistence = VietorisRipsPersistence(n_jobs=-1)
PD = VR_persistence.fit_transform(data)
#plot_diagram(PD[0])
#if you use this in google colab/Jupyter etc, can directly run by plot_diagram(PD[0]) without using the following codes
plt.title("Persistence Diagram")
plot_diagrams(PD[0])
plt.tight_layout()
plt.show()
```
Barcode - https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/blob/master/Method1/Vietoris-Rips%20Complex/VR_persistentce_barcode_dim1.py
```python
import numpy as np
import numpy.random as rd
import gudhi as gd
import gudhi.representations
import matplotlib.pyplot as plt
import random
rd.seed(1)

dgms_normal = []
normal_pcs = [[128, 120], [128, 147], [132, 174], [138, 200], [147, 226], [160, 250], [176, 271], [195, 287], [221, 292], [247, 288], [270, 274], [289, 255], [305, 231], [316, 206], [322, 178], [327, 149], [328, 120], [144, 104], [156, 94], [173, 92], [191, 96], [207, 103], [238, 103], [256, 95], [274, 90], [293, 92], [307, 102], [222, 120], [221, 138], [220, 156], [219, 174], [200, 185], [210, 190], [221, 193], [232, 190], [243, 185], [162, 120], [172, 114], [186, 114], [197, 123], [185, 125], [172, 125], [252, 122], [263, 113], [276, 112], [287, 119], [277, 124], [264, 124], [186, 225], [198, 218], [211, 214], [221, 217], [231, 215], [245, 219], [260, 226], [245, 238], [231, 243], [220, 244], [210, 243], [197, 238], [192, 226], [211, 224], [221, 225], [231, 224], [254, 226], [231, 230], [220, 231], [210, 230]]
rips = gudhi.RipsComplex(points=normal_pcs).create_simplex_tree(max_dimension=2)
rips.compute_persistence()
dgms_normal.append(rips.persistence_intervals_in_dimension(1))
gd.plot_persistence_barcode(dgms_normal[0])
plt.show()
```
Vectorization - https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/tree/master/Method1/Vectorization
----

Since we can not use a diagram as ML input. We have to get an array as features.

However, even though the data size are the same, persistence diagram and barcode may still give different number of points or bars. So we can not use the data of diagrams as input for ML.

Facing this problem, we use several vectorization methods, to change diagrams and barcodes into array.

(The vectorization methods will be mentioned during the part of method 2)

After vectorization, we visualize it through a HTML website - refer to https://github.com/Yuhan0524/TDA_Visualization

Pipeline
----
The whole pipeline for method 1 shown here - https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/blob/master/Method1/METHOD_1.py

```python
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
```
We use the whole pipeline and finally can get 18 features for each diagram (result not good ...)

Method 2 - https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/tree/master/Method2
----

Because the dlib package is not useful for some images (with hand, sideface etc), we try to find some way that can apply for most of the images, or directly for pixels or vixels input.

Then I read the paper "A Topological "Reading" Lesson: Classification of MNIST using TDA" and find a certain way to do so.

Input
----

For FER2013, the input will be the original 48**48 pixels directly.
For JAFFE, even though 256**256 pixels is OKAY for TDA, after binarization, it contains plenty of noise points. So I reduce dimension to a 64**64 one.

- We first change it to a greyscale image
```python
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
```
- Change to 64**64
```python
#change to 64*64 avoid noise
im = []
for i in range(0,256,4):
  for j in range(0,256,4):
    im.append(im_ori[i][j])
im = np.array(im).reshape(64,64)

plt.imshow(im)
plt.show()
```
- Binarization ℬ:𝐼⊂𝑍^𝑑→\{1,0\}
```python
#Binarization
from gtda.images import Binarizer
im_NE = im[None, :, :]
binarizer = Binarizer(threshold=0.4)

im_binarized = binarizer.fit_transform(im_NE)
print(im_binarized)
binarizer.plot(im_binarized,colorscale="blues")
```

Filtration
----
Binarization is one of the filtrations. Other than binarization, we still have some other ways of filtration including:
- Height Filtration
- Radial Filtration
- Density Filtration
- Dilation Filtration
- Erosion Filtration
- Signed Distance Filtration

We can do filtration through Giotto-TDA - https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/blob/master/Method2/Filtration/Filtration.py

For example, height filtration:
```python
#Height filtration
from gtda.images import HeightFiltration

height_filtration = HeightFiltration(direction = np.array([1,0]))
im_filtration_he = height_filtration.fit_transform(im_binarized)

height_filtration.plot(im_filtration_he, colorscale="jet")
```

![image](https://user-images.githubusercontent.com/102588357/179461497-bcdc1834-816c-4d16-b05b-9fcec77c4a7f.png)

Cubical Complex
----
- Cubical Complex shares similar things with VR Complex. The main difference is that Cubical Complex spans 𝒌-cube instead of 𝑘-simplex, so it’s better for pixels input.
- Rescale the diagram before vectorization
```python
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
```

Vectorization
----
Same reason as mentioned in method 1
![image](https://user-images.githubusercontent.com/102588357/179462647-7583321e-02fd-4ed0-b522-8d24c4e37670.png)

Pipeline
----
```python
from sklearn.pipeline import make_pipeline, make_union
from gtda.images import HeightFiltration, RadialFiltration,Binarizer
from gtda.diagrams import Amplitude, Scaler, PersistenceEntropy
from gtda.homology import CubicalPersistence
import numpy as np

direction_list = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

center_list = [
    [120, 60],
    [60, 120],
    [120, 120],
    [180, 120],
    [120, 180],
    [60, 60],
    [60, 180],
    [180, 60],
    [180, 180],
]

# Creating a list of all filtration transformer, we will be applying
filtration_list = (
    [
        HeightFiltration(direction=np.array(direction), n_jobs=-1)
        for direction in direction_list
    ]
    + [RadialFiltration(center=np.array(center), n_jobs=-1) for center in center_list]
)

# Creating the diagram generation pipeline
diagram_steps = [
    [
        Binarizer(threshold=0.4, n_jobs=-1),
        filtration,
        CubicalPersistence(n_jobs=-1),
        Scaler(n_jobs=-1),
    ]
    for filtration in filtration_list
]

# Listing all metrics we want to use to extract diagram amplitudes
metric_list = [
    {"metric": "bottleneck", "metric_params": {}},
    {"metric": "wasserstein", "metric_params": {"p": 1}},
    {"metric": "wasserstein", "metric_params": {"p": 2}},
    {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 1, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 2, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 1, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 2, "n_bins": 100}},
    {"metric": "betti", "metric_params": {"p": 1, "n_bins": 100}},
    {"metric": "betti", "metric_params": {"p": 2, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 1, "sigma": 1.6, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 1, "sigma": 3.2, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 2, "sigma": 1.6, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 2, "sigma": 3.2, "n_bins": 100}},
]

#
feature_union = make_union(
    *[PersistenceEntropy(nan_fill_value=-1)]
    + [Amplitude(**metric, n_jobs=-1) for metric in metric_list]
)

tda_union = make_union(
    *[make_pipeline(*diagram_step, feature_union) for diagram_step in diagram_steps],
    n_jobs=-1
)

#Call Pipeline
#X_train_tda = tda_union.fit_transform(X_train)
```

#PIPELINE
the whole idea is like the chart shown below

![image](https://user-images.githubusercontent.com/102588357/179463156-4023e5af-50a5-4ec0-a84b-a78843718284.png)

However, I try to do a CNN machine learning with these features, the results are not good anyway.

With the same CNN model, the original FER2013 48**48 pixels input can get an accuracy of 58%-60% (not the best one)

the data after TDA can only get an accuracy around 43%

I think maybe because of the complexity topological features of faces

The pipeline (especially the vectorization part) need improvement

Put my CNN here as a reference - I refer to this notebook - https://www.kaggle.com/code/alpertemel/fer2013-with-keras/notebook 

```python
import tensorflow as tf
in_1 = tf.keras.layers.Input((48,48,1),name='in_1')
model = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(in_1)
model = tf.keras.layers.MaxPool2D(pool_size=(5,5), strides=(2, 2))(model)

model=tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(model)
model=tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(model)
model=tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2))(model)

model=tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(model)
model=tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(model)
model=tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2))(model)

model=tf.keras.layers.Flatten()(model)

model=tf.keras.layers.Dense(1024, activation='relu')(model)
model=tf.keras.layers.Dropout(0.2)(model)
model=tf.keras.layers.Dense(1024, activation='relu')(model)
model=tf.keras.layers.Dropout(0.2)(model)

model=tf.keras.layers.Dense(7, activation='softmax')(model)

#------------------------------------------------------------------------------------------------------------------
in_2 = tf.keras.layers.Input((30,100,1),name='in_2')
model_2=tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(in_2)

model_2=tf.keras.layers.MaxPool2D(pool_size=(5,5), strides=(2, 2))(model_2)

model_2=tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(model_2)
model_2=tf.keras.layers.Conv2D(64, (2, 2), activation='relu')(model_2)
model_2=tf.keras.layers.MaxPool2D(pool_size=(1,1), strides=(2, 2))(model_2)

model_2=tf.keras.layers.Flatten()(model_2)

model_2=tf.keras.layers.Dense(1024, activation='relu')(model_2)
model_2=tf.keras.layers.Dropout(0.5)(model_2)
model_2=tf.keras.layers.Dense(1024, activation='relu')(model_2)
model_2=tf.keras.layers.Dropout(0.5)(model_2)

model_2=tf.keras.layers.Dense(7, activation='softmax')(model_2)


merged = tf.keras.layers.Concatenate()([model, model_2])
output = tf.keras.layers.Dense(7, activation='softmax')(merged)

model_final = tf.keras.Model(inputs=[in_1,in_2],outputs=[output])
model_final.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy',metrics=['accuracy'])
#model_final.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'],loss_weights=[1,0.2])

hist = model_final.fit(
    {"in_1": train_ori,"in_2": train_data},train_labels_small,
    validation_data=({"in_1":test_ori, "in_2":private_data},private_labels_small),
    epochs=30,
    batch_size=256)
```
Visualization
![images](https://github.com/Yuhan0524/TDA_Facial_Expression_Recognition/blob/master/CNN_Model.png)

Conclusion and Future Work
----
- TDA methods which shown by pipeline allow us to obtain topological features of different expressions and vectorize them into arrays. 
- However, due to the complexity of the faces, we still need to adjust methods to obtain more details.


-FIN-
