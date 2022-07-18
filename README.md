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
\
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
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
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
\
    
  
  



