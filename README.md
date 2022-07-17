# TDA_Facial_Expression_Recognition
Supplement for Odyssey Programme - Topological Data Analysis for Facial Expression Recognition















# TDA-Facial-Emotion-Recognition---Machine-Learning

发现BUG，监修中，勿用！

jaffe为本项目的数据集 （由@极市平台 在知乎上提供的资源 https://zhuanlan.zhihu.com/p/31378836）

The Japanese Female Facial Expression (JAFFE) Database

链接:https://pan.baidu.com/s/1jnxDAGe9UBM3gj_rR2mrvA  密码:ixtu

The database contains 213 images of 7 facial expressions (6 basic facial expressions + 1 neutral) posed by 10 Japanese female models. 
Each image has been rated on 6 emotion adjectives by 60 Japanese subjects. 
The database was planned and assembled by Michael Lyons, Miyuki Kamachi, and Jiro Gyoba. We thank Reiko Kubota for her help as a research assistant.
The photos were taken at the Psychology Department in Kyushu University.

该数据库是由10位日本女性在实验环境下根据指示做出各种表情，再由照相机拍摄获取的人脸表情图像。
整个数据库一共有213张图像，10个人，全部都是女性，每个人做出7种表情，这7种表情分别是： sad, happy, angry, disgust, surprise, fear, neutral. 
每个人为一组，每一组都含有7种表情，每种表情大概有3,4张样图。
</br>

使用说明：

1. 通过 face_landmark_success 程序读取图片并提取出人脸的81或68个特征点

2. 通过 获得数据集-bottleneck 程序获得ML所需的数据集，包括了60 * 15的数据（有时间我将把其扩展至180 * 15），Bottleneck Distance的基准数据为每个数据集中的“neutral face”

3. 通过 决策树+KNN-ML 程序使用决策树算法进行测试，准确率接近100%（更：发现为BUG），但KNN算法准确率极低（仍在探索中。。。）
