### Early IoT Botnet Detection Through Big Data Analytics

#### Team Members:
1-M A Moyeen - 40190527 </br>
2-Mohammad Reza Navazani - 40111411

#### Abstract: 
Abstract: Despite the rise of the Internet of Things (IoT) promises smart living for the inhabitants of this universe, the increase in the number of botnet attacks challenges their benefits. Botnets are malware-infected devices and mainly remain under the control of the hacker. The compromised IoT devices acting as botnet are not only an attacker itself but also can spread infections among other internet-connected devices. In most cases, the infected devices never know about their contamination and they also participate in the propagation of such infections. Therefore, early detection of botnets could allow us to isolate infected devices and take measures so that infections do not spread out. With the analysis of time series IoT data using Big Data Analytics techniques, it is possible to reveal the botnet patterns and detect their mutated forms. In this project, we propose a novel solution to detect IoT botnets and the mutated forms by analyzing IoT time-series data with semi-supervised learning techniques and compare their performance with existing solutions.
 
#### Introduction: 
In the current era of technology, smart city applications; such as eHealthcare, smart power control, smart traffic control e.t.c. are promising a better lifestyle for the dwellers of this world. To facilitate the smart city, a vast number of IoT devices are being added every day. The botnet, a network of personal computers that are infected by harmful software, controlled without user intention which sends spam messages, or can trigger Distributed Denial of Service Attack (DDoS), has its own behaviour on the network during lifetime after infection and before the attack. As the IoT devices are resource-constrained often, therefore, they are not always equipped with heavy security algorithms and may not detect unusual devices or traffic behaviour. Thus, IoT devices as personal devices can face the challenges of acting as a botnet without even knowing about whom they work for. As a botnet, they can not only hamper the normal network activity but also are more capable of shutting down a whole network.
In most cases, IoT botnet attacks remain hidden and cannot become visible to the IoT devices. Early predictions about an attack can save billions of dollars or even can someones save a life. The only way to detect such botnet attacks can be with attack knowledge from the victim. With the invention of the modern machine learning techniques, it is mostly possible to infer future phenomena by analyzing time-series data.
Some existing solutions already have proposed machine learning-based IoT botnet detections. N-BaIoT uses deep autoencoders to detect unwanted traffic from the affected IoT devices and can detect Mirai and BASHLITE botnet attacks. However, this work is suitable for offline data analysis. Also, botnet attacks are not only identified in data traffic but also their pattern can be found in control traffic. Therefore, We propose a novel solution that at first analyzes network control traffic and tries to infer unusual behaviour in it. If it can find then immediately it isolates the infected devices and stops propagation of traffic to and from those devices. Otherwise, it looks for unusual patterns in the data traffic.

#### Related Works:
There is a similar study called “Timely Detection and Mitigation of Stealthy DDoS Attacks via IoT Networks” by “Keval Doshi, Yasin Yilmaz, and Suleyman Uludag”. This study brought an anomaly based Intrusion Detection System (IDS) which has this feature of timely detecting and mitigating the emerging type of DDoS attacks. This IDS’s is able to detect low attack size DDoS attacks.
 Also in “A Deep Learning Approach Against Botnet Attacks to Reduce the Interference Problem of IoT” by “Pramathesh Majumdar, Archana Singh, Ayushi Pandey and Pratibha Chaudhary”  they implemented a Restricted Boltzmann Machine (RBM) algorithm of deep learning approach to detect botnet attack patterns in IoT through training algorithm to prevent the botnet attacks on IoT devices, thus the interference problem in the network has reduced.

#### Materials and Methods
The dataset which we are planning to work on is [N-BaIoT dataset](https://www.kaggle.com/mkashifn/nbaiot-dataset),  "Detection of IoT Botnet Attacks" which represents real traffic data gathered from 9 commercial IOT devices authentically infected by Mirai and BASHLITE which are malwares that perform DDoS attacks. This 2GB dataset was donated on 2018-03-19, it is Multivariate and Sequential and includes 7062606 number of instances under 115 attributes. It contains both ‘benign’ and ‘Malicious’ traffic data. the malicious data can be divided into 10 attacks carried by 2 botnets. The dataset classification is like 10 classes of attacks, plus 1 class of 'benign'. Benign represents normal network traffic data. By analyzing benign and Malicious traffic through anomaly detection techniques we are planning to recognize the infected devices traffic pattern in terms of preventing botnet attack before it starts. 

We are planning to implement a Classification and a Clustering technique in terms of detecting anomaly on time series Big Data sets. For clustering we are planning to use K-means algorithm. This will give us a more clear dataset view, through categorizing data in terms of having the same feature. As a classification we intend to use Random Forest algorithm which is combination of many decision trees used for feature ranking to predict which dataset features have more weight and contribute the most in the malicious’ one.

##### K-means
One of the popular unsupervised machine learning algorithms is K-Means clustering. It is aiming for finding groups in dataset, lets say when it finds number of groups/clusters and shoed by the variable K. K-Means algorithm assign all data point to the nearest cluster in a spiral way step by step based on the features. Depend on different distance metrics in each loop the algorithm will allocate each data point to the nearest cluster most of the time it is Euclidean distance result in training data centroids of K clusters and the labels. Here is the scenario once we found the groups, by introducing the new point we can assign it to a group. After loading the dataset in a DataFrame format we take each attribute as a feature and we transform each feature on the DataFrame into FeatureVector. So it is prepared to use in machine learning algorithm. Then we can create our k-means model through defining clusters, feature columns and output prediction column and we use the model to detect the new data clusters/category/group. We consider the ‘benign’ as a normal class therefor ‘malicious’ data will categories as attack classes. We hope that by identifying different pattern as attack and normal classes gain this ability to analyse the live network data in terms of recognizing anomaly which represent infected devices 

##### Random Forest
We also planning to implement Random forest as a classification algorithm. We know that Decision tree is a popular method in machine learning tasks when it comes to classification and regression. If we are looking for responding sequential questions as this model behave like “if then else” pattern we will get a result in the end. By the way we need to consider the risk of overfitting in terms of working with decision trees. Here is the reason we use Random Forest to overcome this risk. The way it performs is like combining many decision trees in order to reduce the overfitting risk. After loading the dataset we are planning to set labeled the data as two classes creditable and not creditable. Which is on the record’s attributes. We will identify the dataset attributes as features. As we need to use these features in our machine learning algorithms we will add them to the DataFram as a vector. Then we will add label column and by building our Random Forest Model we intend to train it with our data with spark pipeline. We will use the model to classify the new data. We hope that this model will us recognize the ‘malicious’ anomaly.

##### Data Pre-Processing
In initial steps we will clean and organize the dataset. This will delete that kind of data which is incomplete (if this kind of data presents) and therefore we will provide more accurate and reliable results


##### Dataset Attribute Information(This part copied from the dataset description):
“ The following describes each of the features headers:</br>
Stream aggregation:</br>
H: Stats summarizing the recent traffic from this packet's host (IP)</br>
HH: Stats summarizing the recent traffic going from this packet's host (IP) to the packet's destination host.</br>
HpHp: Stats summarizing the recent traffic going from this packet's host+port (IP) to the packet's destination host+port. </br>
HH_jit: Stats summarizing the jitter of the traffic going from this packet's host (IP) to the packet's destination host.</br>
Time-frame (The decay factor Lambda used in the damped window):</br>
How much recent history of the stream is capture in these statistics</br>
L5, L3, L1, …</br>
The statistics extracted from the packet stream:</br>
weight: The weight of the stream (can be viewed as the number of items observed in recent history)</br>
mean: …</br>
std: …</br>
radius: The root squared sum of the two streams' variances</br>
magnitude: The root squared sum of the two streams' means</br>
cov: an approximated covariance between two streams</br>
pcc: an approximated covariance between two streams ”

