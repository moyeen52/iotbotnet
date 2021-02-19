## Early IoT Botnet Detection Through Big Data Analytics

### Abstract
 Botnet, a network of personal computers which are infected by harmful software, controlled without user intention which sends spam messages has its own behavior on the network during life time after infection and before the attack. Finding this pattern will give us this opportunity to find the infected devices before attack. Through analyzing a year old dataset of 10 attacks carried by 2 botnets we wish to provide the ability to recognize the infected devices before the attack starts. We are planning to implement this approach using neural network technique, clarification and clustering technique.
### Introduction
 Though the rise of the Internet of Things (IoT) promises smart living for the inhabitants of this universe, the increase in the number of botnet attacks challenges their benefits. Botnets are malware-infected devices and mainly remain under the control of the hacker without the owner's knowledge. The compromised IoT devices acting as botnets are not only an attacker itself but also can spread infections among other internet-connected devices which is also categorized as aggressive action. In most cases, the infected devices never know about their contamination and they also participate in the propagation of such infections without having any intention. To avoid putting devices under this situation we need to recognize the infected network’s devices before attack. Therefore, early detection of botnets could allow us to isolate infected devices and take measures so that infections do not spread out. This malicious software have its own activity on the network. These types of activities are known as their behavior. By recognizing this behavior we can define a pattern for them. With the analysis of time series IoT data using Big Data Analytics techniques, it is possible to reveal the botnet patterns and detect their mutated forms. In this project, we are going to propose a novel solution to detect IoT botnets and the mutated forms by analyzing IoT time series data with sem-supervised learning techniques and compare their performance. As related work, “Son Nguyen” and “Anthony Park” has done a research about “A Comparison of Machine Learning Algorithms of Big Data for Time Series Forecasting Using Python” in the book of Open Source Software for Statistical Analysis of Big Data (pp.197-218) which compared different algorithms performance on the same type of data. Also in different papers like “Keval Doshi, Yasin Yilmaz, and Suleyman Uludag” in “Timely Detection and Mitigation of Stealthy DDoS Attacks via IoT Networks” used the same dataset to recognize and mitigate bot’s attack.
### Materials and Methods
The dataset which we are planning to work on is N-BaIoT dataset,  Detection of IoT Botnet Attacks which represents real traffic data gathered from 9 commercial IOT devices authentically infected by Mirai and BASHLITE which are malwares that perform DDoS attacks. This 2GB dataset is donated on 2018-03-19, it is Multivariate and Sequential and include 7062606 number of instances under 115 attributes. It contains both ‘benign’ and ‘Malicious’ traffic data. the malicious data can be divided into 10 attacks carried by 2 botnets. The dataset classification is like 10 classes of attacks, plus 1 class of 'benign'. Benign represent normal network traffic data. By analyzing benign and Malicious traffic through anomaly detection techniques we are planning to recognize the infected devices traffic pattern in terms of prevent botnet attack before it starts. 
We are deciding to implement a Neural Network technique and two Classification and Clustering technique from the course in terms of detecting anomaly on time series Big Data sets. For Neural Network we intend to implement long short-term memory LSTM-Based time series anomaly detection using Apache spark. In terms of implementing clustering technique we intend to use K-means algorithm using MLIB with Apache Spark on time series. One of the classification algorithms that we intend to work on using apache spark is decision tree. At the moment we are not completely sure about the libraries we intend to use in different techniques. 

 
