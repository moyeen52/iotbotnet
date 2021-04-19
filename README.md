### Early IoT Botnet Detection Through Big Data Analytics

#### Team Members:
1-M A Moyeen - 40190527 </br>
2-Mohammad Reza Navazani - 40111411

#### Abstract: 
Despite the rise of the Internet of Things (IoT) promises smart living for the inhabitants of this universe, the increase in the number of botnet attacks challenges their benefits. Botnets are malware-infected devices and mainly remain under the control of the hacker. The compromised IoT devices acting as botnets are not only an attacker itself but also can spread infections among other internet-connected devices. In most cases, the infected devices never know about their contamination and they also participate in the propagation of such infections. Therefore, early detection of botnets could allow us to isolate infected devices and take measures so that infections do not spread out. With the analysis of IoT data using machine learning techniques, it is possible to reveal the botnet patterns and detect their mutated forms. In this project, we analyzed the 9 commercial IoT devices generated Mirai and BASHLITE attack traffic using Random Forest, Decision Tree, Factorization Machine, and k-means algorithm. Based on our analysis, we observed that the Random Forest and k-means can accurately identify Mirai and BASHLITE attacks on N-BaIoT dataset.
 
#### 1. Introduction: 

##### 1.1  Context:
The ability of Internetworking among computing devices gives rise to the IoT era. With the proliferation of internet-enabled devices and ambient intelligence [1], IoT devices are attaining the focal point of interest. IoT products are expected to ease the lifestyle of the inhabitants of smart cities, improve features for intelligent automotive [2], provide smart health care facilities, improve safety, and so on [3].  A recent study [4] indicates that there will be more than 64 billion connected objects by 2022. Furthermore, IoT appliances could generate $4T to $11T in economic value by 2025 [4]. 

Although the benefits of IoT are indisputable, due to the addition of new devices at an exponential rate, security issues are also arising quickly. The rise of botnets challenges the benefits of the IoT era.  The botnets are interconnected networked devices infected by malware and act as a bot to perform Distributed Denial-of-Service (DDoS) attacks, data leakage, spam messaging without the knowledge of their infection. The two most deadly botnets are Mirai and BASHLITE. Where BASHLITE (also known as Gafgyt, LizardStresser, Lizkebab, PinkSlip, Qbot and Torlus) was first identified in 2014 launching DDoS attacks using Linux devices running BusyBox [5].  In October 2016, the Mirai botnet did a major damage with DDoS attack to the Dyn DNS infrastructure by commanding 100,000 IoT devices and made many famous  websites, Netflix, Twitter, CNN, Paypal, etc. down for several hours [6]. Most of the IoT devices examined by Kaspersky Lab in 2016 were found to be using default passwords and were easily compromised by Mirai and Hajime malware [7].

##### 1.2 Objectives:
Therefore, the major goal of this project is to detect Mirai and BASHLITE based attacks from IoT device-generated traffic as they are the root cause of major botnet attacks. If botnets can be detected before they cause their damage, it could save billions of dollars and even could save our life. Therefore, the objective of this project is to detect botnets from real network traffic at a considerable time using distributed Machine Learning (ML) algorithms. Moreover, our aim is to analyze the botnet traffic using supervised and unsupervised ML techniques. Therefore, in this project, we analyzed one of the popular IoT botnet datasets called N-BaIoT [8] using Random Forest (RF), Decision Tree (DT), Factorization machines (FM), and k-means clustering techniques. The dataset contains significant samples of Mirai and BASHLITE and is a good representative of real IoT attack traffic because the dataset contains the network traffic from nine commercial IoT devices. 

##### 1.3 Presentation of the problem:
The botnets often mimic the real traffic pattern and leave a small footprint on the network. Since the public release of the source code of Mirai and BASHLITE [5,7], there has been a huge mutation in those attack types. Therefore, the attack traffic pattern is evolving and botnets are mostly hard to detect. With the labelled  data, supervised ML classification can be fruitful to detect already existing attack types and their traffic patterns. While unsupervised ML algorithms can be effective to detect new attack types. Therefore, we incorporated both to analyze the N-BaIoT dataset and recorded their performance. With our analysis, we found that both supervised and unsupervised ML techniques perform better in terms of Mirai and BASHLITE attack classification. Especially, RF and k-means can accurately detect those attack types. Whereas, other two techniques are more likely to suffer from wrong parameter settings.
##### 1.4 Related works:
Already a plethora of research work has been conducted to detect botnets using machine learning techniques. Network based anomaly detection using deep learning technique became the pioneer of application of autoencoders [8]. They extracted statistical features from the nine commercial IoT devices generated traffic and fed those real traffic data to train their models.  Though their empirical evaluation suggests almost zero false positives, they did not evaluate their technique in a distributed environment. Another autoencoder based technique proposed in [9], claims that their work sustains even if the training data is not clean and their technique will not be affected by outliers. Though they have high recall, their precision value is low and are more likely to produce huge false negatives when data is sparse. Due to the incorporation of online processing, the work proposed in [10], can offer low memory requirement anomaly detection. The technique is lightweight, fast, and can scale at multiple IoT devices. However, their technique depends on external packet parser and anomaly detection is mainly based on RMSE.  

A hybrid approach proposed in [11], employs both network-based and host-based anomaly detection and can offer better detection rate with low computational overhead. However, their technique is not distributed and is susceptible to wireless sensor network topology change. In [12], authors propose a CNN based approach that employs PSI graphs to detect malware without preselecting features. Though their accuracy and F-measures are more than 92%, their technique is not implemented in a scalable environment. Another study proposed in [13], brought an anomaly based Intrusion Detection System (IDS) which has this feature of timely detecting and mitigating the emerging type of DDoS attacks. This IDS is able to detect low attack size DDoS attacks. 

Some techniques use Synthetic Minority Oversampling Technique (SMOTE) along with different ML algorithms [14-16]. In [14], the author uses SMOTE along with Support Vector Machine, Naive Bayes, and Multi Layer Perceptron to detect malware. Another SMOTE based technique [15] uses Random Forest to classify android malware and are capable of achieving 99% accuracy. However, their technique is not implemented in a scalable environment. The work proposed in [16]  combines SMOTE with LSTM and can achieve better detection rate. However, their work is not in the IoT based botnet detection domain and their evaluation is not based on a scalable environment.

Detection of cyber attack at close proximity to the IoT devices is required to save smart cities under attack. Therefore, Random Forest based Fog level detection can alert ISP’s and city owners quickly [17]. They used UNSW-NB15 dataset to detect the binary labelled classification and their model is conceptual. Moreover, they did not use any distributed framework (e,g; Apache Spark ) and they plan to use n-fold cross validation in the future extension of their work. Another technique proposed in [18] uses K-NN, Decision Trees, Naive Bayes, Adaboost, Logistic Regression, Random Forest, Linear Support Vector Machine, and Radial Basis Support Vector Machine to detect IoT botnet and uses N-BaIoT dataset. They consider network traffic as an image and train Deep Neural Network model. With  t-SNE visualization they found that the data is separable and highlighted that the major ML techniques perform better on N-BaIoT dataset for this reason. The work proposed in [19] assumes the consumer IoT traffic pattern is different from non-consumer IoT traffic patterns. Their packet level network analysis with five ML techniques including Random Forest and Decision Tree can show almost 99% accuracy in case of DoS attack detection. However, their technique did not mention analysis in a distributed environment and new attack types. In [20], authors have implemented a Restricted Boltzmann Machine (RBM) algorithm of deep learning approach to detect botnet attack patterns in IoT through training algorithms to prevent the botnet attacks on IoT devices, thus the interference problem in the network has reduced.

It is evident from the existing literature that none of the technique focuses on the scalability of their solution and can respond to new and existing attack types with small processing and memory overhead. 

#### Materials and Methods
##### 2.1 Dataset(s):
In this project, we used the famous IoT botnet attack dataset called “N-BaIoT” from Kaggle [xx]. The dataset comprises 8GB of real attack traffic data from the authentically Mirai and BASHLITE infected nine commercial IoT devices namely, Danmini Doorbell, Ecobee Thermostat, Ennio Doorbell, Philips B120N10 Baby Monitor, Provision PT 737E Security Camera, Provision PT 838 Security Camera, Samsung SNH 1011 N Webcam, SimpleHome XCS7 1002 WHT Security Camera, and SimpleHome XCS7 1003 WHT Security Camera. The dataset we used is multivariate, sequential and comprises 7062606  instances of benign, Miria and BASHLITE infected traffic and 115 features. The major attributes of the dataset are described below:
 - H: Stats that summarizes all recent traffic from the packet’s source IP address. 
 - HH: Stats that summarizes recent traffic from packet’s source IP to a particular destination IP address
 - HpHp: Stats that summarizes the traffic movement between packets source IP and port mapping to the destination IP and port mapping.
 - HH_jit: Stats that summarizes the jitter of traffic movement between packet’s source to destination. 
 - Time-frame: This is the decay factor that defines the interval at which packets were collected. 
Besides the aforementioned attributes, the dataset also contains the weight, mean, standard deviation, radius, magnitude, the covariance between streams of packets. 

The dataset is an important contribution in the case of IoT botnet attack analysis. However, it is imbalanced which is shown in the following figure.

![picture alt](https://github.com/moyeen52/iotbotnet/blob/main/figures/class_distribution.png "Imbalance in Dataset")

From the figure it is evident that the benign class of traffic are nearly 8 percent of the total dataset whereas the Mirai samples are nearly 50 percent of the total dataset and BAHLITE data samples are nearly 40 percent.

#### 2.2 Data Pre-Processing
The dataset obtained from the Kaggle contains lots of CSV file naming with different attack and non-attack types and the header column of each CSV file preserves the features name. Also, some of the features' names contain unsupported dot (.) characters. Moreover, the dataset does not have a class label assigned to each row by default. Furthermore, the imbalance in the chosen dataset can cause inaccurate results. Therefore, we had to preprocess our data such that the data becomes labelled, the column name contains no unsupported characters and the dataset becomes balanced. The following figure represents the data preprocessing stages. 

![picture alt](https://github.com/moyeen52/iotbotnet/blob/main/figures/arch.png "Data Processing Stages")

To label our data, we read each CSV file and based on the name we labelled those CSV data to a particular class such as Benign, Mirai or BASHLITE. While labelling each data we merged each CSV file contents so that the final data file contains the whole data. After that, we removed the class imbalance in the dataset with major class undersampling. We also tried minor class oversampling. However, while doing the minor class oversampling the training time was too long and could not finish the training in a single day. Therefore, we focused on the major class undersampling technique to remove class imbalance. After doing the major class undersampling as shown in the following figure, we have got nearly 32% Benign and BASHLITE and 35% of Mirai data samples.

![picture alt](https://github.com/moyeen52/iotbotnet/blob/main/figures/removed_imbalance.png "Dataset imbalance remvoed")

Which is balanced and can enable accurate data classification. 

After the removal of class imbalance, we did a random shuffle on our data so that the dataset samples became representative of all three classes. At this stage, our column name still contains unsupported characters, which we need to remove. Therefore, we removed those unsupported characters such as dot (.) from the column names. Now, to train and test our model with a new set of data we did 70% and 30% training and test split randomly. The 70% train data is used to fit the machine learning model and allows the model to learn about the data. The other 30% data evaluates the performance of the model because this 30% of data was unseen to the model at the training stage.  To run the classifiers, we need to merge the features into a single feature vector. Therefore, we used `Pyspark VectorAssembler` to merge the dataset features into a single column. 

#### 2.3 Machine Learning Algorithms
To classify the data, we used three popular supervised learning algorithms called Random Forest (RF), Decision Tree (DT), and Factorization machines (FM) classifiers. DT's are non-parametric techniques that learn decision rules from a set of features and infer outcomes based on such rules. FM classifiers can estimate interactions between features even if data has tremendous sparsity. Whereas, RF is a popular ensemble technique and allows multi-class classification. While training our data with the RF classifier we tried different settings for the number of trees and max depth. However, with 3 trees and has a max depth of 2 it gives better results. Besides setting the parameters, we used 3-fold cross-validation and seems like we solved the overfitting issue in RF. However, our nonparametric DT classifier by its nature seems to be overfitted with a large set of input features. In contrast, FM classifiers are trained with different step sizes starting from 0.1 to 0.8 with a step difference of 0.2. During training the FM classifier, we had to change our data label so that they fall between 0 to 2. However, probably due to the wrong estimation of step size, FM classifier does not provide good results. 

We also wanted to observe how the unlabelled data traffic can be classified as Benign, Mirai, and BASHLITE. Therefore, we used the most popular unsupervised K-means clustering algorithm to classify such unlabelled data. We tried setting different numbers of clusters in K-means parameter settings and found that 3 as a number of clusters gives a good performance. Therefore, it makes it evident that K-means can correctly identify all the attack and non-attack classes well. 

#### 2.4 Experimental Setup:
To evaluate the performance of the machine learning models we used a workstation having 32GB of RAM, Intel Xeon Processor, 512 GB of SSD and 2TB of storage. Also, we used the Apache Spark version of 3.0.1 on python 3.6.9. We set random seed values so that results are reproducible and did not run any other applications during the evaluation to confirm the validity of our results. Also, the algorithms used the same set of data processing steps and get the same data and environment. We ran our experiments multiple times and took average of the results so that results are representative. 

### 3. Results
In our project we evaluated three supervised classification algorithms called Random Forest (RF), Decision Tree (DT), and Factorization machines(FM). Also, we evaluated unsupervised k-means for the observation of how unlabelled data can be recognized as attack and non attack traffic. In the following subsections we are going to present our results. 

#### 3.1 Supervised Classification Results:
For supervised classification, we trained RF, DT, and FM for observation of f1 score, precision, recall, and accuracy. We ran our experiments 5 times and took the average such that f1 score, precision, recall, and accuracy  all are average of 5 results. We assumed precision as the number of true positives divided by the number of true positives and false positives. Whereas, the recall is the  number of true positives divided by the number of false negatives and the number of true positives. The accuracy is the percentage of number of correct predictions made divided by the total number of predictions made. The f1 score is calculated based on the following formula. 

  `f1_score=2*((precision*recall)/(precision+recall))`

We presented the three classifiers results in the following table. 
Classifier|F1 Score|Precision|Recall|Accuracy
----------|--------|---------|------|--------
Random Forest|0.9315379079|0.93792476|0.9316180694|93%
Decision Tree|0.998627362|0.99862785|0.99862727|99%
Factorization Machines|0.605042017|1.0|0.4337349397|44%

From the results, it is evident that the RF has good balance between precision and recall. Therefore, it’s f1 score and accuracy is representative of correct classification and possibly there is no overfitting problem here because, we did 3-fold cross validation and controlled the max depth.
Though Decision Tree results are also balanced in all cases, we cannot be certain that it learns and predicts the features correctly. Because we had 115 features and the Decision Tree always suffers with overfitting problems in this scenario. Moreover, our Decision Tree classifier does not have any depth parameter set. Therefore, there is a huge chance that it overfits. 
For the Factorization Machines, we experienced very high precision and that indicates that we might have low false positives. However, our recall value is very low and therefore the classifier is not complete and probably we have lots of false negatives. The low recall value affects the F1 score and therefore it is just 0.61. Also, the classifier accuracy is very bad and is only 44%. This kind of problem could be due to the wrong estimation of step size parameters. 

The following figure indicates the F1 score comparison between RF, DT, and FM.

![picture alt](https://github.com/moyeen52/iotbotnet/blob/main/figures/comparsion_model.png "Model Comparisons")

From the figure it is noticeable that the DT has nearly f1 score of 1 may be due to the overfitting problem. Whereas, FM has almost half f1 score compared to DT due to wrong estimation of step size. The RF shows more authentic results due to estimation with multiple trees and having 3-fold cross validation with max depth set. 

#### 3.2 K-means Clustering Results: 
While observing labelled classification, we wanted to observe the performance of unsupervised algorithms so that we can be sure that attack traffic are correctly identified even if there is no data label. Therefore, we evaluated the k-means algorithm and observed the silhouette score by varying the number of clusters. Our data mainly had 7 different classes, which we have reduced to 3 classes for our analysis purpose. Because, we wanted to observe whether the attack is Mirai or BASHLITE instead of getting more specific TCP SYN attack of Mirai or BASHLITE. 

![picture alt](https://github.com/moyeen52/iotbotnet/blob/main/figures/sihouette.jpg "sihouette score for different k")

The above plot of silhouette score clearly indicates that the silhouette score is higher from 3 to 7 and then just starts falling down. Which indicates that k-means can clearly separate the attack traffic and non-attack traffic. 

### 4. Discussion
The major objective of this project is to correctly detect Mirai and BASHLITE attacks through the real IoT traffic analysis. Therefore, we evaluated the performance of RF, DT, and FM classifiers on N-BaIoT dataset. N-BaIoT dataset consists of real Mirai and BASHLITE attack and non-attack traffic from 9 commercially available IoT devices. Also, we used a distributed Apache Spark environment to deal with the large dataset and real time prediction requirements. The results from the RF classifier seem more accurate with a good balance between precision and recall. The F1 score for RF classifier is 0.93 and a good representation of its precision and recall value.  As we did cross validation for RF and challenged the RF model with unseen data, the model eventually trained well to perform better. Also, we used max depth to control the depth of the trees, therefore, we probably observed better evaluation results for RF. But in the case of DT, our analysis result shows that feeding the Decision Tree with around 115 features the algorithm most of the time uses just a small subset of them. We are of the opinion that this would justify why the F1-score of the Decision Tree classifier is more than 0.99 and we are therefore suspecting that the DT is basically overfits. For our FM classifier we observed low recall and high precision value. Therefore, we have achieved a low F1 score and accuracy. Our observation about FM classifiers is that the step size parameter estimation could be wrong and our data is not sparse. Based on these results, we can confirm that our objectives about Mirai and BASHLITE attack classification have been fulfilled with labelled data in a distributed environment.  

However, we wanted to observe the attacks if there is no label in the dataset. Therefore, we used the k-means algorithm and found that it can correctly cluster the Mirai, BASHLITE and non attack traffic. The silhouette score we observed is nearly 1 and that confirms the dense clusters and nice separation of the clusters. 

Based on the above analysis we can confirm that, with labelled and unlabelled data RF and k-means can accurately identify attack and non-attack traffic. If we can correctly classify attack traffic, we will be able to save the propagation of such traffic to save the internet community from experiencing billion dollar loss and unexpected downtime. 

##### Limitations:
There are certain limitations of our work. Which we are going to highlight here. 
 * We relied on the analysis of offline dataset and that may not be representative of live traffic. 
 * We did not prune the decision tree and ended up overfitted results.
 * Another limitation of our work could be the ignorance of the feature reduction step with principal component analysis.
 * Moreover, we did not implement a concrete architecture to observe how live attack requests are handled using the classifier's decisions. 
 * For the FM classifier, we did not accurately determine the step size. 
##### Future works:
As a future extension we are planning to propose a concrete architecture to handle live attack traffic and we are interested to observe how the decisions from ML algorithms affect the performance of live network traffic. We are planning to do feature reductions in future and train more classifiers and deep neural networks to compare their performance. Moreover, as a future extension of this work we want to implement a model selection algorithm and integrate transfer learning features.


### References:
[1]. D. J. Cook, J. C. Augusto, and V. R. Jakkula, “Ambient intelligence: Technologies, applications, and opportunities,” Pervasive and Mobile Computing, vol. 5, no. 4, pp. 277–298, 2009.

[2]. Singh, Sushil Kumar, Shailendra Rathore, and Jong Hyuk Park. "Blockiotintelligence: A blockchain-enabled intelligent IoT architecture with artificial intelligence." Future Generation Computer Systems 110 (2020): 721-743.

[3]. Ni, Jianbing, Xiaodong Lin, and Xuemin Sherman Shen. "Efficient and secure service-oriented authentication supporting network slicing for 5G-enabled IoT." IEEE Journal on Selected Areas in Communications 36.3 (2018): 644-657.

[4]. Hammi, Badis, et al. "A lightweight ECC-based authentication scheme for Internet of Things (IoT)." IEEE Systems Journal(2020).

[5]. Kovacs, Eduard (14 November 2014). "BASHLITE Malware Uses ShellShock to Hijack Devices Running BusyBox". SecurityWeek.com. Retrieved 21 October 2016.

[6]. C. Kolias, G. Kambourakis, A. Stavrou, and J. Voas, “Ddos in the iot: Mirai and other botnets,” Computer, vol. 50, no. 7, pp. 80–84, 2017.

[7]. M. Antonakakis, T. April, M. Bailey, M. Bernhard, E. Bursztein, J. Cochran, Z. Durumeric, J. A. Halderman, L. Invernizzi, M. Kallitsis et al., “Understanding the mirai botnet,” in USENIX Security Symposium, 2017, pp. 1092–1110.

[8]. Meidan, Yair, et al. "N-baiot—network-based detection of iot botnet attacks using deep autoencoders." IEEE Pervasive Computing 17.3 (2018): 12-22.

[9]. C. Zhou & R. Paffenroth, “Anomaly Detection with Robust Deep Autoencoders,” KDD’17, Halifax, NS, Canada, pp. 665–674, 2017.

[10]. Y. Mirsky, T. Doitshman, Y. Elovici, and A. Shabtai, “Kitsune : An Ensemble of Autoencoders for Online Network Intrusion Detection,” pp. 18–21, 2018. 

[11]. T. Luo and S. G. Nagarajan, “Distributed Anomaly Detection using Autoencoder Neural Networks in WSN for IoT,” no. May 2018.

[12]. H. Nguyen, Q. Ngo and V. Le, "IoT Botnet Detection Approach Based on PSI graph and DGCNN classifier," 2018 IEEE International Conference on Information Communication and Signal Processing (ICICSP), Singapore, 2018, pp. 118-122, doi: 10.1109/ICICSP.2018.8549713.

[13]. Doshi, Keval, Yasin Yilmaz, and Suleyman Uludag. "Timely detection and mitigation of stealthy DDoS attacks via IoT networks." IEEE Transactions on Dependable and Secure Computing (2021).

[14]. H. H. Pajouh, A. Dehghantanha, R. Khayami and K.-K.-R. Choo, "Intelligent OS X malware threat detection with code inspection", J. Comput. Virol. Hacking Techn., vol. 14, no. 3, pp. 213-223, Aug. 2018.

[15]. M. S. Alam and S. T. Vuong, "Random forest classification for detecting Android malware",Proc. IEEE Int. Conf. Green Comput. Commun. IEEE Internet Things IEEE Cyber Phys. Social Comput., pp. 663-669, Aug. 2013.

[16]. S. Kudugunta and E. Ferrara, "Deep neural networks for bot detection", Inf. Sci., vol. 467, pp. 312-322, Oct. 2018.

[17]. Alrashdi, Ibrahim, et al. "Ad-iot: Anomaly detection of iot cyberattacks in smart city using machine learning." 2019 IEEE 9th Annual Computing and Communication Workshop and Conference (CCWC). IEEE, 2019.

[18]. Sriram, S., et al. "Network Flow based IoT Botnet Attack Detection using Deep Learning." IEEE INFOCOM 2020-IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS). IEEE, 2020.

[19]. Doshi, Rohan, Noah Apthorpe, and Nick Feamster. "Machine learning DDoS detection for consumer internet of things devices." 2018 IEEE Security and Privacy Workshops (SPW). IEEE, 2018.

[20]. Majumdar, Pramathesh, et al. "A Deep Learning Approach Against Botnet Attacks to Reduce the Interference Problem of IoT." Intelligent Computing and Applications. Springer, Singapore, 2021. 645-655.
