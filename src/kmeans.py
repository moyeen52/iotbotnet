import os,time
import sys
import copy
import time
import random
import pyspark
import pyspark.sql.functions as f
from data_processor import read_csv,init_spark,print_sample_counts,remove_class_imbalance,read_all_and_merge,random_shuffle,remove_decimal_points_colname
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import ClusteringEvaluator
from matplotlib import pyplot as plt 
'''
 In this project we are going to classify different IoT Botnets
'''
class BotNetKmeansCluster:
  def __init__(self):
    '''
    '''
    spark=init_spark()
    self.RANDOM_SEED=674543
    
    print('Reading and Merging Started.....')
    dataset_df=read_all_and_merge(spark)
    import pdb;pdb.set_trace()
    print('Removing Class Imbalance with under sampling')
    dataset_df=remove_class_imbalance(dataset_df)
    print('Doing Random Shuffle to Mixup the data')
    dataset_df=random_shuffle(dataset_df)# dataset is sequentially constructed
  
    #dataset_df=dataset_df.limit(1000)
    print('collect these values from console.....')
    print_sample_counts(dataset_df)
    print('column name cleaning out decimal point')
    
    dataset_df=remove_decimal_points_colname(dataset_df)
    dataset_df=dataset_df.dropna()
    input_feature_list=dataset_df.columns[:-1]
    #input_feature_list=[f.col(x) for x in input_feature_list]
    output_col='label'
    splits=[0.7,0.3]
    print('mapping all feature to single list')
    #dataset_df=dataset_df.withColumn("features", f.array(input_feature_list)).select('features','label')
    assembler = VectorAssembler(inputCols=input_feature_list, outputCol="features")
    dataset_df = assembler.transform(dataset_df)
    #import pdb;pdb.set_trace()
    
    print('split data for train and test')
    train_data,test_data=dataset_df.randomSplit(splits, self.RANDOM_SEED)
    begin_time=time.time()
    print('Training Started...')
    #import pdb;pdb.set_trace()
    k_vals=[3,6,10,23,33]
    res={}
    for i in range(5):
        trained_model=self.train_model(train_data,assembler,k_vals[i])
        time_dif=time.time()-begin_time
        print(('Total Training Time Taken:%s'%time_dif))
        begin_time=time.time()
        print('Test Started...')
        predictions=self.test_model(trained_model,test_data)
        time_dif=time.time()-begin_time
        print(('Total Test Time Taken:%s'%time_dif))
        print('Evaluation Phase Started...')
        silhoute=self.get_evaluation_results(predictions)
        res[k_vals[i]]=silhoute
    fig = plt.figure(figsize=(20,10))
    x_labels=list(set(res.keys()))
    #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    plt.plot(res.keys(),res.values(),linewidth=3)
    fig.suptitle('Silhouette Score For Different K Value', fontsize=20)
    plt.xlabel('K',fontsize=18,labelpad=10)
    #ax.set_xticklabels([0,3,6,10,23,33])
    #plt.xticks(res.keys(), res.keys())
    plt.ylabel('Silhouette Score',fontsize=15,labelpad=20)
    #plt.yticks(res.values(),[0.9,0.94,0.97,0.98,1.0])
    fig.savefig('sihouette.jpg')
    #import pdb;pdb.set_trace()

  def test_model(self, model, test_data):
      predictions = model.transform(test_data)
      return predictions
  def get_evaluation_results(self, predictions):
      self.evaluator =ClusteringEvaluator()
      silhoute=self.evaluator.evaluate(predictions)
      result = '\nsilhouette_score:'+str(silhoute)
      with open('kmeansresult.txt', 'a+') as fp:
          fp.write(result)
      return silhoute


  def train_model(self,train_df,assembler,k=3):

      kmeans = KMeans().setK(k)
      model = kmeans.fit(train_df.select('features'))
      return model


BotNetKmeansCluster()
