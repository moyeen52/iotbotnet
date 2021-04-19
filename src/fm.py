import os,time
import sys
import copy
import time
import random
import pyspark
import numpy as np 
import pyspark.sql.functions as f
from data_processor import read_csv,init_spark,print_sample_counts,remove_class_imbalance,read_all_and_merge,random_shuffle,remove_decimal_points_colname
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import FMClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when   
'''
 In this project we are going to classify different IoT Botnets
'''
class BotNetFMClassification:
  def __init__(self):
    '''
    '''
    spark=init_spark()
    self.RANDOM_SEED=43524#55343
    
    print('Reading and Merging Started.....')
    
    dataset_df=read_all_and_merge(spark,label_map)
    #import pdb;pdb.set_trace()
    print('Removing Class Imbalance with under sampling')
    dataset_df=remove_class_imbalance(dataset_df)
    print('Doing Random Shuffle to Mixup the data')
    dataset_df=random_shuffle(dataset_df)# dataset is sequentially constructed
  
    dataset_df=dataset_df.limit(1000)
    print_sample_counts(dataset_df)
    print('column name cleaning out decimal point')
  
    dataset_df=dataset_df.withColumn('labl',when(dataset_df.label==3,0))
    
    dataset_df=remove_decimal_points_colname(dataset_df)
    dataset_df=dataset_df.dropna()
    input_feature_list=dataset_df.columns[:-1]
    #input_feature_list=[f.col(x) for x in input_feature_list]
    output_col='labl'
    splits=[0.7,0.3]
    print('mapping all feature to single list')
    #dataset_df=dataset_df.withColumn("features", f.array(input_feature_list)).select('features','label')
    assembler = VectorAssembler(inputCols=input_feature_list, outputCol="features")
    output = assembler.transform(dataset_df)
    #import pdb;pdb.set_trace()
    self.evaluator = MulticlassClassificationEvaluator(labelCol="labl", \
                                                  predictionCol="prediction")
    print('split data for train and test')
    train_data,test_data=dataset_df.randomSplit(splits, self.RANDOM_SEED)
    for step_size in np.arange(0.1,0.8,0.2):
        begin_time=time.time()
        print('Training Started...')
        
        trained_model=self.train_model(train_data,assembler,step_size)
        time_dif=time.time()-begin_time
        print(('Total Training Time Taken:%s'%time_dif))
        begin_time=time.time()
        print('Test Started...')
        predictions=self.test_model(trained_model,test_data)
        time_dif=time.time()-begin_time
        print(('Total Test Time Taken:%s'%time_dif))
        print('Evaluation Phase Started...')
        self.get_evaluation_results(predictions)

  def test_model(self, model, test_data):
      predictions = model.transform(test_data)
      return predictions

  def get_evaluation_results(self, predictions):
      #import pdb;pdb.set_trace()
      f1 = self.evaluator.evaluate(predictions, {self.evaluator.metricName: "f1"})
      print(('F1 Score is: %s') % f1)
      weightedFMeasure = self.evaluator.evaluate(predictions,  {self.evaluator.metricName: "weightedFMeasure"})
      print(('weighted FMeasure is: %s') % weightedFMeasure)
      precision = self.evaluator.evaluate(predictions, {self.evaluator.metricName: "weightedPrecision"})
      print(('Precision is: %s') % precision)
      recall = self.evaluator.evaluate(predictions, {self.evaluator.metricName: "weightedRecall"})
      print(('Recall is: %s') % recall)
      accuracy = self.evaluator.evaluate(predictions, {self.evaluator.metricName: "accuracy"})
      print(('Accuracy is: %s') % accuracy)
      result = '\nF1 Score is:' + str(f1) + '\n' + 'weighted FMeasure is:' + \
               str(weightedFMeasure) + '\n' + 'Precision is:' + str(precision) + \
               '\nRecall is:' + str(recall) + '\n' + 'Accuracy is:' + str(accuracy)
      with open('fmresult.txt', 'a+') as fp:
          fp.write(result)



  def train_model(self,train_df,assembler,step):

      #lsvc =NaiveBayes(smoothing=1.0, modelType="multinomial")
      fm=FMClassifier(labelCol="labl", featuresCol="features",stepSize=step)
      #paramGrid = ParamGridBuilder()\
      #             .addGrid(fm.stepSize, [0.01,0.0001,0.00001])\
      #             .build()
      pipeline = Pipeline(stages=[assembler,fm])
      #crossval = CrossValidator(
      #           estimator=pipeline,
      #            estimatorParamMaps=paramGrid,
      #           evaluator=self.evaluator,
      #           numFolds=3)
      model = pipeline.fit(train_df)
      return model

BotNetFMClassification()
