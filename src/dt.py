import os,time
import sys
import copy
import time
import random
import pyspark
import pyspark.sql.functions as f
from data_processor import read_csv,init_spark,print_sample_counts,remove_class_imbalance,read_all_and_merge,random_shuffle,remove_decimal_points_colname
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
'''
 In this project we are going to classify different IoT Botnets
'''
class BotNetDTClassification:
  def __init__(self):
    '''
    '''
    spark=init_spark()
    self.RANDOM_SEED=1375
    
    print('Reading and Merging Started.....')
    dataset_df=read_all_and_merge(spark)
    #import pdb;pdb.set_trace()
    print('Removing Class Imbalance with under sampling')
    dataset_df=remove_class_imbalance(dataset_df)
    print('Doing Random Shuffle to Mixup the data')
    dataset_df=random_shuffle(dataset_df)# dataset is sequentially constructed

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
    output = assembler.transform(dataset_df)
    #import pdb;pdb.set_trace()
    self.evaluator = MulticlassClassificationEvaluator(labelCol="label", \
                                                  predictionCol="prediction")
    print('split data for train and test')
    train_data,test_data=dataset_df.randomSplit(splits, self.RANDOM_SEED)
    begin_time=time.time()
    for i in range(5):
        print('Training Started...')
        #import pdb;pdb.set_trace()
        trained_model=self.train_model(train_data,assembler)
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
      f1 = self.evaluator.evaluate(predictions, {self.evaluator.metricName: "f1"})
      print(('F1 Score is: %s') % f1)
      weightedFMeasure = self.evaluator.evaluate(predictions, {self.evaluator.metricName: "weightedFMeasure"})
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
      with open('dt_result.txt', 'a+') as fp:
          fp.write(result)



  def train_model(self,train_df,assembler):
      dt =DecisionTreeClassifier(labelCol="label", featuresCol="features",
                                seed=self.RANDOM_SEED)
      pipeline = Pipeline(stages=[assembler,dt])
      
      model = pipeline.fit(train_df)
      return model

BotNetDTClassification()
