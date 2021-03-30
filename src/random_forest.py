import os,time
import sys
import copy
import time
import random
import pyspark
import pyspark.sql.functions as f
from data_processor import init_spark,print_sample_counts,remove_class_imbalance,read_all_and_merge,random_shuffle,remove_decimal_points_colname
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
'''
 In this project we are going to classify different IoT Botnets
'''
class BotNetRFClassification:
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
  
    dataset_df=dataset_df.limit(1000)
    print_sample_counts(dataset_df)
    print('column name cleaning out decimal point')
    dataset_df=remove_decimal_points_colname(dataset_df)
    input_feature_list=dataset_df.columns[:-1]
    #input_feature_list=[f.col(x) for x in input_feature_list]
    output_col='label'
    splits=[0.7,0.3]
    print('mapping all feature to single list')
    #dataset_df=dataset_df.withColumn("features", f.array(input_feature_list)).select('features','label')
    assembler = VectorAssembler(inputCols=input_feature_list, outputCol="features")
    #output = assembler.transform(dataset_df)
    #import pdb;pdb.set_trace()
    self.evaluator = MulticlassClassificationEvaluator(labelCol="label", \
                                                  predictionCol="prediction")
    print('split data for train and test')
    train_data,test_data=dataset_df.randomSplit(splits, self.RANDOM_SEED)
    begin_time=time.time()
    print('Training Started...')
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
    




  def train_model(self,train_df,assembler):
      rf =RandomForestClassifier(labelCol="label", featuresCol="features",
                                seed=self.RANDOM_SEED)
      pipeline = Pipeline(stages=[assembler,rf])
      paramGrid = ParamGridBuilder()\
                   .addGrid(rf.numTrees, [3])\
                   .addGrid(rf.maxDepth, [2])\
                   .build()
      
      crossval = CrossValidator(
                 estimator=pipeline,
                 estimatorParamMaps=paramGrid,
                 evaluator=self.evaluator,
                 numFolds=3)
      model = crossval.fit(train_df)
      return model

BotNetRFClassification()
