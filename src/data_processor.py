import os
import sys
import copy
import time
import random
import pyspark
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from functools import reduce
from pyspark.sql.functions import lit,rand,col,explode,array
'''
 In this project we are going to classify different IoT Botnets
'''
data_folder_loc='/'.join(os.path.abspath(os.getcwd()).split('/')[:-1])+'/data'

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Pyspark IoT Botnet Detector") \
        .config('spark.executor.cores',4)\
        .getOrCreate()

    
    return spark
def read_csv(spark,filename):
    ''' 
      csv read into data frame
    '''
    
    
    df=spark.read.csv(filename,inferSchema =True,header='true')
    return df

def read_all_and_merge(spark):
    '''
      all data read from data folder
      data format 1.class.x.csv
      This method reads each csv and assigns label column with class
      merges df
    '''
   
    label_map={'benign':1,
               'mirai':2,
               'gafgyt':3}
    whole_df=''
    first=True
    for files in os.listdir(data_folder_loc):
        label=label_map[files.split('.')[1]] # assign numeric label
        df=read_csv(spark,data_folder_loc+'/'+files)
        labeled_df=df.withColumn('label',lit(label))
        if first:
           whole_df=labeled_df
           first=False
        else:
           whole_df=whole_df.unionByName(labeled_df)

    return whole_df    
def random_shuffle(df):
    return df.orderBy(rand(seed=1375))
def remove_decimal_points_colname(dataset_df):
    input_feature_list = []
    columns=dataset_df.columns
    for col in columns: # part of data preprocessing
        new_name = col.strip()
        new_name = "".join(new_name.split())
        new_name = new_name.replace('.','')
        input_feature_list.append(new_name)
    dataset_df = dataset_df.toDF(*input_feature_list)
    return dataset_df
def oversample_df(major_df_count,minor_df_count,major_df,minor_df):
    ratio=range(round(major_df_count/minor_df_count))
    oversampled_df = minor_df.withColumn("nv",\
                     explode(array([lit(x) for x in ratio]))).drop('nv') 
    combined_df = major_df.unionAll(oversampled_df)
    return combined_df
def print_sample_counts(df):
    benign_df = df.filter(col("label") == 1)
    mirai_df = df.filter(col("label") == 2)
    gafgyt_df = df.filter(col("label") == 3)
    benign_count=benign_df.count()
    mirai_count=mirai_df.count()
    gafgyt_count=gafgyt_df.count()
    print(('Benign Sample Count:%s')%str(benign_count))
    print(('\nMirai Sample Count:%s')%str(mirai_count))
    print(('\nBashlite Sample Count:%s')%str(gafgyt_count))
def remove_class_imbalance(df,stype='undersample'):
    benign_df = df.filter(col("label") == 1)
    mirai_df = df.filter(col("label") == 2)
    gafgyt_df = df.filter(col("label") == 3)
    benign_count=benign_df.count()
    mirai_count=mirai_df.count()
    gafgyt_count=gafgyt_df.count()
    print(('Benign Sample Count:%s')%str(benign_count))
    print(('\nMirai Sample Count:%s')%str(mirai_count))
    print(('\nBashlite Sample Count:%s')%str(gafgyt_count))
    if stype=='undersample':
       minordf=''
       if gafgyt_count < benign_count and   gafgyt_count < mirai_count:
          minordf=under_sample(benign_df, benign_count, mirai_df, \
            mirai_count, gafgyt_df, gafgyt_count)
          return minordf
       elif benign_count< gafgyt_count and  benign_count < mirai_count:
            minordf=under_sample(gafgyt_df, gafgyt_count, mirai_df, \
            mirai_count,benign_df, benign_count)
            return minordf
       elif mirai_count < benign_count and mirai_count < gafgyt_count:
            minordf=under_sample(gafgyt_df, gafgyt_count,  \
            benign_df, benign_count, mirai_df,mirai_count)
            return minordf
    else:
        major_df = ''
        if benign_count > mirai_count and benign_count > gafgyt_count:
            major_df = benign_df
            major_df = oversample_df(benign_count, mirai_count, major_df, mirai_df)
            major_df = oversample_df(benign_count, gafgyt_count, major_df, gafgyt_df)
        elif mirai_count > benign_count and mirai_count > gafgyt_count:
            major_df = mirai_df
            major_df = oversample_df(mirai_count, benign_count, major_df, benign_df)
            major_df = oversample_df(mirai_count, gafgyt_count, major_df, gafgyt_df)
        elif gafgyt_count > benign_count and mirai_count < gafgyt_count:
            major_df = gafgyt_df
            major_df = oversample_df(gafgyt_count, benign_count, major_df, benign_df)
            major_df = oversample_df(gafgyt_count, mirai_count, major_df, mirai_df)
        return major_df

def under_sample(major_df,major_df_count,major_df2,major_df_count2,minor_df,minor_df_count):
    ratio=int(major_df_count/minor_df_count)
    sampled_maj_df = major_df.sample(False, 1/ratio)
    #import pdb;pdb.set_trace()
    ratio=int(major_df_count2/minor_df_count)
    sampled_maj_df2 = major_df2.sample(False, 1/ratio)
    #import pdb;pdb.set_trace()
    combined_df=unionAll(minor_df,  sampled_maj_df,sampled_maj_df2)
    return combined_df
def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)
              