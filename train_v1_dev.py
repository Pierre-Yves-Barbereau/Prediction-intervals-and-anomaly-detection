# Databricks notebook source
# if INFO:py4j.java_gateway:Received command c on object id p0

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------


# IMPORT LIBRARY  
import numpy as np
import pandas as pd

import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pyspark.sql.functions import *
from pyspark.sql.types import *

#import time 
from datetime import datetime, timedelta

import json
import pathlib

from matplotlib.pyplot import figure
import seaborn as sns


# COMMAND ----------

# MAGIC %run /Tools/library/CFM
# MAGIC

# COMMAND ----------

lib_instance = Cfm()

# COMMAND ----------

# var global path 
global path_models
#path_models = "/dbfs/tmp/models/" # pickle    
path_models = lib_instance.path_models_ml # pickle    

global path_pre_proc_ml_train
#path_pre_proc_ml_train= lib_instance.path_train_preproc_ml # pickle

# var global path workspace
global path_notebook_prepoc_folder_preprocessing
#path_notebook_prepoc_folder_preprocessing = f"{lib_instance.path_notebook_preproc_preprocessing}"

# COMMAND ----------

path_dataframe = lib_instance.define_widget("path_dataframe")
#groupe = lib_instance.define_widget("groupe")
debutDateHist = lib_instance.define_widget("debutDateHist")
finDateHist = lib_instance.define_widget("finDateHist")
date_run_train = lib_instance.define_widget("date_run_train")
debutDate =lib_instance.define_widget("debutDate")

# COMMAND ----------

# Init Class Parametres
param_predict = Parametres(evenements=dict({i: [{"libelle": "", "taux": ""}] for i in range(0, 84)}), date_prev="2022-06-03")

# Init Class TypeGroup
class_groupe = TypeGroup(lib_instance.define_widget('groupe'), param_predict)
#class_groupe.version_ml = 'version_3'
#class_groupe.version_preproc = 'version_3'

# COMMAND ----------

def create_str_type_model(type_model):
  try:
    if 'prophet' in str(type_model):
      str_type_model = 'ProphetRegressor'
    else:
      str_type_model = str(type_model)[:str(type_model).find('(')]
  except:
    str_type_model = str(type_model)
  return str_type_model

# COMMAND ----------

def save_model(model, type_model, groupe, path_models):
  str_type_model = create_str_type_model(type_model)
  
  # on enregistre le modele
  path = path_models
  filename = str_type_model + str(groupe) + '.sav'
  filename_model_ = path_models + str_type_model + str(groupe) + '.sav'
  lib_instance.write_pickle(path, filename, model)
  return filename_model_

# COMMAND ----------

def save_x_train(x_train, type_model, groupe, path_models):
  str_type_model = create_str_type_model(type_model)
  
  # on enregistre le x_train pour intervalle de confiance
  path = path_models
  filename = str_type_model + str(groupe) + '_x_train.sav'
  filename_x_train = path_models + str_type_model + str(groupe) + '_x_train.sav'
  lib_instance.write_pickle(path, filename, x_train)
  return filename_x_train

# COMMAND ----------

def function_stockage_feature_importance(df_, date_run_train_, groupe_):
  """
  Fonction qui permet de stockage les features importance dans la table ADB
  """
  file_type = "delta"
  mode = "append"
  overwriteSchema = "false"
  table_name = "FEATURE_IMPORTANCES_TRAIN_ML"  

  # Pré-Traitement
  #df = df.reset_index(drop=False).rename(columns={"index": "name_feature", 0: "value_feature"})
  df_["date_run_train"] = date_run_train_
  df_["groupe"] = groupe_
  df_ = df_[['date_run_train' ,'groupe' ,'name_feature', 'value_feature']]
  spardf = spark.createDataFrame(df_)
  spardf = spardf.selectExpr("*","current_timestamp() AS DATE_TIME_ZONE") \
                 .withColumn("date_insert", from_utc_timestamp("DATE_TIME_ZONE", 'Europe/Paris')) \
                 .withColumn("dat_maj", from_utc_timestamp("DATE_TIME_ZONE", 'Europe/Paris')) \
                 .selectExpr("*","1 AS ligne_active" ,"0 AS ligne_suppr") \
                 .drop("DATE_TIME_ZONE")
  
  """spardf = spardf.withColumn("date_insert", from_utc_timestamp("DATE_TIME_ZONE", 'Europe/Paris'))
  spardf = spardf.withColumn("dat_maj", from_utc_timestamp("DATE_TIME_ZONE", 'Europe/Paris'))
  spardf= spardf
  """

  idx_row_max = spark.sql(f"select max(idx_row) as count from {table_name}").toPandas()['count'][0]
  if np.isnan(idx_row_max):
    idx_row_max = 0

  #spardf = spardf.withColumn("idx_row",idx_row_max +row_number().over(Window.orderBy(monotonically_increasing_id()))).withColumn("idx_row",col("idx_row").cast(LongType()))
  spardf = spardf.withColumn("idx_row", idx_row_max + row_number().over(Window.orderBy(monotonically_increasing_id())))

  # write TABLE in Databricks
  PySparkClass.write(df=spardf, format=file_type, mode=mode, overwrite_schema=overwriteSchema, name_table=table_name)
  return

# COMMAND ----------

def main_train_model(path_dataframe, debutDateHist, finDateHist, date_run_train, class_group, path_models):
  apply_filterOnDates = False
  
  if path_dataframe == 'path_dataframeDefault' :
    dataframe = class_group.get_only_groupe_from_jointure(class_groupe.label)
    path_dataframe = f"{class_group.path_train_preproc_ml}{class_groupe.label}.sav"
  else :
    dataframe = class_group.load_pickle(path_dataframe)
    
  dataframe = dataframe.sort_values(["DT_VALR"])
  
  if debutDateHist == 'debutDateHistDefault' :
    debutDatePrep = dataframe["DT_VALR"][0]
  else :
    apply_filterOnDates = True
    debutDatePrep = pd.to_datetime(debutDateHist)
    
  if finDateHist == 'finDateHistDefault' :
    finDatePrep = dataframe["DT_VALR"][len(dataframe) - 1]
  else :
    apply_filterOnDates = True
    finDatePrep = pd.to_datetime(finDateHist)
  
  if apply_filterOnDates:
    dataframe = class_group.filter_on_dates(dataframe, debutDatePrep, finDatePrep)
  
  dict_param = class_group.get_params()
  
  type_model = dict_param['type_model']
  train_model = dict_param['train_model']
  nb_a_enlever_post_prec = dict_param['nb_a_enlever_post_prec']
  debutDatePrep = dict_param['debut_date_prep']
  #finDatePrep = dict_param['finDatePrep']
  
  dataframe = class_group.filter_on_dates(dataframe, debutDatePrep, finDatePrep)

  dataframe_ = class_group.creation_table_processed(dataframe = dataframe, path_dataframe = path_dataframe, setter_train = "1", debutDate = debutDate)
  #dataframe_ = class_group.creation_table_processed(dataframe, path_dataframe)
  
  dataframe_ = dataframe_[nb_a_enlever_post_prec:].reset_index(drop=True)
  
  y = dataframe_['Valeur'].to_numpy()
  x = dataframe_.drop(columns=['Valeur'], axis=1, inplace = False)
  
  if (class_groupe.label == 'DecPDV' and class_groupe.version_ml == 'version_4'):
    del x['DT_VALR']
  
  x = x.to_numpy()
  #x = dataframe_.drop(columns=['Valeur'], axis=1, inplace = False).to_numpy()
  df_prophet = dataframe_.rename(columns={"DT_VALR": "ds", "Valeur": "y"})
  
  model = train_model(type_model=type_model, x=x, y=y, df=df_prophet) #type_model.fit(x, y)
  
  if "RandomForestRegressor" in str(type_model): #Importance feature DecTaxAlc
    importances = model.feature_importances_
    importances = pd.Series(importances, index=dataframe_.drop(columns=['Valeur'], axis=1, inplace=False).columns)
    importances = importances.sort_values(ascending = False)
    figure(figsize=(10, 6), dpi=80)
    data = pd.DataFrame(importances).transpose()
    ax = sns.barplot(data=data, orient = 'h')
    ax.set(xlabel='Importance feature', ylabel='Features')
    
    importances = importances.reset_index(drop=False).rename(columns={"index": "name_feature", 0: "value_feature"})
    function_stockage_feature_importance(importances ,date_run_train ,class_groupe.label)
    
  # si on veut retourner un pickle
  filename_model = save_model(model, type_model, class_groupe.label, path_models)
  filename_x_train = save_x_train(x, type_model, class_groupe.label, path_models)
  return filename_model 

# COMMAND ----------

# debutDateHist
if lib_instance.str_format_datetime(debutDateHist):
  pass
else : 
  print("finDateHist is not AAAA-MM-DD")

# finDateHist 
if lib_instance.str_format_datetime(finDateHist):
  pass
else : 
  print("finDateHist is not AAAA-MM-DD")

# date_run_train
date_run_train = lib_instance.str_to_pd_datetime(date_run_train)

print(f"\ndebutDateHist is {debutDateHist}")
print(f"finDateHist is {finDateHist}")
print(f"date_run_train is {date_run_train}")

# COMMAND ----------

path_filename_model = main_train_model(path_dataframe, debutDateHist, finDateHist, date_run_train, class_groupe, path_models)

# COMMAND ----------

#.notebook.exit(path_filename_model)

# COMMAND ----------


#Boucler sur les dates

path_filename_model = main_train_model(path_dataframe, debutDateHist, finDateHist, date_run_train, class_groupe, path_models)

# COMMAND ----------

class_groupe = TypeGroup(lib_instance.define_widget('groupe'), param_predict)
dataframe = class_groupe.get_only_groupe_from_jointure(class_groupe.label)
dataframe["DT_VALR"][0]