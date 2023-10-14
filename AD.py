# Databricks notebook source
pip install eif

# COMMAND ----------

# MAGIC %run /Tools/library/CFM

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/preprocessing

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/AD_functions

# COMMAND ----------

lib_instance = Cfm()

# COMMAND ----------

from sklearn.svm import OneClassSVM
import eif
from sklearn.linear_model import SGDOneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
import warnings
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from itertools import zip_longest
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# COMMAND ----------

groupe = lib_instance.define_widget("groupe")
path_notebook_preproc_preprocessing = lib_instance.define_widget("path_notebook_preproc_preprocessing") # '/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing'

# COMMAND ----------

#dico_hyperparametres = np.load('/dbfs/FileStore/AD_hyperparametres.npy',allow_pickle='TRUE').item()

# COMMAND ----------

dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
df = dataloader.load_train(groupe = groupe)

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/AD_functions

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/preprocessing

# COMMAND ----------

preproc = Preprocessing(groupe = groupe)
df_train = preproc.preproc(df)
labels = preproc.labels
labels_str = preproc.labels_str
ADC = Anomaly_Detection_Dataset_score_cuted2(labels = labels,labels_str = labels_str)
AD = ADC.fit_predict(df_train,plot = True)

# COMMAND ----------

groupes = ["DecPDV","EncPDV",  "DecUP", "EncUP",  "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]

# COMMAND ----------

dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
df_train_set,df_predict_set = dataloader.load_train_predict_set(groupes = groupes)

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/AD_functions

# COMMAND ----------

for groupe,df_train,df_predict in zip_longest(groupes,df_train_set,df_predict_set):
  #dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
  #df_train, df_predict = dataloader.load_train_predict(groupe = groupe)
  preproc = Preprocessing(groupe = groupe)
  df_train,df_predict = preproc.preproc(df_train,df_predict)
  labels = preproc.labels
  labels_str = preproc.labels_str
  ADC = Anomaly_Detection_Dataset_score_cuted2(labels = labels,labels_str = labels_str)
  AD = ADC.fit_predict(df_train,plot = True,seuil=0.9)


# COMMAND ----------

