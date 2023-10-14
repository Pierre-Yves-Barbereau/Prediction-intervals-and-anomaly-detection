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



# COMMAND ----------

names = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
data_loaded = pd.read_csv("/dbfs/FileStore/creditcard.csv",sep = ",",names = names) 
data = data_loaded.iloc[1:,:]
data["Class"][data["Class"] == "1"] = 1
data["Class"][data["Class"] == "0"] = 0
data

# COMMAND ----------

le = preprocessing.LabelEncoder()
data["Class"] = le.fit_transform(data["Class"])
data["Class"][data["Class"] == 1] = -1
data["Class"][data["Class"] == 0] = 1
data

# COMMAND ----------

len(data)

# COMMAND ----------



# COMMAND ----------

data_target = data["Class"]
data_train = data.drop(["Class"],axis = 1)


# COMMAND ----------

data_train

# COMMAND ----------

data_train = preprocessing.normalize(data_train)

# COMMAND ----------

data_train

# COMMAND ----------

len(data[data['Class']==1])

# COMMAND ----------

len(data[data['Class']==-1])

# COMMAND ----------

len(data[data['Class']==-1])/len(data[data['Class']==1])

# COMMAND ----------

contamination = 0.00172

# COMMAND ----------

dic = {}
for i in [100,150,200,250,300,5000,1000]:
  print(i)
  IF = IsolationForest(n_estimators = int(len(data)/i), contamination = contamination)
  IF.fit(data)
  IF_pred = IF.score_samples(data)
  dic[i] = roc_auc_score(data_target, IF_pred)
dic

# COMMAND ----------

0.68

# COMMAND ----------

dic = {}
data_train_np = np.array(data_train,dtype = float)
for i in [400,500,1000,1500]:
  IF = eif.iForest(data_train_np,sample_size=(284000/i), ntrees=i, ExtensionLevel=1)
  pred = -IF.compute_paths(X_in=data_train_np)
  dic[i] = roc_auc_score(data_target, pred)
dic

# COMMAND ----------

dic = {}
for i in [100,200,300]:
  IF = OneClassSVM(kernel = "rbf", max_iter = i, gamma = 'scale', nu = 0.00172)
  IF.fit(data)
  IF_pred = -IF.score_samples(data)
  dic[i] = roc_auc_score(data_target, IF_pred)
dic

# COMMAND ----------

dic = {}
IF = SGDOneClassSVM(nu=0.00172, shuffle=True, fit_intercept=True, random_state=42)
IF.fit(data)
IF_pred = IF.score_samples(data)
roc_auc_score(data_target, IF_pred)

# COMMAND ----------

dic = {}
for i in [5,10,300]:
  IF = LocalOutlierFactor(n_neighbors=i,contamination = 0.00172)
  IF_pred = IF.fit_predict(data)
  dic[i] = roc_auc_score(data_target, IF_pred)
dic

# COMMAND ----------

df = pd.DataFrame(index = ["a","b"],columns = ["c","d"])
df

# COMMAND ----------

