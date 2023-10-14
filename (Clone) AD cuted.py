# Databricks notebook source
# MAGIC %run /Tools/library/CFM

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/preprocessing

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/AD_functions

# COMMAND ----------

lib_instance = Cfm()

# COMMAND ----------

from sklearn.svm import OneClassSVM
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
warnings.filterwarnings("ignore")

# COMMAND ----------

groupe = lib_instance.define_widget("groupe")
confidence = float(lib_instance.define_widget('confidence'))/100
path_notebook_preproc_preprocessing = lib_instance.define_widget("path_notebook_preproc_preprocessing") # '/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing'

# COMMAND ----------

dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
df_train = dataloader.load_train(groupe = groupe)

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/AD_functions

# COMMAND ----------

preproc = Preprocessing(groupe = groupe)
df_labelised = preproc.preproc(df_train)
ADC = Anomaly_Detection_score_cuted(groupe = groupe,label_str =)
AD = ADC.fit_predict(df_labelised,plot = True,gridsearch =True)

# COMMAND ----------



# COMMAND ----------

groupes = ["DecPDV","EncPDV",  "DecUP", "EncUP",  "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]

# COMMAND ----------

dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
df_train_set,df_predict_set = dataloader.load_train_predict_set(groupes = groupes)

# COMMAND ----------

#gridsearch_parameters for gradient boosting

param_grid_gb = {'n_estimators': [1000,1500,2000,2500,3000],
                                      'learning_rate' : [0.001,0.005,0.01,0.05,0.1,0.5],
                                      'max_depth' : [3,5,10,50,100]
                          }

#gridsearch_parameters for lgbm
param_grid_lgbm = {'n_estimators': [1000,1500,2000,2500,3000],
                                      'learning_rate' : [0.001,0.005,0.01,0.05,0.1,0.5]
                          }

# COMMAND ----------

for groupe,df_train,df_predict in zip_longest(groupes,df_train_set,df_predict_set):
  #dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
  #df_train, df_predict = dataloader.load_train_predict(groupe = groupe)

  preproc = Preprocessing(groupe = groupe)
  df_train,df_predict = preproc.preproc(df_train,df_predict)
  ADC = Anomaly_Detection_score_cuted()
  AD = ADC.fit_predict(df_labelised,plot = True,gridsearch =True,param_grid_gb = param_grid_gb , param_grid_lgbm = param_grid_lgbm)


# COMMAND ----------

ADC.df

# COMMAND ----------

df_train_set = dataloader.load_train_set(groupes = ["EncUP","EncPDV", "DecPDV", "DecUP", "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"])

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/AD_functions

# COMMAND ----------

for groupe,df_train in zip([ "EncUP","EncPDV", "DecPDV", "DecUP", "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"],df_train_set):
  print(groupe)
  #dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
  #df_train = dataloader.load_train(groupe = groupe)
  preproc = Preprocessing(groupe = groupe)
  df_labelised = preproc.preproc(df_train)
  ADC = Anomaly_Detection_cuted()
  AD = ADC.fit_predict(df_labelised,plot = True,gridsearch =False)

# COMMAND ----------

np.sum(ADC.df["AD_aggreg_score"])

# COMMAND ----------

np.sum(d["AD_aggreg_score"])

# COMMAND ----------

d = ADC.df

# COMMAND ----------

d

# COMMAND ----------

d.loc[d["Valeur"]==0,["AD_aggreg_score"]]

# COMMAND ----------

np.sum(ADC.df.loc[d["Valeur"]==0]["AD_aggreg_score"])

# COMMAND ----------

ADC = Anomaly_Detection_cuted()
AD = ADC.fit_predict(df_labelised,plot = True,gridsearch = True)

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/AD_functions

# COMMAND ----------

admodel = distance_model(model = "QuantileRandomForest",confidence = 0.9)

# COMMAND ----------

out = admodel.fit_predict(df_train.drop(["DT_VALR","Valeur"],axis = 1),df_train["Valeur"],gridsearch = True)

# COMMAND ----------

admodel.model

# COMMAND ----------

np.sum(admodel.ad_up)/len(out)

# COMMAND ----------

np.sum(admodel.ad_down)/len(out)

# COMMAND ----------

np.sum(out)/len(out)

# COMMAND ----------

np.sum(admodel.pred_down > df_train["Valeur"]) / len(df_train["Valeur"])

# COMMAND ----------

np.sum(admodel.pred_up < df_train["Valeur"]) / len(df_train["Valeur"])

# COMMAND ----------

import matplotlib.pyplot as plt 
plt.plot(admodel.pred_down)

# COMMAND ----------

plt.plot(admodel.pred_up)

# COMMAND ----------

preproc = Preprocessing("/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing")
df_train = preproc.preproc(df_train)

# COMMAND ----------

df_train

# COMMAND ----------

df_train.loc[:,["Valeur","label"]]

# COMMAND ----------

IF = IsolationForest(n_estimators = 1000, contamination= 0.1)
IF.fit(df_train.loc[:,["Valeur","label"]])
ifscore = IF.score_samples(df_train.loc[:,["Valeur","label"]])
print(ifscore)

# COMMAND ----------

ocsvm = OneClassSVM(kernel = "rbf",max_iter = 3000,gamma = 'scale',nu = 0.1)
ocsvm.fit(df_train.drop(["DT_VALR"],axis = 1))
ifscore = ocsvm.score_samples(df_train.drop(["DT_VALR"],axis = 1))
print(ifscore)

# COMMAND ----------

#Faire la moyenne des scores

# COMMAND ----------

for df in df_train_set:
  print(df) #Ajouter jour, jour semaine, mois