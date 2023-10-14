# Databricks notebook source
pip install quantile-forest

# COMMAND ----------

# MAGIC %run /Tools/library/CFM

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/IC_functions

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/preprocessing

# COMMAND ----------

lib_instance = Cfm()

# COMMAND ----------

#import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
#from quantile_forest import RandomForestQuantileRegressor
import math
import copy
from datetime import timedelta
from jours_feries_france import JoursFeries
import warnings
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from itertools import zip_longest
warnings.filterwarnings("ignore")

# COMMAND ----------

groupes = ["DecPDV","EncPDV",  "DecUP", "EncUP",  "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]

# COMMAND ----------

path_notebook_preproc_preprocessing = "/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing"

# COMMAND ----------

dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
df_train_set = dataloader.load_train_set(groupes = groupes)

# COMMAND ----------



# COMMAND ----------

for groupe,df_train in zip(groupes,df_train_set):
  print(groupe)
  print(df_train.describe())

# COMMAND ----------

for groupe,df_train in zip(groupes,df_train_set):
  print(groupe)
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df_train["DT_VALR"], y=df_train["Valeur"],
                      mode='lines',
                      line=dict(
                          color='rgb(0, 0, 256)',
                          width=1),
                      showlegend = False))
      
  fig.update_layout(title = f"{groupe}")
  fig.show()

# COMMAND ----------

for groupe,df_train in zip(groupes,df_train_set):
  preproc = Preprocessing(groupe)
  df_train = preproc.preproc(df_train)
  print(groupe)
  print(df_train.columns)

# COMMAND ----------

