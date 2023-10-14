# Databricks notebook source
# MAGIC %run /Tools/library/CFM

# COMMAND ----------

lib_instance = Cfm()

# COMMAND ----------

groupe = lib_instance.define_widget("groupe") #DecPDV

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/preprocessing

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/IC_functions

# COMMAND ----------

path_notebook_preproc_preprocessing = "/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing"
groupe = 'EncRprox'

# COMMAND ----------

#import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
import math
import copy
from datetime import timedelta
from jours_feries_france import JoursFeries
import warnings

# COMMAND ----------

warnings.filterwarnings("ignore")

# COMMAND ----------

#Importing data
dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
df, df_predict = dataloader.load_train_predict(groupe = groupe)

# COMMAND ----------

df_train,df_test = train_test_split(df,test_size=0.2,shuffle=False)

# COMMAND ----------

target_df_test = df_test['Valeur']
DT_VALR_DF_test = df_test["DT_VALR"]
df_test = df_test.drop(["Valeur","DT_VALR"],axis = 1)

# COMMAND ----------

preproc = preprocessing()

# COMMAND ----------

preproc.set_df(df_train)

# COMMAND ----------

X_train_index,X_train,y_train = preproc.x_y_split_clean(df_train)


# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/IC_functions

# COMMAND ----------

agaci = AgACI(alphas = [0.05,0.95] ,adaptative_window_length = 100 ,cal_size = 100,val_size = 100,gammas = [0.0001,0.001,0.01,0.1,1])

# COMMAND ----------

agaci.fit(df_train)

# COMMAND ----------

agaci.predict(df_predict)