# Databricks notebook source
pip install quantile-forest

# COMMAND ----------

# MAGIC %run /Tools/library/CFM

# COMMAND ----------

#%run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/IC_functions

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/IC_functions

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/preprocessing

# COMMAND ----------

lib_instance = Cfm()

# COMMAND ----------

#dbutils.widgets.removeAll()

# COMMAND ----------

# ["EncPDV", "DecPDV", "EncUP", "DecUP", "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]

# COMMAND ----------

#Variables widget
groupe = lib_instance.define_widget("groupe") 
quantile_top = float(lib_instance.define_widget('quantile_top'))
quantile_bottom = float(lib_instance.define_widget('quantile_bottom'))

#val_size = float(lib_instance.define_widget("val_size")) #365
cal_size = float(lib_instance.define_widget("cal_size")) # 365
gridsearch = int(lib_instance.define_widget("grid search"))

#PAth notebooks
path_notebook_preproc_preprocessing = lib_instance.define_widget("path_notebook_preproc_preprocessing") #'/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing'

# COMMAND ----------

#import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import MinMaxScaler
import math
import copy
from datetime import timedelta
from jours_feries_france import JoursFeries
import warnings
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from itertools import zip_longest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

# COMMAND ----------

dico_hyperparametres = np.load(f'/dbfs/FileStore/IC_hyperparametres.npy',allow_pickle = True).item()

# COMMAND ----------

alphas = [quantile_bottom,quantile_top] #quantiles

# COMMAND ----------

#/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ###Loading Data and preprocessing

# COMMAND ----------

dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing) #load dataloader
df_train_loaded, df_predict_loaded = dataloader.load_train_predict(groupe = groupe) #load data

# COMMAND ----------

df_train, df_predict = df_train_loaded, df_predict_loaded

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict

# COMMAND ----------

preproc2 = Preprocessing2(groupe = groupe) #mode PCA 3 a régler en fonction du nombre de dimensions à garder par groupe
df_train = preproc2.preproc(df_train) 

# COMMAND ----------

#best model
ci = Conformal_Inference(alphas = alphas,groupe = groupe,cal_size = cal_size, mode = "test") # Initialisation
ci.fit(df_train) # Fit and gridsearch
df_output = ci.predict(df_predict,plot = True) # Prediction