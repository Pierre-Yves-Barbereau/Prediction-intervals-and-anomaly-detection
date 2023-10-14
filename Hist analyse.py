# Databricks notebook source
# MAGIC %run /Tools/library/CFM

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

from pyspark.sql.functions import *
from pyspark.sql.types import *
import os
import pandas as pd
import datetime
import numpy as np

#Nom Bases
NAME_ADB_BASE_CFM_IC = "IntervalleConfiance"

#Nom table

#Nom des vues à créer
NAME_ADB_VIEW_FLUX_HISTORIQUE = "cfm_flux_historique"
NAME_ADB_VIEW_FLUX_FIN_TRAN = "cfm_fin_tran"

df = spark.sql(f"SELECT * FROM {NAME_ADB_BASE_CFM_IC}.{NAME_ADB_VIEW_FLUX_HISTORIQUE} WHERE descriptionFlux = 'DecPDV' ").toPandas()
df.set_index(pd.DatetimeIndex(df['DT_VALR']), inplace=True)
df["DT_VALR"] = pd.to_datetime(df["DT_VALR"])
lendf = df.shape[0]
df.head()

# COMMAND ----------




# COMMAND ----------



# COMMAND ----------

import plotly.figure_factory as ff

# Add histogram data
lundis = df[df["DT_VALR"].dt.weekday == 0]["Valeur"]
mardis = df[df["DT_VALR"].dt.weekday == 1]["Valeur"]
mercredis = df[df["DT_VALR"].dt.weekday == 2]["Valeur"]
jeudis = df[df["DT_VALR"].dt.weekday == 3]["Valeur"]
vendredis = df[df["DT_VALR"].dt.weekday == 4]["Valeur"]
samedis = df[df["DT_VALR"].dt.weekday == 5]["Valeur"]


# Group data together
hist_data = [lundis, mardis, mercredis, jeudis, vendredis, samedis]

group_labels = ['Lundis', 'Mardis', 'Mercredis', 'Jeudis','Vendredis','Samedis']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels)
fig.show()

# COMMAND ----------



# COMMAND ----------

bins = 'auto'
fig, ax = plt.subplots(2, 3, sharex='row', sharey=False, figsize=(20, 10)) 
ax[0,0].hist(lundis,bins = bins)
ax[0,1].hist(mardis,bins = bins)
ax[0,2].hist(mercredis,bins = bins)
ax[1,0].hist(jeudis,bins = bins)
ax[1,1].hist(vendredis,bins = bins)
ax[1,2].hist(samedis,bins = bins)
ax[0,0].set_title("lundis")
ax[0,1].set_title("mardis")
ax[0,2].set_title("mercredis")
ax[1,0].set_title("jeudis")
ax[1,1].set_title("vendredis")
ax[1,2].set_title("samedis")

# COMMAND ----------

import numpy as np
from sklearn.mixture import GaussianMixture
gm_lundis = GaussianMixture(n_components=3, random_state=0,means_init = [[0.6*1e8],[1.9*1e8],[3*1e8]]).fit(np.array(lundis).reshape(-1, 1))
lundis_means = gm_lundis.means_
lundis_covariances = gm_lundis.covariances_
lundis_weights = gm_lundis.weights_



# COMMAND ----------

from scipy.stats import norm 

#densities of mixture models
x_axis = np.arange(np.min(lundis),np.max(lundis),1000)
y_axis0 = norm.pdf(x_axis, float(lundis_means[0][0]), np.sqrt(float(lundis_covariances[0][0][0])))*lundis_weights[0] # 1st gaussian
y_axis1 = norm.pdf(x_axis, float(lundis_means[1][0]), np.sqrt(float(lundis_covariances[1][0][0])))*lundis_weights[1] # 2nd gaussian
y_axis2 = norm.pdf(x_axis, float(lundis_means[2][0]), np.sqrt(float(lundis_covariances[2][0][0])))*lundis_weights[2] # 3nd gaussian

 #plotting
fig, ax = plt.subplots()
ax.hist(lundis, density=True, color='black', bins=bins)
ax.plot(x_axis,y_axis0)
ax.plot(x_axis,y_axis1)
ax.plot(x_axis,y_axis2)
ax.set_title("Mixture model Bof Bof")

# COMMAND ----------

