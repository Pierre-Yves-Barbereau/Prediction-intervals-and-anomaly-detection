# Databricks notebook source
# MAGIC %run /Tools/library/CFM

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/IC_functions

# COMMAND ----------

lib_instance = Cfm()

# COMMAND ----------

import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go

# COMMAND ----------

class preprocessing():
  def __init__(self,NAME_ADB_BASE_CFM_IC,NAME_ADB_VIEW_FLUX_HISTORIQUE,NAME_ADB_VIEW_FLUX_FIN_TRAN,
               path_notebook_preproc_preprocessing,path_dataframe_train,path_dataframe_predict,groupe):
    self.NAME_ADB_BASE_CFM_IC = NAME_ADB_BASE_CFM_IC
    self.NAME_ADB_VIEW_FLUX_HISTORIQUE = NAME_ADB_VIEW_FLUX_HISTORIQUE
    self.NAME_ADB_VIEW_FLUX_FIN_TRAN = NAME_ADB_VIEW_FLUX_FIN_TRAN
    self.path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing
    self.path_dataframe_train = path_dataframe_train
    self.path_dataframe_predict = path_dataframe_predict
    self.groupe = groupe

  def load_train_predict(self):
    self.df_group = lib_instance.get_only_groupe_from_jointure(self.groupe)
    self.df_group = self.df_group[self.df_group['DT_VALR'] >= "2017-1-1"]
    self.df_group.reset_index(inplace = True)

    self.path_train = dbutils.notebook.run(path_notebook_preproc_preprocessing,
                            600,
                            {"path_dataframe": self.path_dataframe_train,
                             'groupe': self.groupe,
                             'setter_train': 1,
                             'debutDate': "2017-01-01"
                             })
    
    self.pickle_in_train = open(self.path_train, 'rb')
    self.dataframe_dectrain = pickle.load(self.pickle_in_train)
    self.dataframe_dectrain['DT_VALR'] = self.df_group['DT_VALR']

    self.path_predict = dbutils.notebook.run(path_notebook_preproc_preprocessing,
                            600,
                            {"path_dataframe": path_dataframe_predict,
                             'groupe': self.groupe,
                             'setter_train': 1,
                             'debutDate': "2017-01-01"
                             })
    
    pickle_in_predict = open(self.path_predict, 'rb')
    self.dataframe_decpredict = pickle.load(self.pickle_in_predict)
    self.dataframe_decpredict = self.dataframe_decpredict.drop(["Valeur"],axis = 1)

    return self.dataframe_dectrain,self.dataframe_decpredict


# COMMAND ----------




#groupe = "DecPDV
#Undefined_widget_group

#Nom Bases
NAME_ADB_BASE_CFM_IC = "IntervalleConfiance"

#Nom des vues à créer
NAME_ADB_VIEW_FLUX_HISTORIQUE = "cfm_flux_historique"
NAME_ADB_VIEW_FLUX_FIN_TRAN = "cfm_fin_tran"

#PAth notebooks
path_notebook_preproc_preprocessing = '/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing'
path_dataframe_train = '/dbfs/tmp/pre_proc_ml/train/DecPDV.sav'

path_dataframe_predict = '/dbfs/tmp/pre_proc_ml/predict/DecPDV.sav'


#EncPDV DecPDV EncUP DecUP EncPet DecPet DecTaxAlc DecTaxPet EncRist EncRprox


# COMMAND ----------

"""df = spark.sql(f"SELECT * FROM {NAME_ADB_BASE_CFM_IC}.{NAME_ADB_VIEW_FLUX_HISTORIQUE} WHERE descriptionFlux = 'DecPDV' ").toPandas()
lendf = df.shape[0]
df["DT_VALR"].apply(pd.to_datetime)
df.sort_values(by = "DT_VALR")
df"""

# COMMAND ----------

decpdv = lib_instance.get_only_groupe_from_jointure('DecPDV')

decpdv = decpdv[decpdv['DT_VALR'] >= "2017-1-1"]

decpdv.reset_index(inplace = True)

# COMMAND ----------

path = dbutils.notebook.run(path_notebook_preproc_preprocessing,
                            600,
                            {"path_dataframe": path_dataframe_train,
                             'groupe': 'DecPDV',
                             'setter_train': 1,
                             'debutDate': "2017-01-01"
                             })
pickle_in = open(path, 'rb')
dataframe_dectrain = pickle.load(pickle_in)
dataframe_dectrain['DT_VALR'] = decpdv['DT_VALR']

# COMMAND ----------


path = dbutils.notebook.run(path_notebook_preproc_preprocessing,
                            600,
                            {"path_dataframe": path_dataframe_predict,
                             'groupe': 'DecPDV',
                             'setter_train': 1,
                             'debutDate': "2017-01-01"
                             })
pickle_in = open(path, 'rb')
dataframe_decpredict = pickle.load(pickle_in)
dataframe_decpredict = dataframe_decpredict.drop(["Valeur"],axis = 1)

# COMMAND ----------

#dataframe_decpredict = dataframe_decpredict.fillna(method = 'ffill')