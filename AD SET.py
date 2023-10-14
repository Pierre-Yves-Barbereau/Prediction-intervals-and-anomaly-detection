# Databricks notebook source
# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/preprocessing

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/AD_functions

# COMMAND ----------

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# COMMAND ----------

fluxs = ["EncPDV", "DecPDV", "EncUP", "DecUP", "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]
models_names = ["IsolationForest","OneClassSVM","LocalOutlierFactor"]
NAME_ADB_BASE_CFM_IC = "IntervalleConfiance"
#Nom des vues à créer
NAME_ADB_VIEW_FLUX_HISTORIQUE = "cfm_flux_historique"
NAME_ADB_VIEW_FLUX_FIN_TRAN = "cfm_fin_tran"

#PAth notebooks
path_notebook_preproc_preprocessing = '/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing'

confidence = 0.9



# COMMAND ----------

import numpy as np
from plotly.subplots import make_subplots

# COMMAND ----------

preproc = preprocessing(NAME_ADB_BASE_CFM_IC = NAME_ADB_BASE_CFM_IC,
                        NAME_ADB_VIEW_FLUX_HISTORIQUE = NAME_ADB_VIEW_FLUX_HISTORIQUE,
                        NAME_ADB_VIEW_FLUX_FIN_TRAN = NAME_ADB_VIEW_FLUX_FIN_TRAN,
                        path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing
                        )

# COMMAND ----------

dataframe_dectrain_set = preproc.load_train_set(fluxs)

# COMMAND ----------

if 0:  #generation des titres
  subtitles = []
  model_names = AD.models_names
  model_names.append("Aggregation Score")
  for flux in fluxs:
    for model in model_names:
      subtitles.append(flux + "  " + model)
  subtitles

# COMMAND ----------



# COMMAND ----------

fluxs = fluxs[:2]
fig = make_subplots(rows=2*len(fluxs),
                    cols=3,
                    specs=[ [{}, {},{}],
                            [{"colspan": 3}, None,None],
                            [{}, {},{}],
                            [{"colspan": 3}, None,None],
                          ],
                    subplot_titles=('EncPDV  IsolationForest',
                                      'EncPDV  OneClassSVM',
                                      'EncPDV  LocalOutlierFactor',
                                      'EncPDV  Aggregation Score',
                                      'DecPDV  IsolationForest',
                                      'DecPDV  OneClassSVM',
                                      'DecPDV  LocalOutlierFactor',
                                      'DecPDV  Aggregation Score'))
                    #subplot_titles=("EncPDV", "DecPDV", "EncUP", "DecUP", "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"))

for i,flux in enumerate(fluxs):
  AD = anomaly_detection(models_names = models_names,confidence = confidence)
  AD.detect_anomaly(dataframe_dectrain_set[i])
  
  fig.append_trace(go.Scatter(x = AD.df["DT_VALR"], y=AD.df["Valeur"],
                          mode='lines',
                          name="y_true",
                          line=dict(
                          color='rgb(0, 0, 256)',
                          width=1)),
                      row=(2*i)+1, col=1)
  
  fig.append_trace(go.Scatter(x = AD.df["DT_VALR"], y=AD.df["Valeur"],
                          mode='lines',
                          name="y_true",
                          line=dict(
                          color='rgb(0, 0, 256)',
                          width=1)),
                      row=(2*i)+1, col=2)
  
  fig.append_trace(go.Scatter(x = AD.df["DT_VALR"], y=AD.df["Valeur"],
                          mode='lines',
                          name="y_true",
                          line=dict(
                          color='rgb(0, 0, 256)',
                          width=1)),
                      row=(2*i)+1, col=3)
  
  fig.append_trace(go.Scatter(x = AD.df["DT_VALR"], y=AD.df["Valeur"],
                          mode='lines',
                          name="y_true",
                          line=dict(
                          color='rgb(0, 0, 256)',
                          width=1)),
                      row=(2*i)+2, col=1)
  
  fig.append_trace(go.Scatter(x=AD.df["DT_VALR"][[i for i in range(len(AD.df["DT_VALR"])) if AD.df["LocalOutlierFactor"][i]==1 ]], y=AD.df["Valeur"],
                              mode='markers',
                              marker=dict(color=AD.df["LocalOutlierFactor"])),
                        row=(2*i)+1, col=1)

#dict_error = { k: v for k, v in zip(fluxs,error_list) }
#fig.for_each_annotation(lambda a: a.update(text = a.text + ': ' +  f"Confidence = {1 - dict_error[a.text]}, Expected = {alphas[1]-alphas[0]}"))
fig.update_layout(height=500*len(fluxs), title_text="Subplots by flux")     
fig.show()