# Databricks notebook source
# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/preprocessing

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/AD_functions

# COMMAND ----------

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# COMMAND ----------

groupe = lib_instance.define_widget("groupe")
confidence = float(lib_instance.define_widget('confidence'))/100
path_notebook_preproc_preprocessing = lib_instance.define_widget("path_notebook_preproc_preprocessing") #'/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing'


# COMMAND ----------

fluxs = ["EncPDV", "DecPDV", "EncUP", "DecUP", "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]

# COMMAND ----------

preproc = preprocessing(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
dataframe_dectrain = preproc.load_train(groupe = groupe)

# COMMAND ----------

AD = Anomaly_Detection(confidence =  confidence)
AD.fit_predict(dataframe_dectrain)

# COMMAND ----------

error_list = []
fig = make_subplots(rows=len(AD.models_names),
                    cols=1,
                    #subplot_titles=("EncPDV", "DecPDV"))
                    subplot_titles=("EncPDV", "DecPDV", "EncUP", "DecUP", "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"))

for i,model in enumerate(AD.models_names):

  fig1 = px.line(AD.df, x="DT_VALR", y="Valeur")
  fig1.update_traces(line=dict(color = 'rgba(50,50,50,0.2)'))
  fig2 = px.scatter(AD.df, x="DT_VALR", y="Valeur",size =model,color =model)
  fig = go.Figure(data=fig1.data + fig2.data)
  fig.update_layout(title = f"{model}")    
  #fig.update_traces(mode='lines')                  
  fig.show()

# COMMAND ----------

AD.models_names

# COMMAND ----------

