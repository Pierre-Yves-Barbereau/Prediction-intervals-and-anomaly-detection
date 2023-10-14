# Databricks notebook source
# MAGIC %run /Tools/library/CFM

# COMMAND ----------

lib_instance = Cfm()

# COMMAND ----------

groupe = lib_instance.define_widget("groupe") #DecPDV

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/preprocessing

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/IC_functions

# COMMAND ----------

path_notebook_preproc_preprocessing = "/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing"

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
from sklearn_quantile import RandomForestQuantileRegressor

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

preproc = Preprocessing()

# COMMAND ----------

preproc.set_df(df_train)

# COMMAND ----------

X_train_index,X_train,y_train = preproc.x_y_split_clean(df_train)

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/IC_functions

# COMMAND ----------

alphas = [0.05,0.95]

# COMMAND ----------

enbpi = EnbPi_quantile(alphas,n_bootstrap = 50,batch_size = 1)

# COMMAND ----------

enbpi.fit(X_train,y_train)

# COMMAND ----------

enbpi_down, enbpi_up, enbpi_median = enbpi.predict(df_test)

# COMMAND ----------

fig = go.Figure()
fig.add_trace(go.Scatter(y=enbpi_up,
                mode='lines',
                name=f'q_{alphas[1]}',
                line=dict(
                    color='rgb(0, 256, 0)',
                    width=0),
                showlegend = False))

fig.add_trace(go.Scatter( y=enbpi_down,
                mode='lines',
                name=f'{int(np.round(100*(alphas[1]-alphas[0])))}% Confidence Interval',
                line=dict(
                    color='rgb(0, 256, 0)',
                    width=0),
                fill='tonexty',
                fillcolor='rgba(0,176,246,0.2)',
                line_color='rgba(255,255,255,0)'))

fig.add_trace(go.Scatter(y=enbpi_median,
              mode='lines',
              name=f'y_median',
              line=dict(
                  color='rgb(256,0, 0)',
                  width=1),
              showlegend = True))

fig.add_trace(go.Scatter( y=target_df_test,
              mode='lines',
              name=f'y_true',
              line=dict(
                  color='rgb(0,0, 256)',
                  width=1),
              showlegend = True))
error = (np.sum(target_df_test<enbpi_down) + np.sum(target_df_test>enbpi_up))/len(target_df_test)
fig.update_layout(title = f"Test : {(1-error)*100}% Confidence Interval Prediction")