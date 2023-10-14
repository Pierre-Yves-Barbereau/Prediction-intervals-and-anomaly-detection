# Databricks notebook source
# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/preprocessing

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/AD_functions

# COMMAND ----------

groupe = "DecPDV"

NAME_ADB_BASE_CFM_IC = "IntervalleConfiance"
#Nom des vues à créer
NAME_ADB_VIEW_FLUX_HISTORIQUE = "cfm_flux_historique"
NAME_ADB_VIEW_FLUX_FIN_TRAN = "cfm_fin_tran"

#PAth notebooks
path_notebook_preproc_preprocessing = '/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing'
path_dataframe_train = '/dbfs/tmp/pre_proc_ml/train/DecPDV.sav'

path_dataframe_predict = '/dbfs/tmp/pre_proc_ml/predict/DecPDV.sav'

confidence = 0.9

# COMMAND ----------

cd/dbfs

# COMMAND ----------

ls

# COMMAND ----------

cd ..

# COMMAND ----------

preproc = preprocessing(NAME_ADB_BASE_CFM_IC = NAME_ADB_BASE_CFM_IC,
                        NAME_ADB_VIEW_FLUX_HISTORIQUE = NAME_ADB_VIEW_FLUX_HISTORIQUE,
                        NAME_ADB_VIEW_FLUX_FIN_TRAN = NAME_ADB_VIEW_FLUX_FIN_TRAN,
                        path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing,
                        path_dataframe_train = path_dataframe_train,
                        path_dataframe_predict = path_dataframe_predict,
                        groupe = groupe
                        )

dataframe_dectrain = preproc.load_train()

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
import os

#d91381fs01/brutes/exploratoire/Stage_Intervalle_Confiance/prediction/Prod/

# COMMAND ----------

datalake_container = f"/mnt/datalake_cfm_{os.getenv('ENVIRONMENT_PREFIXE')}d91381fs01/brutes/"
folder_init = "exploratoire/Stage_Intervalle_Confiance/prediction/Prod/"

# COMMAND ----------

# Table jours_target_csv 
file_name = "16_Jan-22Jan2023_seb_Phase_22.csv"
file_type = "csv"

infer_schema = "false"
first_row_is_header = "true"
delimiter = ","
encoding="ISO-8859-1"

path = f"{datalake_container}{folder_init}{file_name}"
df = spark.read \
              .format(file_type) \
              .option("header", first_row_is_header) \
              .option("inferSchema", infer_schema) \
              .option("delimiter", delimiter) \
              .option("encoding", encoding) \
              .load(path)






# COMMAND ----------

display(df)

# COMMAND ----------

#Nom Bases
NAME_ADB_BASE_CFM_IC = "IntervalleConfiance"

#Nom table

#Nom des vues à créer
NAME_ADB_VIEW_FLUX_HISTORIQUE = "cfm_flux_historique"
NAME_ADB_VIEW_FLUX_FIN_TRAN = "cfm_fin_tran"

display(spark.sql(f"SELECT * FROM {NAME_ADB_BASE_CFM_IC}.{NAME_ADB_VIEW_FLUX_HISTORIQUE}"))

display(spark.sql(f"SELECT * FROM {NAME_ADB_BASE_CFM_IC}.{NAME_ADB_VIEW_FLUX_FIN_TRAN}"))

# COMMAND ----------

import pandas as pd
df = spark.sql(f"SELECT * FROM {NAME_ADB_BASE_CFM_IC}.{NAME_ADB_VIEW_FLUX_HISTORIQUE} WHERE descriptionFlux = 'DecPDV' ").toPandas()
lendf = df.shape[0]
pd.to_datetime(df["DT_VALR"])
df.sort_values(by = "DT_VALR")
df.head()

# COMMAND ----------

import plotly.express as px
fig = px.histogram(df, x="Valeur")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Quantiles Mobiles

# COMMAND ----------

import plotly.graph_objects as go
import numpy as np

x = df["DT_VALR"][1:100]
y0 = df["Valeur"][1:100]
window_size = 7
qlow = 0.05
qhigh = 0.95
ewm_param = 0.1

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y0,
                    mode='lines',
                    name='Serie'))

fig.add_trace(go.Scatter(x=x, y=np.repeat(a = np.quantile(y0,qhigh), repeats = len(df["Valeur"])),name = f"{qhigh}% Quantile all history"))
fig.add_trace(go.Scatter(x=x, y=y0.rolling(window=window_size,center=True).quantile(qhigh),name = f"{qhigh}% Quantile mobile {window_size}"))
fig.add_trace(go.Scatter(x=x, y=y0.rolling(window=window_size,center=True).quantile(qhigh).ewm(alpha = ewm_param).mean(),name = f"{qhigh}% Quantile mobile ewm {window_size}"))

fig.add_trace(go.Scatter(x=x, y=y0.rolling(window=window_size).mean(),name = f"Mooving average {window_size}"))

fig.add_trace(go.Scatter(x=x, y=np.repeat(a = np.quantile(y0,qlow), repeats = len(df["Valeur"])),name =f"{qlow}% Quantile All history"))
fig.add_trace(go.Scatter(x=x, y=y0.rolling(window=window_size).quantile(qlow),name = f"{qlow}% Quantile mobile {window_size}"))



fig.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ##Anomaly detection

# COMMAND ----------

import matplotlib.pyplot as plt
alphas = [0.05,0.95]
colors = ["green","red"]
fig,ax = plt.subplots()
x_index = df["DT_VALR"]
ax.plot(x_index,df["Valeur"],color="blue",label = f"Valeur")

for alpha,color in zip(list(reversed(alphas)),list(reversed(colors))):
  ax.axhline(y=df["Valeur"].quantile(alpha),color=color,label = f"{alpha}-Quantile")

ax.legend()

x_anomaly_upper = [df["DT_VALR"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]>df["Valeur"].quantile(alphas[1])]
y_anomaly_upper = [df["Valeur"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]>df["Valeur"].quantile(alphas[1])]

x_anomaly_lower = [df["DT_VALR"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]<df["Valeur"].quantile(alphas[0])]
y_anomaly_lower = [df["Valeur"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]<df["Valeur"].quantile(alphas[0])]

ax.scatter(x_anomaly_upper,y_anomaly_upper,color = "red")
ax.scatter(x_anomaly_lower,y_anomaly_lower,color = "green")


# COMMAND ----------

# Enregistrement du DataFrame dans un fichier CSV avec point-virgule comme délimiteur et un en-tête
#path target
#df_clean.write.format("csv")\     #Pour stocker en zone brute

#    .option("sep", ";")\

#    .option("header", first_row_is_header = True) \

#    .save(path_target)



#df_clean.write.format("delta")\    #Pour sotcker en zone raffinée

#    .mode("upsert") # 

#    .saveAsTable(f"{base_name}.{table_name}")


# COMMAND ----------


def quantile_by_day(df,alphas):
  q_lundi = []
  q_mardi = []
  q_mercredi = []
  q_jeudi = []
  q_vendredi = []
  q_samedi = []
  for alpha in alphas :
    q_lundi.append(df[df["DT_VALR"].dt.weekday == 0]["Valeur"].quantile(alpha))
    q_mardi.append(df[df["DT_VALR"].dt.weekday == 1]["Valeur"].quantile(alpha))
    q_mercredi.append(df[df["DT_VALR"].dt.weekday == 2]["Valeur"].quantile(alpha))
    q_jeudi.append(df[df["DT_VALR"].dt.weekday == 3]["Valeur"].quantile(alpha))
    q_vendredi.append(df[df["DT_VALR"].dt.weekday == 4]["Valeur"].quantile(alpha))
    q_samedi.append(df[df["DT_VALR"].dt.weekday == 5]["Valeur"].quantile(alpha))
  return pd.DataFrame([q_lundi,q_mardi,q_mercredi,q_jeudi,q_vendredi,q_samedi],columns = alphas)

# COMMAND ----------



# COMMAND ----------

if 0:
  fig,ax = plt.subplots()
  x_index = df["DT_VALR"]
  ax.plot(x_index,df["Valeur"],color="blue",label = f"Valeur")

  for alpha,color in zip(list(reversed(alphas)),list(reversed(colors))):
    ax.axhline(y=df["Valeur"].quantile(alpha),color=color,label = f"{alpha}-Quantile")

  ax.legend()

  x_anomaly_upper = [df["DT_VALR"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]>df["Valeur"].quantile(alphas[1])]
  y_anomaly_upper = [df["Valeur"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]>df["Valeur"].quantile(alphas[1])]

  x_anomaly_lower = [df["DT_VALR"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]<df["Valeur"].quantile(alphas[0])]
  y_anomaly_lower = [df["Valeur"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]<df["Valeur"].quantile(alphas[0])]

  ax.scatter(x_anomaly_upper,y_anomaly_upper,color = "red")
  ax.scatter(x_anomaly_lower,y_anomaly_lower,color = "green")

is_anormal_absolute = [df["Valeur"][i]<df["Valeur"].quantile(alphas[0]) or df["Valeur"][i]>df["Valeur"].quantile(alphas[1]) for i in range(lendf)]
df[f"{int(np.round((1-confidence)*100))}%_AD_absolute"] = is_anormal_absolute
plot_anomaly(df,is_anormal_absolute,title = f"{int(np.round((1-confidence)*100))}%_AD_absolute")

# COMMAND ----------

if 0:
  x_anomaly_upper = [df["DT_VALR"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]>weekday_quantile(df["DT_VALR"][i],df,alphas[1])]
  y_anomaly_upper = [df["Valeur"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]>weekday_quantile(df["DT_VALR"][i],df,alphas[1])]

  x_anomaly_lower = [df["DT_VALR"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]<weekday_quantile(df["DT_VALR"][i],df,alphas[0])]
  y_anomaly_lower = [df["Valeur"][i] for i in range(len(df["Valeur"])) if df["Valeur"][i]<weekday_quantile(df["DT_VALR"][i],df,alphas[0])]

df["DT_VALR"] = pd.to_datetime(df["DT_VALR"])

is_anormal_by_weekday = [(df["Valeur"][i]>weekday_quantile(df["DT_VALR"][i],df,alphas[1]) or (df["Valeur"][i]<weekday_quantile(df["DT_VALR"][i],df,alphas[0])))for i in range(df.shape[0]) ]
df[f"{int(np.round((1-confidence)*100))}%_AD_by_weekday"] = is_anormal_by_weekday
plot_anomaly(df,is_anormal_by_weekday,title = f"{int(np.round((1-confidence)*100))}%_AD_by_weekday")

# COMMAND ----------

pd.to_datetime(df["DT_VALR"]).dt.weekday

# COMMAND ----------

# MAGIC %md
# MAGIC ##Isolation Forest
# MAGIC

# COMMAND ----------

from sklearn.ensemble import IsolationForest
X = [[df["Valeur"][i]] for i in range(df["Valeur"].shape[0])]
i_f = IsolationForest(n_estimators = 1000, contamination=1-confidence).fit(X)
is_anormal_isolation_forest = i_f.predict([[df["Valeur"][i]] for i in range(df["Valeur"].shape[0])]) == -1
plot_anomaly(df,is_anormal_isolation_forest,title = f"{int(np.round((1-confidence)*100))}%_AD_by_isolation_forest")
df[f"{int(np.round((1-confidence)*100))}%_AD_by_isolation_forest"] = is_anormal_isolation_forest

# COMMAND ----------

from sklearn.svm import OneClassSVM
OCSVM = OneClassSVM(gamma = 'scale',nu = 1-confidence).fit(X)
is_anormal_OCSVM = OCSVM.predict(X) == -1
plot_anomaly(df,is_anormal_OCSVM,title = f"{int(np.round((1-confidence)*100))}%_AD_by_OCSVM")
df[f"{int(np.round((1-confidence)*100))}%_AD_by_OCSVM"] = is_anormal_isolation_forest
print("outliers fraction = ",np.sum(is_anormal_isolation_forest)/df.shape[0])

# COMMAND ----------

from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=2,contamination = 1-confidence)
is_anormal_lof = lof.fit_predict(X) == -1
plot_anomaly(df,is_anormal_lof,title = f"{int(np.round((1-confidence)*100))}%_AD_by_OCSVM")
df[f"{int(np.round((1-confidence)*100))}%_AD_by_LOF"] = is_anormal_lof
print("outliers fraction = ",np.sum(is_anormal_lof)/df.shape[0])

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/AD_functions

# COMMAND ----------

AD = anomaly_detection(df)

# COMMAND ----------



# COMMAND ----------

np.max(AD.AD_aggreg_score)

# COMMAND ----------

#AD By models

# COMMAND ----------

AD.df

# COMMAND ----------

import plotly.graph_objects as go
import plotly.express as px
fig1 = px.line(AD.df, x="DT_VALR", y="Valeur")
fig1.update_traces(line=dict(color = 'rgba(50,50,50,0.2)'))
fig2 = px.scatter(AD.df, x="DT_VALR", y="Valeur",size ="AD_aggreg_score",color ="AD_aggreg_score")
fig = go.Figure(data=fig1.data + fig2.data)
fig.update_layout(title = f"AD ")    
#fig.update_traces(mode='lines')                  
fig.show()