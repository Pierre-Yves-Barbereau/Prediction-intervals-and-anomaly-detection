# Databricks notebook source
pip install quantile-forest 

# COMMAND ----------

# MAGIC %run /Tools/library/CFM

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/Prod2/ICF

# COMMAND ----------



# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/preprocessing

# COMMAND ----------

lib_instance = Cfm()

# COMMAND ----------

#dbutils.widgets.removeAll()

# COMMAND ----------

groupes =  ["EncPDV", "DecPDV", "EncUP", "DecUP", "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]

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

dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing) #load dataloader
df_train_loaded_set, df_predict_loaded_set = dataloader.load_train_predict_set(groupes = groupes) #load data

# COMMAND ----------

df_train_set, df_predict_set = df_train_loaded_set, df_predict_loaded_set

# COMMAND ----------

df_train, df_predict = df_train_loaded, df_predict_loaded

# COMMAND ----------

# DBTITLE 1,u
for groupe, df_train in zip(groupes,df_train_set):
  print(groupe)
  display(df_train.describe())
  print("")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict

# COMMAND ----------

preproc = Preprocessing2(groupe = groupe) 
df_train = preproc.preproc(df_train,PCA = False) 
df_train["label"] = preproc.labels


# COMMAND ----------

np.mean(df_train["Valeur"])

# COMMAND ----------

gb = GradientBoostingRegressor(loss = "quantile")
gb.fit(df_train.drop(["DT_VALR","Valeur"], axis = 1),df_train["Valeur"])

# COMMAND ----------

import plotly.express as px
for groupe,df_train in zip(groupes,df_train_set):
  preproc = Preprocessing2(groupe = groupe)
  df_train = preproc.preproc(df_train)
  gb = GradientBoostingRegressor(loss = "quantile")
  gb.fit(df_train.drop(["DT_VALR","Valeur"], axis = 1),df_train["Valeur"])
  data_canada = px.data.gapminder().query("country == 'Canada'")
  fig = px.bar( x=df_train.drop(["DT_VALR","Valeur"], axis = 1).columns, y=gb.feature_importances_)
  fig.update_layout(title = f"{groupe} : Features importance")  
  fig.show()

# COMMAND ----------

gb.feature_importances_

# COMMAND ----------

path_df_train = "/dbfs/tmp/pre_proc_ml/train/" + "DecUP" + ".sav" #set path for train dataset
df_group = lib_instance.get_only_groupe_from_jointure(groupe) 
df_group = df_group[df_group['DT_VALR'] >= "2017-1-1"] 
df_group.reset_index(inplace = True)   #reset index
path_train = dbutils.notebook.run(path_notebook_preproc_preprocessing,
                        600,
                        {"path_dataframe": path_df_train,
                          'groupe': groupe,
                          'setter_train': 1,
                          'debutDate': "2017-01-01"
                          })
pickle_in_train = open(path_train, 'rb')
df_train = pickle.load(pickle_in_train) #download df_trian
df_train['DT_VALR'] = df_group['DT_VALR']

# COMMAND ----------

fig = px.line(df_train, x="DT_VALR", y="Valeur", title=f'{"DecUP"}')
fig.show()

# COMMAND ----------

path_df_train = "/dbfs/tmp/pre_proc_ml/train/" + "DecTaxPet" + ".sav" #set path for train dataset
df_group = lib_instance.get_only_groupe_from_jointure(groupe) 
df_group = df_group[df_group['DT_VALR'] >= "2017-1-1"] 
df_group.reset_index(inplace = True)   #reset index
path_train = dbutils.notebook.run(path_notebook_preproc_preprocessing,
                        600,
                        {"path_dataframe": path_df_train,
                          'groupe': groupe,
                          'setter_train': 1,
                          'debutDate': "2017-01-01"
                          })
pickle_in_train = open(path_train, 'rb')
df_train = pickle.load(pickle_in_train) #download df_trian
df_train['DT_VALR'] = df_group['DT_VALR']

# COMMAND ----------

fig = px.line(df_train, x="DT_VALR", y="Valeur", title=f'{"DecTaxPet"}')
fig.show()

# COMMAND ----------

import plotly.figure_factory as ff
import numpy as np
df_train, df_predict = df_train_loaded, df_predict_loaded
groupe = "DecPDV"

preproc = Preprocessing2(groupe = groupe) 
df_train = preproc.preproc(df_train,PCA = False)
labels_str = list(set(preproc.labels_str))
labels = list(set(preproc.labels))
hist_data = []
for label in labels:
  hist_data.append(df_train["Valeur"][df_train["label"] == label])
fig = ff.create_distplot(hist_data, labels_str, bin_size=10000000)
fig.update_layout(title = f"{groupe}")
fig.show()

# COMMAND ----------

len(hist_data)

# COMMAND ----------

hist_data[0]

# COMMAND ----------

len(labels)

# COMMAND ----------

labels_str

# COMMAND ----------

labels

# COMMAND ----------

  df_train, df_predict = df_train_loaded, df_predict_loaded
  groupe = "DecPDV"
  preproc = Preprocessing2(groupe = groupe) 
  df_train = preproc.preproc(df_train,PCA = False)

  labels_str_out = []
  hist_data = []
  df_train["label"] = preproc.labels_str
  labels = list(set(df_train["label"]))
  for i,label in enumerate(labels_str):
    if len(df_train[df_train["label"] == label]) != 0:
      hist_data.append(df_train["Valeur"][df_train["label"] == label])
      labels_str_out.append(labels_str[i])
  fig = ff.create_distplot(hist_data, labels_str_out, bin_size=10000000)
  fig.update_layout(title = f"{groupe}")

  fig.show()

# COMMAND ----------

for groupe,df_train in zip(groupes,df_train_set):
  #df_train, df_predict = df_train_loaded, df_predict_loaded
  groupe = groupe
  preproc = Preprocessing2(groupe = groupe) 
  df_train = preproc.preproc(df_train,PCA = False)

  labels_str_out = []
  hist_data = []
  df_train["label"] = preproc.labels_str
  labels_str = list(set(df_train["label"]))
  for i,label in enumerate(labels_str):
    if len(df_train[df_train["label"] == label]) != 0:
      hist_data.append(df_train["Valeur"][df_train["label"] == label])
      labels_str_out.append(labels_str[i])
  fig = ff.create_distplot(hist_data, labels_str_out, bin_size=50000000)
  fig.update_layout(title = f"{groupe}")

  fig.show()

# COMMAND ----------

hist_data

# COMMAND ----------

set(df_train["label"])

# COMMAND ----------

df_train.drop()

# COMMAND ----------

import plotly.express as px
for groupe,df_train in zip(groupes,df_train_set):
  fig = px.line(df_train, x="DT_VALR", y="Valeur", title=f'{groupe}')
  fig.show()

# COMMAND ----------

import plotly.express as px
df_train = df_train_loaded
preproc = Preprocessing2(groupe = "DecPDV")
df_train = preproc.preproc(df_train)
df_train["label"] = preproc.labels_str
df_train["Day_Of_Week"] = preproc.day_of_week
df = px.data.tips()
fig = px.box(df_train, x="Day_Of_Week", y="Valeur", color="label", notched=True)
fig.show()

# COMMAND ----------

df_train

# COMMAND ----------

fig = px.line(df_train, x="DT_VALR", y="Valeur", title=f'{"EncUP"}')
fig.show()

# COMMAND ----------

describe = pd.DataFrame([])
for groupe,df_train in zip(groupes,df_train_loaded_set):
  describe[groupe] = df_train.describe()["Valeur"]

# COMMAND ----------

describe

# COMMAND ----------

describe

# COMMAND ----------

skew_and_kurtosis = pd.DataFrame([],columns = groupes,index = ["Skewness","Kurtosis"])
from scipy.stats import skew
from scipy.stats import norm, kurtosis

for groupe, df_train in zip(groupes,df_train_set):
  skew_and_kurtosis.loc["Skewness",groupe]= skew(df_train["Valeur"])
  skew_and_kurtosis.loc["Kurtosis",groupe] = kurtosis(df_train["Valeur"])

# COMMAND ----------

skew_and_kurtosis

# COMMAND ----------



# COMMAND ----------

describe.loc["count","EncPDV"]

# COMMAND ----------

describe

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/Prod2/ICF

# COMMAND ----------

df_train = df_train_loaded

# COMMAND ----------

qrf =  Conformal_Inference_qrf(groupe = groupe, cal_size = 0.2,mode = "test",test_size = None)
qrf.fit(df_train)

# COMMAND ----------

piqrf = Prediction_intervals(alphas = alphas,groupe = "DecPDV",models = ["QRF"],cal_size = 0.2,val_size = 0.1, mode = "test",gridsearch = True) # Initialisation
df_output = piqrf.split_conformal_ic(df_train,plot = True) # Prediction

# COMMAND ----------

fig = go.Figure()
error = []
# Add traces, one for each slider step
for step in range(100):

    if step == 0:
      fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                          mode='lines',
                          name='True'))
      

    #fig.add_vline(x=list(xX_test_1)[-1], line_width=3, line_dash="dash", line_color="black")


# Make 10th trace visible
#fig.data[0].visible = True


# Create and add slider
steps = []
for i in range(int(len(fig.data)/5)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": f"Conformal {model_name} IC predict, confidence_test_2 = {1 - error[i]}, expected = {confidence}"}],  # layout attribute
    )
    step["args"][0]["visible"][0] = True #True line always visible
    step["args"][0]["visible"][4*i+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][4*i+2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][4*i+3] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][4*i+4] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=1,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 1},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()

# COMMAND ----------

