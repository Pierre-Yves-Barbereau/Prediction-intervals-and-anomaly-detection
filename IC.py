# Databricks notebook source
import pandas as pd
import numpy as np
# ADB BASE
global NAME_ADB_BASE_CFM_IC
NAME_ADB_BASE_CFM_IC = "IntervalleConfiance"

# ADB TABLE
global NAME_ADB_TABLE_CFM_PRED
NAME_ADB_TABLE_CFM_PRED = "cfm_prediction"


df = spark.sql(f"SELECT * FROM {NAME_ADB_BASE_CFM_IC}.{NAME_ADB_TABLE_CFM_PRED}").toPandas()

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/IC_functions

# COMMAND ----------

df.head()

# COMMAND ----------


BASE_ = "default"
TABLE_ = "jours_target_csv"
jours_targets = spark.sql(f"SELECT * FROM {BASE_ }.{TABLE_ }")

# COMMAND ----------

jours_targets


# COMMAND ----------

from datetime import datetime,timedelta

# COMMAND ----------

#déviation moyenne des predictions des lundis en fonction de h
var_lundi_h = []
var_mardi_h = []
var_mercredi_h = []
var_jeudi_h = []
var_vendredi_h = []
var_samedi_h = []
var_dimanche_h = []
for h in range(84):
  var_lundi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 0][f"{h}"].var())
  var_mardi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 1][f"{h}"].var())
  var_mercredi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 2][f"{h}"].var())
  var_jeudi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 3][f"{h}"].var())
  var_vendredi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 4][f"{h}"].var())
  var_samedi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 5][f"{h}"].var())
  var_dimanche_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 6][f"{h}"].var())


# COMMAND ----------

import matplotlib.pyplot as plt
fig, ax = plt.subplots() 
ax.plot(var_lundi_h, label = "lundi")
ax.plot(var_mardi_h,label = "mardi")
ax.plot(var_mercredi_h,label = "mercredi")
ax.plot(var_jeudi_h,label = "jeudi")
ax.plot(var_vendredi_h,label = "vendredi")
ax.plot(var_samedi_h,label = "samedi")
ax.plot(var_dimanche_h,label = "dimanche")
ax.set_title("Variance des prédictions en fonction de h par jour de la semaine")
ax.legend()

# COMMAND ----------

def quantiles_by_weekday(alphas,df):
  """return 6 dataframes ( 1 for each day ) with in columns the values of alpha and in row the 84 quantiles for each pred time"""
  q_lundi_df =  pd.DataFrame([])
  q_mardi_df =  pd.DataFrame([])
  q_mercredi_df = pd.DataFrame([])
  q_jeudi_df =  pd.DataFrame([])
  q_vendredi_df = pd.DataFrame([])
  q_samedi_df = pd.DataFrame([])
  q_dimanche_df = pd.DataFrame([])

  for alpha in alphas :
    q_lundi_h = []
    q_mardi_h = []
    q_mercredi_h = []
    q_jeudi_h = []
    q_vendredi_h = []
    q_samedi_h = []
    q_dimanche_h = []
    for h in range(84):
      q_lundi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 0][f"{h}"].quantile(alpha))
      q_mardi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 1][f"{h}"].quantile(alpha))
      q_mercredi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 2][f"{h}"].quantile(alpha))
      q_jeudi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 3][f"{h}"].quantile(alpha))
      q_vendredi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 4][f"{h}"].quantile(alpha))
      q_samedi_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 5][f"{h}"].quantile(alpha))
      q_dimanche_h.append(df[(df["PredDebutDate"]+timedelta(days = h)).dt.weekday   == 6][f"{h}"].quantile(alpha))

    q_lundi_df[f"{alpha}"] = q_lundi_h
    q_mardi_df[f"{alpha}"] = q_mardi_h
    q_mercredi_df[f"{alpha}"] = q_mercredi_h
    q_jeudi_df[f"{alpha}"] = q_jeudi_h
    q_vendredi_df[f"{alpha}"] = q_vendredi_h
    q_samedi_df[f"{alpha}"] = q_samedi_h
    q_dimanche_df[f"{alpha}"] = q_dimanche_h
  return q_lundi_df,q_mardi_df,q_mercredi_df, q_jeudi_df, q_vendredi_df, q_samedi_df, q_dimanche_df



# COMMAND ----------

confidence = 0.9
alphas = [0.05,0.95] #alpha-Quantiles
alphas = [round(((1-confidence)/2),2),round((1-(1-confidence)/2),2)] #Symetric IC
q_lundi_df,q_mardi_df,q_mercredi_df, q_jeudi_df, q_vendredi_df, q_samedi_df, q_dimanche_df = quantiles_by_weekday(alphas,df)
fig, ax = plt.subplots(2, 3, sharex='row', sharey=False, figsize=(20, 10)) 
for alpha in alphas : 
  ax[0,0].plot(q_lundi_df[f"{alpha}"], label = f"alpha = {alpha}")
  ax[0,1].plot(q_mardi_df[f"{alpha}"],label = f"alpha = {alpha}")
  ax[0,2].plot(q_mercredi_df[f"{alpha}"],label = f"alpha = {alpha}")
  ax[1,0].plot(q_jeudi_df[f"{alpha}"],label = f"alpha = {alpha}")
  ax[1,1].plot(q_vendredi_df[f"{alpha}"],label = f"alpha = {alpha}")
  ax[1,2].plot(q_samedi_df[f"{alpha}"],label = f"alpha = {alpha}")
  ax[0,0].set_title(f"{confidence} IC for lundis")
  ax[0,1].set_title(f"{confidence} IC for mardis")
  ax[0,2].set_title(f"{confidence} IC for mercredis")
  ax[1,0].set_title(f"{confidence} IC for jeudis")
  ax[1,1].set_title(f"{confidence} IC for vendredis")
  ax[1,2].set_title(f"{confidence} IC for samedis")
  ax[0,0].legend(loc ="upper left")
  ax[0,1].legend(loc ="upper left")
  ax[0,2].legend(loc ="upper left")
  ax[1,0].legend(loc ="upper left")
  ax[1,1].legend(loc ="upper left")
  ax[1,2].legend(loc ="upper left")

# COMMAND ----------

def Pred_quantile_by_first_weekday(df,alpha):
  df_predDebutDate_lundi = df[["PredDebutDate"].dt.weekday == 0]
  df_predDebutDate_lundi = df[["PredDebutDate"].dt.weekday == 0].apply(np.quantile(alpha),axis = 1)

# COMMAND ----------

def alpha_quantile_by_first_day_of_pred (first_day:int,alpha:float):
  "return alpha-quantile for first day = day"
  return df[df["PredDebutDate"].dt.weekday == first_day].iloc[:,4:].quantile(alpha, axis = 0)

# COMMAND ----------

fig, ax = plt.subplots(2, 3, sharex='row', sharey=False, figsize=(20, 10)) 
colors = ["red","green"]
for alpha,color in zip(list(reversed(alphas)),list(reversed(colors))) : 
  ax[0,0].plot(alpha_quantile_by_first_day_of_pred (0,alpha), label = f"{alpha}-Quantile",color = color)
  ax[0,1].plot(alpha_quantile_by_first_day_of_pred (1,alpha),label = f"{alpha}-Quantile",color = color)
  ax[0,2].plot(alpha_quantile_by_first_day_of_pred (2,alpha),label = f"{alpha}-Quantile",color = color)
  ax[1,0].plot(alpha_quantile_by_first_day_of_pred (3,alpha),label = f"{alpha}-Quantile",color = color)
  ax[1,1].plot(alpha_quantile_by_first_day_of_pred (4,alpha),label = f"{alpha}-Quantile",color = color)
  ax[1,2].plot(alpha_quantile_by_first_day_of_pred (5,alpha),label = f"{alpha}-Quantile",color = color)
  ax[0,0].set_title(f"{confidence} IC for first day = lundis")
  ax[0,1].set_title(f"{confidence} IC for first day = mardis")
  ax[0,2].set_title(f"{confidence} IC for first day = mercredis")
  ax[1,0].set_title(f"{confidence} IC for first day = jeudis")
  ax[1,1].set_title(f"{confidence} IC for first day = vendredis")
  ax[1,2].set_title(f"{confidence} IC for first day = samedis")
  ax[0,0].legend(loc ="upper left")
  ax[0,1].legend(loc ="upper left")
  ax[0,2].legend(loc ="upper left")
  ax[1,0].legend(loc ="upper left")
  ax[1,1].legend(loc ="upper left")
  ax[1,2].legend(loc ="upper left")

ax[0,0].fill_between([i for i in range(84)], alpha_quantile_by_first_day_of_pred (0,alphas[0]),alpha_quantile_by_first_day_of_pred (0,alphas[1]), color="blue",alpha = 0.3)

# COMMAND ----------

def plot_IC(confidences):
  color = "blue"
  fig, ax = plt.subplots(2, 3, sharex='row', sharey=False, figsize=(20, 10)) 
  for i,confidence in enumerate(confidences) :
    alphas = [round(((1-confidence)/2),2),round((1-(1-confidence)/2),2)] #Symetric IC
    ax[0,0].fill_between([i for i in range(84)], alpha_quantile_by_first_day_of_pred (0,alphas[0]),alpha_quantile_by_first_day_of_pred (0,alphas[1]), color=color,alpha = (1 + i)/(5*len(confidences)),label = f"{confidence} IC")
    ax[0,1].fill_between([i for i in range(84)], alpha_quantile_by_first_day_of_pred (1,alphas[0]),alpha_quantile_by_first_day_of_pred (1,alphas[1]), color=color,alpha = (1 + i)/(5*len(confidences)),label = f"{confidence} IC")
    ax[0,2].fill_between([i for i in range(84)], alpha_quantile_by_first_day_of_pred (2,alphas[0]),alpha_quantile_by_first_day_of_pred (2,alphas[1]), color=color,alpha = (1 + i)/(5*len(confidences)),label = f"{confidence} IC")
    ax[1,0].fill_between([i for i in range(84)], alpha_quantile_by_first_day_of_pred (3,alphas[0]),alpha_quantile_by_first_day_of_pred (3,alphas[1]), color=color,alpha = (1 + i)/(5*len(confidences)),label = f"{confidence} IC")
    ax[1,1].fill_between([i for i in range(84)], alpha_quantile_by_first_day_of_pred (4,alphas[0]),alpha_quantile_by_first_day_of_pred (4,alphas[1]), color=color,alpha = (1 + i)/(5*len(confidences)),label = f"{confidence} IC")
    ax[1,2].fill_between([i for i in range(84)], alpha_quantile_by_first_day_of_pred (5,alphas[0]),alpha_quantile_by_first_day_of_pred (5,alphas[1]), color=color,alpha = (1 + i)/(5*len(confidences)),label = f"{confidence} IC")
  ax[0,0].set_title(f"{confidence} IC for first day = lundis")
  ax[0,1].set_title(f"{confidence} IC for first day = mardis")
  ax[0,2].set_title(f"{confidence} IC for first day = mercredis")
  ax[1,0].set_title(f"{confidence} IC for first day = jeudis")
  ax[1,1].set_title(f"{confidence} IC for first day = vendredis")
  ax[1,2].set_title(f"{confidence} IC for first day = samedis")
  ax[0,0].legend(loc ="upper left")
  ax[0,1].legend(loc ="upper left")
  ax[0,2].legend(loc ="upper left")
  ax[1,0].legend(loc ="upper left")
  ax[1,1].legend(loc ="upper left")
  ax[1,2].legend(loc ="upper left")
plot_IC([0.95,0.9,0.75,0.5])

# COMMAND ----------

import pandas as pd
#Nom Bases
NAME_ADB_BASE_CFM_IC = "IntervalleConfiance"

#Nom table

#Nom des vues à créer
NAME_ADB_VIEW_FLUX_HISTORIQUE = "cfm_flux_historique"
NAME_ADB_VIEW_FLUX_FIN_TRAN = "cfm_fin_tran"

df = spark.sql(f"SELECT * FROM {NAME_ADB_BASE_CFM_IC}.{NAME_ADB_VIEW_FLUX_HISTORIQUE} WHERE descriptionFlux = 'DecPDV' ").toPandas()
lendf = df.shape[0]
pd.to_datetime(df["DT_VALR"])
df.sort_values(by = "DT_VALR")
df.head()

# COMMAND ----------

target = df["Valeur"]
df2 = pd.DataFrame([])
df2["DT_VALR"] = pd.to_datetime(df["DT_VALR"])
df2["Valeur"] = df["Valeur"]
df2["Jour_de_la_semaine"] = pd.to_datetime(df['DT_VALR']).dt.weekday
df2["Jour"] = pd.to_datetime(df['DT_VALR']).dt.day
df2["Mois"] = pd.to_datetime(df['DT_VALR']).dt.month
df2

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2, target,train_size = 0.8,shuffle = False)
y_test = list(y_test)
xX_train = X_train["DT_VALR"]
X_train = X_train.drop(["DT_VALR"],axis = 1)
X_train = X_train.drop(["Valeur"],axis = 1)
xX_test = X_test["DT_VALR"]
X_test = X_test.drop(["DT_VALR"],axis = 1)
X_test = X_test.drop(["Valeur"],axis = 1)


# COMMAND ----------

X_test

# COMMAND ----------

df2

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go
GB = pd.DataFrame([])
GB["DT_VALR"] = xX_test
GB["Valeur"] = y_test
fig = go.Figure()
fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                    mode='lines',
                    name='True'))
for alpha in alphas :
  gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha,n_estimators = 1000,random_state = 42,learning_rate = 0.0092)
  gbr.fit(X_train,y_train)
  GB[f"Q_{alpha}"] = gbr.predict(X_test)
  
  fig.add_trace(go.Scatter(x=GB["DT_VALR"], y=GB[f"Q_{alpha}"],
                      mode='lines',
                      name=f'gbr_{alpha}'))
print("GB_error = {}".format((np.sum(GB["Valeur"]<GB["Q_0.05"]) + np.sum(GB["Valeur"]>GB["Q_0.95"]))/GB["Valeur"].shape[0]))   
error = (np.sum(GB["Valeur"]<GB["Q_0.05"]) + np.sum(GB["Valeur"]>GB["Q_0.95"]))/GB["Valeur"].shape[0]              
fig.update_layout(title = f"Gradient Boosting IC predict, confidence = {1 - error} / 90")                      
fig.show()


# COMMAND ----------

#pip install quantile-forest

# COMMAND ----------

from quantile_forest import RandomForestQuantileRegressor
qrf = RandomForestQuantileRegressor()
qrf.fit(X_train, y_train)
y_pred = qrf.predict(X_test, quantiles=alphas)

# COMMAND ----------

from lightgbm import LGBMRegressor
LGBM = pd.DataFrame([])
LGBM["DT_VALR"] = xX_test
LGBM["Valeur"] = y_test
fig = go.Figure()
fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                    mode='lines',
                    name='True'))
for alpha in alphas :
  lgbm = LGBMRegressor(alpha=alpha,  objective='quantile')
  lgbm.fit(X_train,y_train)
  LGBM[f"Q_{alpha}"] = lgbm.predict(X_test)
  fig.add_trace(go.Scatter(x=LGBM["DT_VALR"], y=LGBM[f"Q_{alpha}"],
                      mode='lines',
                      name=f'LGBM_{alpha}'))
print("LGBM_error = {}".format((np.sum(LGBM["Valeur"]<LGBM["Q_0.05"]) + np.sum(LGBM["Valeur"]>LGBM["Q_0.95"]))/LGBM["Valeur"].shape[0]))
fig.update_layout(title = "Light GBM IC predict")                      
fig.show()


# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/IC_functions
# MAGIC

# COMMAND ----------

IC(GradientBoostingRegressor,alphas,df)

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
GB = pd.DataFrame([])
GB["DT_VALR"] = xX_test
GB["Valeur"] = y_test
alphas_full = np.arange(0.025,1,0.025)
alphas_full = [np.round(alpha,3) for alpha in alphas_full]
for alpha in alphas_full :
  print(alpha)
  gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha,random_state=42)
  gbr.fit(X_train,y_train)
  GB[f"Q_{alpha}"] = gbr.predict(X_test)


# COMMAND ----------

GB.iloc[:,2:].apply(np.sort,axis = 1)

# COMMAND ----------

GB.iloc[:,2:]

# COMMAND ----------

import plotly.graph_objects as go
import numpy as np

# Create figure
fig = go.Figure()

# Add traces, one for each slider step
for step in alphas_full:
    fig.add_trace(
        go.Scatter(
            visible=False,
            mode='lines',
            name=str(step)+"-quantile",
            x=GB["DT_VALR"],
            y=GB[f"Q_{step}"]))

# Make 10th trace visible
fig.data[0].visible = True
fig.data[len(alphas_full)-1].visible = True

# Create and add slider
steps = []
for i in list(reversed(range(round(len(fig.data)/2)))):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": f"Confiance : {5*(round(len(fig.data)/2)-i-1)}%"}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][-i-1] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=19,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)
fig.update_layout(
    title="Confiance : 95%"
)
fig.show()

# COMMAND ----------

functions_set = [list(GB["Q_0.025"]),list(GB["Q_0.05"]),list(GB["Q_0.075"]),list(GB["Q_0.1"])]
functions_set

# COMMAND ----------

def pinball_loss(beta,theta,alpha):
  return (alpha*(beta-theta) - min(0,beta-theta))
pinball_loss(0.045,0.055,0.05)

# COMMAND ----------

if 1:
  def boa(X,y,loss,eta,functions_set,alpha):
    import math
    y = list(y)
    J = len(functions_set)
    #loss(beta,theta) = alpha*(beta-theta) - np.min(0,beta-theta)
    pi_t_j = [np.ones(len(functions_set))/len(functions_set)]
    l_t_j = []
    for t in range(len(X)):
      epi = np.dot(pi_t_j[-1],[pinball_loss(y[t],functions_set[i][t],alpha = alpha) for i in range(J)])
      l_t_j.append([pinball_loss(y[t],functions_set[i][t],alpha = alpha) - epi for i in range(J)])
      regularisation = np.sum([np.exp(-eta*l_t_j[-1][j]*(1 + eta*l_t_j[-1][j]))*pi_t_j[-1][j] for j in range(J)])
      pi_t_j.append([np.exp(-eta*l_t_j[-1][j]*(1 + eta*l_t_j[-1][j]))*pi_t_j[-1][j] / regularisation for j in range(J)])
    return pi_t_j
boa(X_test,y_test,pinball_loss,eta = 0.0000001,functions_set = functions_set,alpha = 0.05)

# COMMAND ----------

def faboa(X,y,loss,functions_set,alpha): # not terminated, t-1 to do
  import math
  y = list(y)
  J = len(functions_set)
  #loss(beta,theta) = alpha*(beta-theta) - np.min(0,beta-theta)
  pi_t_j = [np.ones(len(functions_set))/len(functions_set)]
  L_t_j = [list(np.ones(J))]
  n_t_j = [list(np.divide(np.ones(J),1000000000))]
  E_t_j = [list(np.ones(J))]
  l_t_j = []
  for t in range(len(X)):
    print("pinball_loss = ",[pinball_loss(y[t],functions_set[j][t],alpha = alpha) for j in range(J)])
    epi = np.dot(pi_t_j[-1],[pinball_loss(y[t],functions_set[j][t],alpha = alpha) for j in range(J)])
    print("pi_t_j = ",pi_t_j)
    print("epi = ",epi)
    l_t_j.append([pinball_loss(y[t],functions_set[j][t],alpha = alpha) - epi for j in range(J)])
    print("L_t_j = ",[L_t_j[-1][j] for j in range(J)])
    print("l_t_j = ",[l_t_j[-1][j] for j in range(J)])
    print("n_t_j = ",[n_t_j[-1][j] for j in range(J)])
    L_t_j.append([L_t_j[-1][j] + l_t_j[-1][j]*(1 + n_t_j[-1][j]*l_t_j[-1][j])/2 + E_t_j[-1][j]*(n_t_j[-1][j]*l_t_j[-1][j]>0.5) for j in range(J)])
    m = [np.max([l_t_j[s][j] for s in range(t)]) for j in range(J)]
    
    n_t_j.append([min(1/E_t_j[-1][j], math.sqrt(math.log(1/pi_t_j[0][j])/np.sum([l_t_j[s][j] for s in range(t)]))) for j in range(J)])
    
    regularisation = np.sum([np.exp(-n_t_j[-1][j]*l_t_j[-1][j]*(1 + n_t_j[-1][j]*l_t_j[-1][j]))*pi_t_j[-1][j] for j in range(J)])
    pi_t_j.append([np.exp(-n_t_j[-1][j]*l_t_j[-1][j]*(1 + n_t_j[-1][j]*l_t_j[-1][j]))*pi_t_j[-1][j] / regularisation for j in range(J)])

  return pi_t_j

# COMMAND ----------

faboa(np.divide(X_test,1000000),np.divide(y_test,1000000),pinball_loss,functions_set = functions_set,alpha = 0.05)