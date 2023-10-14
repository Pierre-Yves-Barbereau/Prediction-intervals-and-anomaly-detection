# Databricks notebook source
import numpy as np
confidence = 0.9
alphas = [round(((1-confidence)/2),2),round((1-(1-confidence)/2),2)]

alphas = [0.05,0.95]
confidence = np.round(alphas[1]-alphas[0],2)

gamma=0.005
t = 50

import pandas as pd
import plotly.graph_objects as go
#Nom Bases
NAME_ADB_BASE_CFM_IC = "IntervalleConfiance"

#Nom table

#Nom des vues à créer
NAME_ADB_VIEW_FLUX_HISTORIQUE = "cfm_flux_historique"
NAME_ADB_VIEW_FLUX_FIN_TRAN = "cfm_fin_tran"

#EncPDV DecPDV EncUP DecUP EncPet DecPet DecTaxAlc DecTaxPet EncRist EncRprox

df = spark.sql(f"SELECT * FROM {NAME_ADB_BASE_CFM_IC}.{NAME_ADB_VIEW_FLUX_HISTORIQUE} WHERE descriptionFlux = 'DecPDV' ").toPandas()
lendf = df.shape[0]
df["DT_VALR"].apply(pd.to_datetime)
df.sort_values(by = "DT_VALR")
df

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/IC_functions

# COMMAND ----------

xX_train, xX_cal, xX_val, xX_test, X_train, X_cal, X_val, X_test, y_train, y_cal, y_val, y_test = train_cal_val_test_split(df,0.5,0.33,0.5)
len(y_test)
y_test = list(y_test)

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
model_name = "Gradient_Boosting"
#Entrainement des modèles de regression quantile pour chacune des bornes de l'intervalle
model_up = GradientBoostingRegressor(loss="quantile", alpha=alphas[1],n_estimators = 1000, learning_rate = 0.01).fit(X_train,y_train)
model_down = GradientBoostingRegressor(loss="quantile", alpha=alphas[0],n_estimators = 1000, learning_rate = 0.01).fit(X_train,y_train)

# COMMAND ----------

def boa(X,y,loss,eta,functions_set,alpha): # not terminated, t-1 to do
  import math
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

# COMMAND ----------

def AgACI(gammas : list):
  #Initialisation 
  y = list(y_test_1)

  
  for t in y_test_2:
    #AgACI init
    low_t = []
    high_t = []
    omega_t_low = []
    omega_t_high = []
    
    low = pred_down_test_2 - np.quantile(f_conformity_score_low(pred_down_test_1,y_test_1),1-alpha_star_low_updated)
    high = pred_up_test_2 + np.quantile(f_conformity_score_high(pred_up_test_1,y_test_1),1-alpha_star_high_updated)

    #Boa init
    pi_t_j = [np.ones(len(gammas))/len(gammas)]
    l_t_j = []


    for gamma in gammas:

      alpha_star_low_upd = alpha_star_low_updated + gamma*(alphas[0] - np.mean([err_t_low(y_test_1,pred_down_test_1,i) for i in range(20)]))
      if 0<alpha_star_low_upd <1:
        low = pred_down_test_2 - np.quantile(f_conformity_score_low(pred_down_test_1,y_test_1),1-alpha_star_low_upd)
      else :
        print("alpha_star_low_updated out of 0-1")

      alpha_star_high_upd = alpha_star_high_updated + gamma*(1-alphas[1] - np.mean([err_t_high(y_test_1,pred_up_test_1,i) for i in range(20)]))
      if 0<alpha_star_high_upd <1:
        high = pred_up_test_2 + np.quantile(f_conformity_score_high(pred_up_test_1,y_test_1),1-alpha_star_high_upd)
      else :
        print("alpha_star_high_updated out of 0-1")

      #Boa
      epi = np.dot(pi_t_j[-1],[pinball_loss(y[t],functions_set[i][t],alpha = alpha) for i in range(J)])
      l_t_j.append([pinball_loss(y[t],functions_set[i][t],alpha = alpha) - epi for i in range(J)])
      regularisation = np.sum([np.exp(-eta*l_t_j[-1][j]*(1 + eta*l_t_j[-1][j]))*pi_t_j[-1][j] for j in range(J)])
      pi_t_j.append([np.exp(-eta*l_t_j[-1][j]*(1 + eta*l_t_j[-1][j]))*pi_t_j[-1][j] / regularisation for j in range(J)])


      #omega_t_low.append()
      #omega_t_high.append()
      low_t.append(low)
      high_t.append(high)
    C_low = np.mean(low_t,axis = 0)
    C_high = np.mean(high_t,axis = 0)
  return C_low,C_high