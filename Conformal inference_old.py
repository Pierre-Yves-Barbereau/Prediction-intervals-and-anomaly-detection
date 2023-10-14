# Databricks notebook source
# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/preprocessing

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/IC_functions_old

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/IC_functions

# COMMAND ----------



# COMMAND ----------

#Variables widget
groupe = lib_instance.define_widget("groupe") #DecPDV
confidence = float(lib_instance.define_widget('confidence'))/100
val_size = int(lib_instance.define_widget("val_size")) #365
cal_size = int(lib_instance.define_widget("cal_size")) # 365
#PAth notebooks
path_notebook_preproc_preprocessing = lib_instance.define_widget("path_notebook_preproc_preprocessing") #'/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing'

alphas = [0.05,0.95]
confidence = np.round(alphas[1]-alphas[0],2)

gamma=0.005
t = 50



# COMMAND ----------

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

# COMMAND ----------

preproc = preprocessing(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
dataframe_dectrain = preproc.load_train(groupe = groupe)

# COMMAND ----------

alphas = [np.round((1-confidence)/2,2),np.round(1-(1-confidence)/2,2)]

# COMMAND ----------

ic = Confidence_Interval(alphas = alphas, cal_size = cal_size,val_size = val_size)

# COMMAND ----------

#xX_train, xX_cal, xX_val, xX_test, X_train, X_cal, X_val, X_test, y_train, y_cal, y_val, y_test = train_cal_val_test_split(dataframe_dectrain,0.5,0.33,0.5)


# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
model_name = "Gradient_Boosting"
#Entrainement des modèles de regression quantile pour chacune des bornes de l'intervalle
model_up = GradientBoostingRegressor(loss="quantile", alpha=alphas[1],n_estimators = 1000, learning_rate = 0.01).fit(X_train,y_train)
model_down = GradientBoostingRegressor(loss="quantile", alpha=alphas[0],n_estimators = 1000, learning_rate = 0.01).fit(X_train,y_train)

# COMMAND ----------

!pip install quantile-forest

# COMMAND ----------

from quantile_forest import RandomForestQuantileRegressor
qrf = RandomForestQuantileRegressor()
qrf.fit(X_train, y_train)
y_pred = qrf.predict(X_test, quantiles=alphas)

# COMMAND ----------



# COMMAND ----------

#Prédiction sur le set de calibration
pred_down_cal = model_down.predict(X_cal)
pred_up_cal = model_up.predict(X_cal)

fig = go.Figure()
fig.add_trace(go.Scatter(x=xX_cal, y=y_cal,
                    mode='lines',
                    name='True'))
fig.add_trace(go.Scatter(x=xX_cal, y=pred_down_cal,
                      mode='lines',
                      name=f'{model_name}_{alphas[0]}'))
fig.add_trace(go.Scatter(x=xX_cal, y=pred_up_cal,
                      mode='lines',
                      name=f'{model_name}_{alphas[1]}'))
error = error = (np.sum(y_cal<pred_down_cal) + np.sum(y_cal>pred_up_cal))/len(y_cal) 
fig.update_layout(title = f"{model_name} cal predict, confidence = {1 - error}, expected = {confidence}")  
fig.show()

# COMMAND ----------

import matplotlib.pyplot as plt

plt.plot(f_conformity_score(pred_down_cal,pred_up_cal,y_cal))

# COMMAND ----------

conformity_score = f_conformity_score(pred_down_cal,pred_up_cal,y_cal)
conformity_score_low = f_conformity_score_low(pred_down_cal,y_cal)
conformity_score_high = f_conformity_score_high(pred_up_cal,y_cal)

# COMMAND ----------

pred_down_val = model_down.predict(X_val)
pred_up_val = model_up.predict(X_val)

# COMMAND ----------

betas = np.arange(0,1,0.0001)
alpha_star = np.max([b for b in np.arange(0,1,0.0001) if (miscoverage_rate(pred_down_cal,pred_up_cal,pred_down_val,pred_up_val,conformity_score,alpha = b) < (1 - confidence))])
alpha_star_low = np.max([b for b in np.arange(0,1,0.0001) if (miscoverage_rate_low(pred_down_val,conformity_score_low,alpha = b) < alphas[0])])
alpha_star_high = np.max([b for b in np.arange(0,1,0.0001) if (miscoverage_rate_high(pred_up_val,conformity_score_high,alpha = b) < 1 - alphas[1])])
alpha_star_low_updated = alpha_star_low
alpha_star_high_updated = alpha_star_high
alpha_star_updated = alpha_star

# COMMAND ----------

pred_down_test = model_down.predict(X_test)
pred_up_test = model_up.predict(X_test)

# COMMAND ----------

fig = go.Figure()
fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                   mode='lines',
                    name='True'))
fig.add_trace(go.Scatter(x=xX_test, y=pred_down_test,
                      mode='lines',
                      name=f'{model_name}_{alphas[0]}'))
fig.add_trace(go.Scatter(x=xX_test, y=pred_up_test,
                      mode='lines',
                      name=f'{model_name}_{alphas[1]}'))

error = error = (np.sum(y_test<pred_down_test) + np.sum(y_test>pred_up_test))/len(y_test) 
fig.update_layout(title = f"Non-conformal {model_name} IC predict, confidence = {1 - error}, expected = {confidence}")  
fig.show()

# COMMAND ----------

fig = go.Figure()
pred_down_test_conformal = pred_down_test - np.quantile(conformity_score,1-alpha_star)
pred_up_test_conformal = pred_up_test + np.quantile(conformity_score,1-alpha_star)
fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                    mode='lines',
                    name='True'))
fig.add_trace(go.Scatter(x=xX_test, y=pred_down_test_conformal,
                      mode='lines',
                      name=f'{model_name}_Conformalised_{alphas[0]}'))
fig.add_trace(go.Scatter(x=xX_test, y=pred_up_test_conformal,
                      mode='lines',
                      name=f'{model_name}_Conformalised_{alphas[1]}'))

error = error = (np.sum(y_test<pred_down_test_conformal) + np.sum(y_test>pred_up_test_conformal))/len(y_test) 
fig.update_layout(title = f"Conformal {model_name} IC predict, confidence = {1 - error}, expected = {confidence}")  
fig.show()

# COMMAND ----------

fig = go.Figure()
pred_down_test_conformal = pred_down_test - np.quantile(conformity_score_low,1-alpha_star_low)
pred_up_test_conformal = pred_up_test + np.quantile(conformity_score_high,1-alpha_star_high)
fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                    mode='lines',
                    name='True'))
fig.add_trace(go.Scatter(x=xX_test, y=pred_down_test_conformal,
                      mode='lines',
                      name=f'{model_name}_Conformalised_{alphas[0]}'))
fig.add_trace(go.Scatter(x=xX_test, y=pred_up_test_conformal,
                      mode='lines',
                      name=f'{model_name}_Conformalised_{alphas[1]}'))

error = error = (np.sum(y_test<pred_down_test_conformal) + np.sum(y_test>pred_up_test_conformal))/len(y_test) 
fig.update_layout(title = f"Conformal asymetric {model_name} IC predict, confidence = {1 - error}, expected = {confidence}")  
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adaptative Conformal Inférence

# COMMAND ----------

X_test_full = X_test
X_test_full["DT_VALR"] = xX_test
from sklearn.model_selection import train_test_split
X_test_1, X_test_2, y_test_1, y_test_2 = train_test_split(X_test_full,y_test,train_size = 0.5,shuffle = False)
xX_test_1 = X_test_1["DT_VALR"]
X_test_1 = X_test_1.drop(["DT_VALR"],axis = 1)
xX_test_2 = X_test_2["DT_VALR"]
X_test_2 = X_test_2.drop(["DT_VALR"],axis = 1)

# COMMAND ----------

X_test_1

# COMMAND ----------

X_test_full = X_test
X_test_full["DT_VALR"] = xX_test
from sklearn.model_selection import train_test_split
X_test_1, X_test_2, y_test_1, y_test_2 = train_test_split(X_test_full,y_test,train_size = 0.5,shuffle = False)
xX_test_1 = X_test_1["DT_VALR"]
X_test_1 = X_test_1.drop(["DT_VALR"],axis = 1)
xX_test_2 = X_test_2["DT_VALR"]
X_test_2 = X_test_2.drop(["DT_VALR"],axis = 1)

pred_down_test_1 = model_down.predict(X_test_1)
pred_up_test_1 = model_up.predict(X_test_1)

pred_down_test_2 = model_down.predict(X_test_2)
pred_up_test_2 = model_up.predict(X_test_2)

pred_down_test_1_conformal = pred_down_test_1 - np.quantile(conformity_score,1-alpha_star)
pred_up_test_1_conformal = pred_up_test_1 + np.quantile(conformity_score,1-alpha_star)


alpha_star_upd = alpha_star_updated +gamma*(alpha_star - np.sum([err_t(y_test_1,pred_up_test_1,pred_down_test_1,i) for i in range(t)])/t)
if 0<alpha_star_upd<1:
  alpha_star_updated = alpha_star_upd

pred_down_test_2_adaptative_conformal = pred_down_test_2 - np.quantile(conformity_score,1-alpha_star_updated)
pred_up_test_2_adaptative_conformal = pred_up_test_2 + np.quantile(conformity_score,1-alpha_star_updated)

fig = go.Figure()

fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                    mode='lines',
                    name='True'))
fig.add_trace(go.Scatter(x=xX_test_1, y=pred_down_test_1_conformal,
                      mode='lines',
                      name=f'{model_name}_Conformalised_{alphas[0]}'))
fig.add_trace(go.Scatter(x=xX_test_1, y=pred_up_test_1_conformal,
                      mode='lines',
                      name=f'{model_name}_Conformalised_{alphas[1]}'))

fig.add_trace(go.Scatter(x=xX_test_2, y=pred_down_test_2_adaptative_conformal,
                      mode='lines',
                      name=f'{model_name}_Adaptative_Conformalised_{alphas[0]}'))
fig.add_trace(go.Scatter(x=xX_test_2, y=pred_up_test_2_adaptative_conformal,
                      mode='lines',
                      name=f'{model_name}_Adaptative_Conformalised_{alphas[1]}'))

fig.add_vline(x=list(xX_test_1)[-1], line_width=3, line_dash="dash", line_color="black")

error = (np.sum(y_test_2<pred_down_test_2_adaptative_conformal) + np.sum(y_test_2>pred_up_test_2_adaptative_conformal))/len(y_test_2) 
fig.update_layout(title = f"Conformal {model_name} IC predict, confidence = {1 - error}, expected = {confidence}")  
fig.show()

# COMMAND ----------

import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
import math

X_test_full = X_test
X_test_full["DT_VALR"] = xX_test
# Create figure
fig = go.Figure()
error = []
# Add traces, one for each slider step
for step in range(int(len(X_test)/2 -1)):
    
    
    X_test_1, X_test_2, y_test_1, y_test_2 = train_test_split(X_test_full,y_test,test_size = int(len(X_test)/2) - step,shuffle = False)
    xX_test_1 = X_test_1["DT_VALR"]
    X_test_1 = X_test_1.drop(["DT_VALR"],axis = 1)
    xX_test_2 = X_test_2["DT_VALR"]
    X_test_2 = X_test_2.drop(["DT_VALR"],axis = 1)

    pred_down_test_1 = model_down.predict(X_test_1)
    pred_up_test_1 = model_up.predict(X_test_1)

    pred_down_test_2 = model_down.predict(X_test_2)
    pred_up_test_2 = model_up.predict(X_test_2)



    pred_down_test_1_conformal = pred_down_test_1 - np.quantile(f_conformity_score(pred_down_cal,pred_up_cal,y_cal),1-alpha_star)
    pred_up_test_1_conformal = pred_up_test_1 + np.quantile(f_conformity_score(pred_down_cal,pred_up_cal,y_cal),1-alpha_star)

    
    alpha_star_upd = alpha_star_updated +gamma*(1-confidence - np.mean([err_t(y_test_1,pred_up_test_1,pred_down_test_1,i) for i in range(t)]))
    if 0<alpha_star_upd<1:
      alpha_star_updated = alpha_star_upd
    else :
      print(alpha_star_upd)

    pred_down_test_2_adaptative_conformal = pred_down_test_2 - np.quantile(f_conformity_score(pred_down_test_1,pred_up_test_1,y_test_1),1-alpha_star_updated)
    pred_up_test_2_adaptative_conformal = pred_up_test_2 + np.quantile(f_conformity_score(pred_down_test_1,pred_up_test_1,y_test_1),1-alpha_star_updated)

    error.append((np.sum(y_test_2<pred_down_test_2_adaptative_conformal) + np.sum(y_test_2>pred_up_test_2_adaptative_conformal))/len(y_test_2))

    if step == 0:
      fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                          mode='lines',
                          name='True'))
    fig.add_trace(go.Scatter(x=xX_test_1, y=pred_down_test_1_conformal,
                          mode='lines',
                          name=f'{model_name}_Conformalised_{alphas[0]}'))
    fig.add_trace(go.Scatter(x=xX_test_1, y=pred_up_test_1_conformal,
                          mode='lines',
                          name=f'{model_name}_Conformalised_{alphas[1]}'))

    fig.add_trace(go.Scatter(x=xX_test_2, y=pred_down_test_2_adaptative_conformal,
                          mode='lines',
                          name=f'{model_name}_Adaptative_Conformalised_{alphas[0]}'))
    fig.add_trace(go.Scatter(x=xX_test_2, y=pred_up_test_2_adaptative_conformal,
                          mode='lines',
                          name=f'{model_name}_Adaptative_Conformalised_{alphas[1]}'))

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

import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
import math

X_test_full = X_test
X_test_full["DT_VALR"] = xX_test
# Create figure
fig = go.Figure()
# Add traces, one for each slider step

for step in range(int(len(X_test)/2 -1)):
    
    
    X_test_1, X_test_2, y_test_1, y_test_2 = train_test_split(X_test_full,y_test,test_size = int(len(X_test)/2) - step,shuffle = False)
    xX_test_1 = X_test_1["DT_VALR"]
    X_test_1 = X_test_1.drop(["DT_VALR"],axis = 1)
    xX_test_2 = X_test_2["DT_VALR"]
    X_test_2 = X_test_2.drop(["DT_VALR"],axis = 1)

    pred_down_test_1 = model_down.predict(X_test_1)
    pred_up_test_1 = model_up.predict(X_test_1)

    pred_down_test_2 = model_down.predict(X_test_2)
    pred_up_test_2 = model_up.predict(X_test_2)



    pred_down_test_1_conformal = pred_down_test_1 - np.quantile(f_conformity_score_low(pred_down_cal,y_cal),1-alpha_star_low)
    pred_up_test_1_conformal = pred_up_test_1 + np.quantile(f_conformity_score_high(pred_up_cal,y_cal),1-alpha_star_high)
 
    
    alpha_star_low_upd = alpha_star_low_updated + gamma*(alphas[0] - np.mean([err_t_low(y_test_1,pred_down_test_1,i) for i in range(t)]))
    if 0<alpha_star_low_upd <1: 
      alpha_star_low_updated = alpha_star_low_upd
    else :
      print("alpha_star_low_updated out of 0-1")

    alpha_star_high_upd = alpha_star_high_updated +gamma*(1-alphas[1] - np.mean([err_t_high(y_test_1,pred_up_test_1,i) for i in range(t)]))

    if 0<alpha_star_high_upd <1: 
      alpha_star_high_updated = alpha_star_high_upd
    else :
      print("alpha_star_high_updated out of 0-1")



    pred_down_test_2_adaptative_conformal_asymetric = pred_down_test_2 - np.quantile(f_conformity_score_low(pred_down_test_1,y_test_1),1-alpha_star_low_updated)
    pred_up_test_2_adaptative_conformal_asymetric = pred_up_test_2 + np.quantile(f_conformity_score_high(pred_up_test_1,y_test_1),1-alpha_star_high_updated)

    error.append((np.sum(y_test_2<pred_down_test_2_adaptative_conformal_asymetric) + np.sum(y_test_2>pred_up_test_2_adaptative_conformal_asymetric))/len(y_test_2))

    if step == 0:
      fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                          mode='lines',
                          name='True'))
    fig.add_trace(go.Scatter(x=xX_test_1, y=pred_down_test_1_conformal,
                          mode='lines',
                          name=f'{model_name}_Conformalised_{alphas[0]}'))
    fig.add_trace(go.Scatter(x=xX_test_1, y=pred_up_test_1_conformal,
                          mode='lines',
                          name=f'{model_name}_Conformalised_{alphas[1]}'))

    fig.add_trace(go.Scatter(x=xX_test_2, y=pred_down_test_2_adaptative_conformal_asymetric,
                          mode='lines',
                          name=f'{model_name}_Adaptative_asymetric_Conformalised_{alphas[0]}'))
    fig.add_trace(go.Scatter(x=xX_test_2, y=pred_up_test_2_adaptative_conformal_asymetric,
                          mode='lines',
                          name=f'{model_name}_Adaptative_asymetric_Conformalised_{alphas[1]}'))

    #fig.add_vline(x=list(xX_test_1)[-1], line_width=3, line_dash="dash", line_color="black")


# Make 10th trace visible
#fig.data[0].visible = True


# Create and add slider
steps = []
for i in range(int(len(fig.data)/5)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": f"Conformal asymetric {model_name} IC predict, confidence_test_2 = {1 - error[i]}, expected = {confidence}"}],  # layout attribute
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

# MAGIC %md
# MAGIC ## Online Expert Aggregation on ACI (AgACI)

# COMMAND ----------


def AgACI(gammas : list):
  #Initialisation 
  
  for t in y_test_2:
    low_t = []
    high_t = []
    omega_t_low = []
    omega_t_high = []
    low = pred_down_test_2 - np.quantile(f_conformity_score_low(pred_down_test_1,y_test_1),1-alpha_star_low_updated)
    high = pred_up_test_2 + np.quantile(f_conformity_score_high(pred_up_test_1,y_test_1),1-alpha_star_high_updated)
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

      low_t.append(low)
      high_t.append(high)
    C_low = np.mean(low_t,axis = 0)
    C_high = np.mean(high_t,axis = 0)
  return C_low,C_high

# COMMAND ----------

if 1:
  import plotly.graph_objects as go
  import numpy as np
  from sklearn.model_selection import train_test_split
  import math
  gammas = [0.1,0.05,0.01,0.005,0.001]
  alpha_star_low = np.max([b for b in np.arange(0,1,0.0001) if (miscoverage_rate_low(pred_down_val,conformity_score_low,alpha = b) < alphas[0])])
  alpha_star_high = np.max([b for b in np.arange(0,1,0.0001) if (miscoverage_rate_high(pred_up_val,conformity_score_high,alpha = b) < 1 - alphas[1])])
  alpha_star_low_updated = alpha_star_low
  alpha_star_high_updated = alpha_star_high

  X_test_full = X_test
  X_test_full["DT_VALR"] = xX_test

  # Create figure
  fig = go.Figure()
  # Add traces, one for each slider step

  for step in range(int(len(X_test)/2 -1)):
      
      
      X_test_1, X_test_2, y_test_1, y_test_2 = train_test_split(X_test_full,y_test,test_size = int(len(X_test)/2) - step,shuffle = False)
      xX_test_1 = X_test_1["DT_VALR"]
      X_test_1 = X_test_1.drop(["DT_VALR"],axis = 1)
      xX_test_2 = X_test_2["DT_VALR"]
      X_test_2 = X_test_2.drop(["DT_VALR"],axis = 1)

      pred_down_test_1 = model_down.predict(X_test_1)
      pred_up_test_1 = model_up.predict(X_test_1)

      pred_down_test_2 = model_down.predict(X_test_2)
      pred_up_test_2 = model_up.predict(X_test_2)


      
      C_low,C_high = AgACI(gammas)
      if step%10 == 0:
        print(step, "/" , len(X_test)/2)
      error.append((np.sum(y_test_2<C_low) + np.sum(y_test_2>C_high))/len(y_test_2))
      if step == 0:
        fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                            mode='lines',
                            name='True'))
      fig.add_trace(go.Scatter(x=xX_test_1, y=pred_down_test_1_conformal,
                            mode='lines',
                            name=f'{model_name}_Conformalised_{alphas[0]}'))
      fig.add_trace(go.Scatter(x=xX_test_1, y=pred_up_test_1_conformal,
                            mode='lines',
                            name=f'{model_name}_Conformalised_{alphas[1]}'))

      fig.add_trace(go.Scatter(x=xX_test_2, y=C_low,
                            mode='lines',
                            name=f'{model_name}_Adaptative_asymetric_Conformalised_{alphas[0]}'))
      fig.add_trace(go.Scatter(x=xX_test_2, y=C_high,
                            mode='lines',
                            name=f'{model_name}_Adaptative_asymetric_Conformalised_{alphas[1]}'))

      #fig.add_vline(x=list(xX_test_1)[-1], line_width=3, line_dash="dash", line_color="black")


  # Make 10th trace visible
  #fig.data[0].visible = True


  # Create and add slider
  steps = []
  for i in range(int(len(fig.data)/5)):
      step = dict(
          method="update",
          args=[{"visible": [False] * len(fig.data)},
                {"title": f"Conformal asymetric adaptative agregated {model_name} IC predict, confidence_test_2 = {1 - error[i]}, expected = {confidence}"}],  # layout attribute
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

# MAGIC %md
# MAGIC ### Fully Adaptative Conformal Inference (FACI)
# MAGIC

# COMMAND ----------

def FACI_up(pvalues : list,gammas_k : list,alphas_i : list ,sigma, eta,alpha_expected):
  k = len(pvalues)
  omegas = np.ones(len(gammas_k))
  output = []
  for t in range(k):
    p = omegas / np.sum(omegas)
    alpha_barre = np.dot(p,alphas_i)
    output.append(alpha_barre)
    omegas_barre = []
    for i in range(len(alphas_i)):
      omegas_barre.append(omegas[i]*np.exp(-eta*(alpha_expected*(pvalues[t]-alphas_i[i]) - min(0,pvalues[t]-alphas_i[i]))))
    omegas = (np.multiply((1-sigma),omegas_barre) + np.sum(omegas_barre)*sigma/k)
    err_t_i = [(pred_up_test_1[t] + np.quantile(f_conformity_score_high(pred_up_test_1,y_test_1),1-alpha_i))>list(y_test_1)[t] for alpha_i in alphas_i]
    for i in range(len(alphas_i)):
      alphas_i[i] = max(0,min(1,alphas_i[i] + gammas_k[i]*(alpha_expected - err_t_i[i])))
  return output

# COMMAND ----------

p_values = [np.max([b for b in np.arange(0,1,0.01) if (pred_up_test_1[t] + np.quantile(f_conformity_score_high(pred_up_cal,y_cal),1-b))>list(y_test_1)[t] ]) for t in range(len(pred_up_test_1))]

# COMMAND ----------

FACI_up(pvalues = p_values,gammas_k =[0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128],alphas_i =[0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999],sigma = 0.9, eta = 0.5,alpha_expected = 0.95)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Bernstein Online Agregation

# COMMAND ----------

def boa(X,y,loss,eta,functions_set,alpha): # not terminated, t-1 to do
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

# COMMAND ----------

def faboa(X,y,loss,eta,functions_set,alpha): # not terminated, t-1 to do
  import math
  y = list(y)
  J = len(functions_set)
  #loss(beta,theta) = alpha*(beta-theta) - np.min(0,beta-theta)
  pi_t_j = [np.ones(len(functions_set))/len(functions_set)]
  L_t_j = 0
  n_t_j = 0
  E_t_j = 2
  for t in range(len(X)):
    epi = np.dot(pi_t_j[-1],[pinball_loss(y[t],functions_set[i][t],alpha = alpha) for i in range(J)])
    l_t_j.append([pinball_loss(y[t],functions_set[i][t],alpha = alpha) - epi for i in range(J)])
    L_t_j = [L_t_j[-1][j] + l_j_t[-1][j](1 + n_t_j[-1][j]*l_t_j[-1][j])/2 + E_t_j[-1][j](n_t_j[-1][j]*l_t_j[-1][j]>0.5) for j in range(J)]

    n_t_j.append([math.min(1/E_t_j[-1][j], math.sqrt(math.log(1/pi_t_j[0][j])/np.sum([l_t_j[s][j] for s in range(t)]))) for j in range(J)])
    
    regularisation = np.sum([np.exp(-eta*l_t_j[-1][j]*(1 + eta*l_t_j[-1][j]))*pi_t_j[-1][j] for j in range(J)])
    pi_t_j.append([np.exp(-n_t_j[-1][j]*l_t_j[-1][j]*(1 + n_t_j[-1][j]*l_t_j[-1][j]))*pi_t_j[-1][j] / regularisation for j in range(J)])

  return pi_t_j  

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/IC_functions

# COMMAND ----------

ci = conformal_inference(df_train = dataframe_dectrain, alphas = [0.05,0.95])


# COMMAND ----------

print(ci.X_train.shape)
print(ci.X_cal.shape)
print(ci.X_val.shape)
print(ci.y_val.shape)
print(ci.pred_down_val.shape)

# COMMAND ----------

dataframe_decpredict = dataframe_decpredict.fillna(method = 'ffill')
dataframe_decpredict

# COMMAND ----------

conformal_ic = ci.conformal_IC(dataframe_decpredict)

# COMMAND ----------

asymetric_conformal_ic = ci.asymetric_conformal_IC(dataframe_decpredict)