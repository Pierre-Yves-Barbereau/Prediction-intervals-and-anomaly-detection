# Databricks notebook source
pip install quantile-forest

# COMMAND ----------

# MAGIC %run /Tools/library/CFM

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/IC_functions

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/preprocessing

# COMMAND ----------

lib_instance = Cfm()

# COMMAND ----------

#import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from quantile_forest import RandomForestQuantileRegressor
import math
import copy
from datetime import timedelta
from jours_feries_france import JoursFeries
import warnings
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from itertools import zip_longest
warnings.filterwarnings("ignore")

# COMMAND ----------

groupes = ["DecPDV","EncPDV",  "DecUP", "EncUP",  "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]

# COMMAND ----------

path_notebook_preproc_preprocessing = "/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing"

# COMMAND ----------

dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
df_train_set,df_predict_set = dataloader.load_train_predict_set(groupes = groupes)

# COMMAND ----------

dico_boosting = np.load('/dbfs/FileStore/dico_boosting.npy',allow_pickle = True).item()
dico_qrf = np.load("/dbfs/FileStore/dico_qrf.npy",allow_pickle = True).item()

# COMMAND ----------

dico_qrf

# COMMAND ----------

alphas = [0.05,0.95]
for groupe,df_train in zip(groupes,df_train_set) :
  print(groupe)
  preproc = Preprocessing(groupe = groupe)
  df_train = preproc.preproc(df_train)
  X = df_train.drop(["Valeur"],axis = 1)
  y = df_train["Valeur"]
  X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, shuffle = False)
  index_train = X_train["DT_VALR"]
  index_test = X_test["DT_VALR"]
  X_train = X_train.drop(["DT_VALR"],axis = 1)
  X_test = X_test.drop(["DT_VALR"],axis = 1)

  GB_up = GradientBoostingRegressor(loss="quantile", alpha=alphas[1],**dico_boosting[groupe]["GB"]["best_model"]["up"])
  GB_down = GradientBoostingRegressor(loss="quantile", alpha=alphas[0],**dico_boosting[groupe]["GB"]["best_model"]["down"])
  LGBM_up = LGBMRegressor(alpha=alphas[1],  objective='quantile', **dico_boosting[groupe]["LGBM"]["best_model"]["up"])
  LGBM_down = LGBMRegressor(alpha=alphas[0],  objective='quantile', **dico_boosting[groupe]["LGBM"]["best_model"]["down"])
  qrf = RandomForestQuantileRegressor(n_estimators= 64, bootstrap = True, min_samples_leaf = int(math.log(len(X_train))), max_depth = 5)

  print("fit gb up")
  GB_up.fit(X_train,y_train)
  print("fit gb down")
  GB_down.fit(X_train,y_train)
  print("fit lgbm up")
  LGBM_up.fit(X_train,y_train)
  print("fit lgbm down")
  LGBM_down.fit(X_train,y_train)
  print("fit qrf")
  qrf.fit(X_train,y_train)

  pred_gb_up = GB_up.predict(X_test)
  pred_gb_down = GB_down.predict(X_test)
  pred_lgbm_up = LGBM_up.predict(X_test)
  pred_lgbm_down = LGBM_down.predict(X_test)
  pred_qrf_up = qrf.predict(X_test,quantiles = alphas[1])
  pred_qrf_down = qrf.predict(X_test,quantiles = alphas[0])

  fig = go.Figure()
  fig.add_trace(go.Scatter( y=pred_gb_up,
                      mode='lines',
                      name=f'q_{alphas[1]}',
                      line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                      showlegend = False))

  fig.add_trace(go.Scatter( y=pred_gb_down,
                      mode='lines',
                      name=f'{int(np.round(100*(alphas[1]-alphas[0])))}% Confidence Interval',
                      line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                      fill='tonexty',
                      fillcolor='rgba(0,176,246,0.2)',
                      line_color='rgba(255,255,255,0)'))
      
  fig.update_layout(title = f"{groupe} GB confidence {(np.sum(np.array(y_test)<pred_gb_down) + np.sum(np.array(y_test)>pred_gb_up))/len(np.array(y_test))}% efficience : {integral_score(pred_gb_up,pred_gb_down)}")
  fig.show()
  
  fig = go.Figure()
  fig.add_trace(go.Scatter( y=pred_lgbm_up,
                    mode='lines',
                    name=f'q_{alphas[1]}',
                    line=dict(
                        color='rgb(0, 256, 0)',
                        width=0),
                    showlegend = False))

  fig.add_trace(go.Scatter( y=pred_lgbm_down,
                    mode='lines',
                    name=f'{int(np.round(100*(alphas[1]-alphas[0])))}% Confidence Interval',
                    line=dict(
                        color='rgb(0, 256, 0)',
                        width=0),
                    fill='tonexty',
                    fillcolor='rgba(0,176,246,0.2)',
                    line_color='rgba(255,255,255,0)'))
    
  fig.update_layout(title = f"{groupe} LGBM confidence {(np.sum(np.array(y_test)<pred_lgbm_down) + np.sum(np.array(y_test)>pred_lgbm_up))/len(np.array(y_test))}% efficience : {integral_score(pred_lgbm_up,pred_lgbm_down)}")
  fig.show()

  fig = go.Figure()
  fig.add_trace(go.Scatter( y=pred_lgbm_up,
                      mode='lines',
                      name=f'q_{alphas[1]}',
                      line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                      showlegend = False))

  fig.add_trace(go.Scatter( y=pred_lgbm_down,
                      mode='lines',
                      name=f'{int(np.round(100*(alphas[1]-alphas[0])))}% Confidence Interval',
                      line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                      fill='tonexty',
                      fillcolor='rgba(0,176,246,0.2)',
                      line_color='rgba(255,255,255,0)'))
      
  fig.update_layout(title = f"{groupe} QRF confidence {(np.sum(np.array(y_test)<pred_qrf_down) + np.sum(np.array(y_test)>pred_qrf_up))/len(np.array(y_test))}% efficience : {integral_score(pred_qrf_up,pred_qrf_down)}")
  fig.show()

# COMMAND ----------

#test by models 

mqloss_scorer_up = make_scorer(mqloss, alpha=alphas[1])
mqloss_scorer_down = make_scorer(mqloss, alpha=alphas[0])
mqloss_scorer_median = make_scorer(mqloss, alpha=0.5)

for groupe,df_train in zip(groupes,df_train_set) :
  X = df_train.drop(["Valeur"],axis = 1)
  y = df_train["Valeur"]
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, shuffle = False)
  index_train = X_train["DT_VALR"]
  index_test = X_test["DT_VALR"]
  X_train = X_train.drop(["DT_VALR"],axis = 1) 
  X_test = X_test.drop(["DT_VALR"],axis = 1) 

  efficience_gb = []
  efficience_lgbm = []
  error_gb = []
  efficience_qrf = []
  error_qrf = []
  error_lgbm = []

  GB_down = GridSearchCV(
                          estimator=GradientBoostingRegressor(loss="quantile", alpha=alphas[0]),
                          param_grid=param_grid_gb,
                          scoring=mqloss_scorer_down,
                          cv=3,
                          n_jobs=-1,
                          verbose=1
    )
  pred_gb_down = np.array(GB_down.fit(X_train, y_train).best_estimator_.predict(X_test))
  print("GB_down.best_params = ", GB_down.best_params_)

  print("GridSearchCV Gradient_boosting up")
  GB_up = GridSearchCV(
                        estimator=GradientBoostingRegressor(loss="quantile", alpha=alphas[1]),
                        param_grid=param_grid_gb,
                        scoring=mqloss_scorer_up,
                        cv=3,
                        n_jobs=-1,
                        verbose=1
  )
  pred_gb_up = np.array(GB_up.fit(X_train, y_train).best_estimator_.predict(X_test))
  print("GB_up.best_params = ", GB_up.best_params_)

  error_gb.append((np.sum(np.array(y_test)<pred_gb_down) + np.sum(np.array(y_test)>pred_gb_up))/len(np.array(y_test)))


  
  #LGBM gridsearch
  print("GridSearchCV LGBM down")
  LGBM_down = GridSearchCV(
                        estimator=LGBMRegressor(alpha=alphas[0],  objective='quantile'),
                        param_grid=param_grid_lgbm,
                        scoring=mqloss_scorer_down,
                        cv=3,
                        n_jobs=-1,
                        verbose=1
  )
  pred_lgbm_down = np.array(LGBM_down.fit(X_train, y_train).best_estimator_.predict(X_test))
  print("LGBM_down.best_params = ", LGBM_down.best_params_)

  print("GridSearchCV LGBM up")
  LGBM_up = GridSearchCV(
                        estimator=LGBMRegressor(alpha=alphas[1],  objective='quantile'),
                        param_grid=param_grid_lgbm,
                        scoring=mqloss_scorer_up,
                        cv=3,
                        n_jobs=-1,
                        verbose=1
  )
  pred_lgbm_up = np.array(LGBM_up.fit(X_train, y_train).best_estimator_.predict(X_test))
  print("LGBM_up.best_params = ", LGBM_up.best_params_)
  
  error_lgbm.append((np.sum(np.array(y_test)<pred_lgbm_down) + np.sum(np.array(y_test)>pred_lgbm_up))/len(np.array(y_test)))
  efficience_gb.append(integral_score(pred_gb_up,pred_gb_down))
  efficience_lgbm.append(integral_score(pred_lgbm_up,pred_lgbm_down))

  qrf = RandomForestQuantileRegressor()
  qrf.fit(X_train, y_train)
  pred_qrf_up = qrf.predict(X_test, quantiles=0.95)
  pred_qrf_down = qrf.predict(X_test, quantiles=0.05)
  efficience_qrf.append(integral_score(pred_qrf_up,pred_qrf_down))
  error_qrf.append((np.sum(np.array(y_test)<pred_qrf_down) + np.sum(np.array(y_test)>pred_qrf_up))/len(np.array(y_test)))

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=index_test, y=y_test,
                mode='lines',
                name=f'y_true',
                line=dict(
                    color='rgb(0,0, 256)',
                    width=1),
                showlegend = True))
  
  fig.add_trace(go.Scatter(x=index_test, y=pred_qrf_up,
                  mode='lines',
                  name=f'q_{alphas[1]}',
                  line=dict(
                      color='rgb(0, 256, 0)',
                      width=0),
                  showlegend = False))
  
  fig.add_trace(go.Scatter(x=index_test, y=pred_qrf_down,
                  mode='lines',
                  name=f'{int(np.round(100*(alphas[1]-alphas[0])))}% Confidence Interval',
                  line=dict(
                      color='rgb(0, 256, 0)',
                      width=0),
                  fill='tonexty',
                  fillcolor='rgba(0,176,246,0.2)',
                  line_color='rgba(255,255,255,0)'))
  fig.update_traces(mode='lines')
  fig.update_layout(title = f"Test : {1-error_qrf[-1]}% gradient boosting {groupe}, efficience = {efficience_qrf[-1]}")                 
  fig.show()
  

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=index_test, y=y_test,
                mode='lines',
                name=f'y_true',
                line=dict(
                    color='rgb(0,0, 256)',
                    width=1),
                showlegend = True))
  
  fig.add_trace(go.Scatter(x=index_test, y=pred_gb_up,
                  mode='lines',
                  name=f'q_{alphas[1]}',
                  line=dict(
                      color='rgb(0, 256, 0)',
                      width=0),
                  showlegend = False))
  
  fig.add_trace(go.Scatter(x=index_test, y=pred_gb_down,
                  mode='lines',
                  name=f'{int(np.round(100*(alphas[1]-alphas[0])))}% Confidence Interval',
                  line=dict(
                      color='rgb(0, 256, 0)',
                      width=0),
                  fill='tonexty',
                  fillcolor='rgba(0,176,246,0.2)',
                  line_color='rgba(255,255,255,0)'))
  fig.update_traces(mode='lines')
  fig.update_layout(title = f"Test : {1-error_gb[-1]}% gradient boosting {groupe}, efficience = {efficience_gb[-1]}")                 
  fig.show()

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=index_test, y=y_test,
                mode='lines',
                name=f'y_true',
                line=dict(
                    color='rgb(0,0, 256)',
                    width=1),
                showlegend = True))
          
  fig.add_trace(go.Scatter(x=index_test, y=pred_lgbm_up,
                  mode='lines',
                  name=f'q_{alphas[1]}',
                  line=dict(
                      color='rgb(0, 256, 0)',
                      width=0),
                  showlegend = False))

  fig.add_trace(go.Scatter(x=index_test, y=pred_lgbm_down,
                  mode='lines',
                  name=f'{int(np.round(100*(alphas[1]-alphas[0])))}% Confidence Interval',
                  line=dict(
                      color='rgb(0, 256, 0)',
                      width=0),
                  fill='tonexty',
                  fillcolor='rgba(0,176,246,0.2)',
                  line_color='rgba(255,255,255,0)'))
  fig.update_traces(mode='lines')
  fig.update_layout(title = f"Test : {1-error_lgbm[-1]}% LGBM {groupe}, efficience = {efficience_lgbm[-1]}")                      
  fig.show()