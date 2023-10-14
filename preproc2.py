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
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

groupes = ["DecPDV","EncPDV",  "DecUP", "EncUP",  "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]

# COMMAND ----------

dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
df_train_set,df_predict_set = dataloader.load_train_predict_set(groupes = groupes)

# COMMAND ----------

for groupe,df_train in zip(groupes,df_train_set):
  preproc = Preprocessing(groupe = groupe)
  df_train = preproc.preproc(df_train)
  print(groupe)
  y = df_train["Valeur"]
  X = df_train.drop(["Valeur"],axis = 1)
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, shuffle = False)
  index_train = X_train["DT_VALR"]
  index_test = X_test["DT_VALR"]
  X_train = X_train.drop(["DT_VALR"],axis = 1) 
  X_test = X_test.drop(["DT_VALR"],axis = 1) 
  forest = RandomForestRegressor()
  forest.fit(X_train,y_train)

  start_time = time.time()
  importances = forest.feature_importances_
  std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
  elapsed_time = time.time() - start_time

  print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
  forest_importances = pd.Series(importances, index=X_train.columns)

  fig, ax = plt.subplots()
  forest_importances.plot.bar(yerr=std, ax=ax)
  ax.set_title(f"{groupe} Feature importances")
  ax.set_ylabel("Mean decrease in impurity")
  fig.tight_layout()

# COMMAND ----------

alphas = [0.05,0.95]

# COMMAND ----------

param_grid_lgbm = {'n_estimators': [100,500],
                                    'learning_rate' : [0.01,0.1]
                        }

param_grid_gb = {'n_estimators': [100,500],
                                      'learning_rate' : [0.01,0.1],
                                      'max_depth' : [3,5,10]
                          }     

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


  print("Qrf")
  qrf = RandomForestQuantileRegressor(n_estimators= 1000)
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
  fig.update_layout(title = f"Test : {1-error_qrf[-1]}% qrf {groupe}, efficience = {efficience_qrf[-1]}")                 
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

# COMMAND ----------

groupes = ["DecPDV","EncPDV",  "DecUP", "EncUP",  "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]
for groupe,df_train in zip(groupes,df_train_set) :

  param_grid_qrf = {'n_estimators': [100,500,1000],
                    'min_samples_leaf' : [32,64,128,256],
                    'bootstrap'}    



  X = df_train.drop(["Valeur"],axis = 1)
  y = df_train["Valeur"]
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, shuffle = False)
  index_train = X_train["DT_VALR"]
  index_test = X_test["DT_VALR"]
  X_train = X_train.drop(["DT_VALR"],axis = 1) 
  X_test = X_test.drop(["DT_VALR"],axis = 1) 
  qrf = RandomForestQuantileRegressor(n_estimators= 10,bootstrap = False)
  qrf.fit(X_train, y_train)
  pred_qrf_up = qrf.predict(X_test, quantiles=0.95)
  pred_qrf_down = qrf.predict(X_test, quantiles=0.05)

  mqloss_scorer_up = make_scorer(mqloss, alpha=alphas[1])
  mqloss_scorer_down = make_scorer(mqloss, alpha=alphas[0])
  mqloss_scorer_median = make_scorer(mqloss, alpha=0.5)

  qrf_up = GridSearchCV(
                          estimator=RandomForestQuantileRegressor(quantiles=0.95),
                          param_grid=param_grid_qrf,
                          scoring=mqloss_scorer_up,
                          cv=3,
                          n_jobs=-1,
                          verbose=1
    )
  pred_qrf_up = np.array(qrf_up.fit(X_train, y_train).best_estimator_.predict(X_test))
  print("qrf_up.best_params = ", qrf_up.best_params_)
  
  qrf_down = GridSearchCV(
                          estimator=RandomForestQuantileRegressor(),
                          param_grid=param_grid_qrf,
                          scoring=mqloss_scorer_down,
                          cv=3,
                          n_jobs=-1,
                          verbose=1
    )
  pred_qrf_down = np.array(qrf_up.fit(X_train, y_train).best_estimator_.predict(X_test))
  print("qrf_down.best_params = ", qrf_down.best_params_)
    
  error_qrf = (np.sum(np.array(y_test)<pred_qrf_down) + np.sum(np.array(y_test)>pred_qrf_up))/len(np.array(y_test))
  efficience_qrf = integral_score(pred_qrf_up,pred_gb_down)


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
  fig.update_layout(title = f"Test : {1-((np.sum(np.array(y_test)<pred_qrf_down) + np.sum(np.array(y_test)>pred_qrf_up))/len(np.array(y_test)))}% qrf {groupe}, efficience = {integral_score(pred_qrf_up,pred_qrf_down)}")                 
  fig.show()

# COMMAND ----------



# COMMAND ----------

groupes = ["DecPDV","EncPDV",  "DecUP", "EncUP",  "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]
dico = {}
n_estimators = [32,64,128,256]
max_depths = [3,5,8,13,21]
for groupe,df_train in zip(groupes,df_train_set) :
  print(groupe)
  dico[groupe] = {}
  dico[groupe]["efficiency"] = pd.DataFrame(index = n_estimators, columns = max_depths)
  dico[groupe]["loss_up"] = pd.DataFrame(index = n_estimators, columns = max_depths)
  dico[groupe]["loss_down"] = pd.DataFrame(index = n_estimators, columns = max_depths)

  for n_estimator in n_estimators:
    for max_depth in max_depths:
      print("n_estimator= ",n_estimator)
      print("max_depth = ",max_depth)
      start_time = time.time()
      qrf = RandomForestQuantileRegressor(n_estimators= n_estimator,bootstrap = True,min_samples_leaf = int(math.log(len(X_train))),max_depth = max_depth)
      qrf.fit(X_train, y_train)
      pred_qrf_up = qrf.predict(X_test, quantiles=0.95)
      pred_qrf_down = qrf.predict(X_test, quantiles=0.05)
      end_time = time.time()
      dico[groupe]["efficiency"].loc[n_estimator,max_depth] = integral_score(pred_qrf_up,pred_qrf_down)
      dico[groupe]["loss_up"].loc[n_estimator,max_depth] = mqloss(y_test,pred_qrf_up,0.95)
      dico[groupe]["loss_down"].loc[n_estimator,max_depth] = mqloss(y_test,pred_qrf_up,0.05)

      
      print("time elapsed = ",end_time - start_time)
      print("mqloss_up = ",mqloss(y_test,pred_qrf_up,0.95))
      print("mqloss_down = ",mqloss(y_test,pred_qrf_down,0.05))
      print("efficience = ",integral_score(pred_qrf_up,pred_qrf_down))
      print("")