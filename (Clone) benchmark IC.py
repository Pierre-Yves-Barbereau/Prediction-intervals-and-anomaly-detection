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
alphas = [0.05,0.95]
from sklearn.preprocessing import MinMaxScaler

param_grid_lgbm = {'n_estimators': [100,200,500,1000],
                                    'learning_rate' : [0.01,0.05,0.1]
                        }

param_grid_gb = {'n_estimators': [100,200,500,1000],
                                      'learning_rate' : [0.01,0.05,0.1],
                                      'max_depth' : [3,5,10,50]
                          }   

mqloss_scorer_up = make_scorer(mqloss, alpha=alphas[1])
mqloss_scorer_down = make_scorer(mqloss, alpha=alphas[0])
mqloss_scorer_median = make_scorer(mqloss, alpha=0.5)
dico = {}

for groupe,df_train in zip(groupes,df_train_set) :
  dico[groupe] = {}
  dico[groupe]["GB"] = {}
  dico[groupe]["LGBM"] = {}
  dico[groupe]["LGBM"]["best_model"] = {}
  dico[groupe]["GB"]["best_model"] = {}

  X = df_train.drop(["Valeur"],axis = 1)
  y = df_train["Valeur"]
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, shuffle = False)
  index_train = X_train["DT_VALR"]
  index_test = X_test["DT_VALR"]
  X_train = X_train.drop(["DT_VALR"],axis = 1) 
  X_test = X_test.drop(["DT_VALR"],axis = 1) 

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
  dico[groupe]["GB"]["best_model"]["down"] = GB_down.best_params_
  
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
  dico[groupe]["GB"]["best_model"]["up"] = GB_up.best_params_
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
  dico[groupe]["LGBM"]["best_model"]["down"] = LGBM_down.best_params_

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
  dico[groupe]["LGBM"]["best_model"]["up"] = LGBM_up.best_params_
  
  error_lgbm.append((np.sum(np.array(y_test)<pred_lgbm_down) + np.sum(np.array(y_test)>pred_lgbm_up))/len(np.array(y_test)))
  efficience_gb.append(integral_score(pred_gb_up,pred_gb_down))
  efficience_lgbm.append(integral_score(pred_lgbm_up,pred_lgbm_down))

  
  dico[groupe]["GB"]["efficience"] =  integral_score(pred_gb_up,pred_gb_down)
  dico[groupe]["GB"]["mqloss_up"] = mqloss(y_test,pred_gb_up,0.95)
  dico[groupe]["GB"]["mqloss_down"] = mqloss(y_test,pred_gb_down,0.05)
  dico[groupe]["LGBM"]["efficience"] =  integral_score(pred_lgbm_up,pred_lgbm_down)
  dico[groupe]["LGBM"]["mqloss_up"] = mqloss(y_test,pred_lgbm_up,0.95)
  dico[groupe]["LGBM"]["mqloss_down"] = mqloss(y_test,pred_lgbm_down,0.05)
  dico[groupe]["LGBM"]["mqloss_down"] = mqloss(y_test,pred_lgbm_down,0.05)

  print("efficience_gb = ",integral_score(pred_gb_up,pred_gb_down))
  print("error_up_gb = ",mqloss(y_test,pred_gb_up,0.95))
  print("error_down_gb = ",mqloss(y_test,pred_gb_down,0.05))

  print("efficience_lgbm = ",integral_score(pred_lgbm_up,pred_lgbm_down))
  print("error_up_lgbm = ",mqloss(y_test,pred_lgbm_up,0.95))
  print("error_down_lgbm = ",mqloss(y_test,pred_lgbm_down,0.05))

np.save('/dbfs/FileStore/dico_boosting.npy',dico)
print(dico)

# COMMAND ----------



# COMMAND ----------

for groupe,df_train in zip(groupes,df_train_set) :
  GB_up = GradientBoostingRegressor(loss="quantile", alpha=alphas[1],**dico[groupe]["GB"]["best_model"]["up"])
  GB_down = GradientBoostingRegressor(loss="quantile", alpha=alphas[0],**dico[groupe]["GB"]["best_model"]["down"])
  LGBM_up = LGBMRegressor(alpha=alphas[1],  objective='quantile',**dico[groupe]["LGBM"]["best_model"]["up"])
  LGBM_up = LGBMRegressor(alpha=alphas[0],  objective='quantile',**dico[groupe]["LGBM"]["best_model"]["down"])
  qrf = 

# COMMAND ----------

dico

# COMMAND ----------



# COMMAND ----------

