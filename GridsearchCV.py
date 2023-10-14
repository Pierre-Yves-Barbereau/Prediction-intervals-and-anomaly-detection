# Databricks notebook source
# MAGIC %run /Tools/library/CFM

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/preprocessing

# COMMAND ----------

lib_instance = Cfm()

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn_quantile import RandomForestQuantileRegressor

# COMMAND ----------

#Variables widget
groupe = lib_instance.define_widget("groupe") 

#val_size = float(lib_instance.define_widget("val_size")) #365

#PAth notebooks
path_notebook_preproc_preprocessing = lib_instance.define_widget("path_notebook_preproc_preprocessing") #'/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing'

# COMMAND ----------

#dbutils.widgets.removeAll()

# COMMAND ----------

dataloader = Dataloader(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)
df_train, df_predict = dataloader.load_train_predict(groupe = groupe)

# COMMAND ----------

y_train = df_train["Valeur"]
X_train = df_train.drop(["DT_VALR","Valeur"],axis = 1)

# COMMAND ----------

qrf005 = RandomForestQuantileRegressor(q=0.05, n_estimators=100, criterion='absolute_error', max_depth=None, verbose=1)
params = {
    'n_estimators': [10,50,100,200,1000],

}

Grid_search_qrf005 = GridSearchCV(
    estimator=qrf005,
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
)
Grid_search_qrf005.fit(X_train, y_train)
Grid_search_qrf005.best_params_

# COMMAND ----------

qrf095 = RandomForestQuantileRegressor(q=0.95, n_estimators=100, criterion='mae', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, n_jobs=1, random_state=None, verbose=0)
params = {
    'n_estimators': [10,50,100,200,1000],

}

Grid_search_qrf095 = GridSearchCV(
    estimator=qrf095,
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
)
Grid_search_qrf095.fit(X_train, y_train)
Grid_search_qrf095.best_params_

# COMMAND ----------

qrf050 = RandomForestQuantileRegressor(q=0.5, n_estimators=100, criterion='mae', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, n_jobs=1, random_state=None, verbose=0)
params = {
    'n_estimators': [10,50,100,200,1000],

}

Grid_search_qrf050 = GridSearchCV(
    estimator=qrf050,
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
)

Grid_search_qrf050.fit(X_train, y_train)
Grid_search_qrf050.best_params_

# COMMAND ----------


GradientBoosting_005 = GradientBoostingRegressor(loss="quantile", alpha=0.05,n_estimators = 1000,
                                                      learning_rate = 0.01)
params = {
    'n_estimators': [1500,2000,3000,6000],
    'learning_rate': [0.15,0.1,0.09,0.08],
    'max_depth': [9,10,11,12,13]
}

Grid_search_GB_005 = GridSearchCV(
    estimator=GradientBoosting_005,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)

#Runs précédents
"""
params = {
    'n_estimators': [800,1000,1200,1500,2000],
    'learning_rate': [0.1,0.05,0.01,0.005,0.001],
    'max_depth': [2,3,4,5,6]   }  ==>  best_params_ = {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 1200}

params = {
    'n_estimators': [800,1000,1200,1500,2000],
    'learning_rate': [2,1,0.5,0.1,0.05],
    'max_depth': [5,6,7,8,9]
} =>     {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 2000} 
"""


# COMMAND ----------

Grid_search_GB_005.fit(X_train, y_train)
Grid_search_GB_005.best_params_


# COMMAND ----------


GradientBoosting_095 = GradientBoostingRegressor(loss="quantile", alpha=0.95)
params = {
    'n_estimators': [600,800,1000],
    'learning_rate': [0.2,0.1,0.05],
    'max_depth': [8,9,10,11,12]
}

Grid_search_GB_095 = GridSearchCV(
    estimator=GradientBoosting_095,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)
#Runs précédents
"""
params = {
    'n_estimators': [800,1000,1200,1500,2000],
    'learning_rate': [0.1,0.05,0.01,0.005,0.001],
    'max_depth': [5,6,7,8,9]
}

==>  best_params_ = {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 800}


params = {
    'n_estimators': [600,800,1000],
    'learning_rate': [0.2,0.1,0.05],
    'max_depth': [8,9,10,11,12]
}  ==> {'learning_rate': 0.05, 'max_depth': 12, 'n_estimators': 1000}


"""

# COMMAND ----------

Grid_search_GB_095.fit(X_train, y_train)
Grid_search_GB_095.best_params_

# COMMAND ----------


GradientBoosting_median = GradientBoostingRegressor(loss="quantile", alpha=0.5)
params = {
    'n_estimators': [800,1000,1200,1500,2000],
    'learning_rate': [0.1,0.05,0.01,0.005,0.001],
    'max_depth': [5,6,7,8,9]
}

Grid_search_GB_median = GridSearchCV(
    estimator=GradientBoosting_median,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)


#Runs précédents
"""
params = {
    'n_estimators': [800,1000,1200,1500,2000],
    'learning_rate': [0.1,0.05,0.01,0.005,0.001],
    'max_depth': [5,6,7,8,9]
}
#Grid_search_GB_median.best_params_ = {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 800}
"""



# COMMAND ----------

Grid_search_GB_median.fit(X_train, y_train)
Grid_search_GB_median.best_params_

# COMMAND ----------



LGBM_median = LGBMRegressor(loss="quantile", alpha=0.5)
params = {
    'boosting_type' : ['gbdt','dart','rf'],
    'n_estimators': [800,1000,1200,1500,2000],
    'learning_rate': [0.2,0.1,0.05,0.01],
    'num_leaves': [30,40,50]
}

Grid_search_LGBM_median = GridSearchCV(
    estimator=LGBM_median,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)

"""
params = {
    'boosting_type' : ['gbdt','dart','rf'],
    'n_estimators': [800,1000,1200,1500,2000],
    'learning_rate': [0.2,0.1,0.05,0.01],
    'num_leaves': [30,40,50]
} ==>  {'boosting_type': 'dart',
 'learning_rate': 0.01,
 'n_estimators': 1500,
 'num_leaves': 30}
 """

# COMMAND ----------

Grid_search_LGBM_median.fit(X_train, y_train)
Grid_search_LGBM_median.best_params_

# COMMAND ----------



LGBM_005 = LGBMRegressor(loss="quantile", alpha=0.05)
params = {
    'boosting_type' : ['gbdt','dart','rf'],
    'n_estimators': [800,1000,1200,1500,2000],
    'learning_rate': [0.2,0.1,0.05,0.01],
    'num_leaves': [30,40,50]
}

Grid_search_LGBM_005 = GridSearchCV(
    estimator=LGBM_005,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)

"""
LGBM_median = LGBMRegressor(loss="quantile", alpha=0.5)
params = {
    'boosting_type' : ['gbdt','dart','rf'],
    'n_estimators': [800,1000,1200,1500,2000],
    'learning_rate': [0.2,0.1,0.05,0.01],
    'num_leaves': [30,40,50]
} ==>  {'boosting_type': 'dart',
 'learning_rate': 0.01,
 'n_estimators': 1500,
 'num_leaves': 30}
 """

# COMMAND ----------

Grid_search_LGBM_005.fit(X_train, y_train)
Grid_search_LGBM_005.best_params_

# COMMAND ----------

LGBM_095 = LGBMRegressor(loss="quantile", alpha=0.95)
params = {
    'boosting_type' : ['rf','dart',"gbdt"],
    'n_estimators': [800,1000,1200,1500,2000],
    'learning_rate': [0.2,0.1,0.05,0.01],
    'num_leaves': [30,40,50]
}

Grid_search_LGBM_095 = GridSearchCV(
    estimator=LGBM_095,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)

"""
{
    'boosting_type' : ['gbdt','dart','rf'],
    'n_estimators': [800,1000,1200,1500,2000],
    'learning_rate': [0.2,0.1,0.05,0.01],
    'num_leaves': [30,40,50]
}
=>
{'boosting_type': 'dart',
 'learning_rate': 0.01,
 'n_estimators': 1500,
 'num_leaves': 30}


 
 """

# COMMAND ----------

Grid_search_LGBM_095.fit(X_train, y_train)
Grid_search_LGBM_095.best_params_