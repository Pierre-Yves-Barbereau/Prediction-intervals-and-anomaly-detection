# Databricks notebook source
# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/preprocessing

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/IC_functions

# COMMAND ----------

preproc = preprocessing("/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing")
df = preproc.load_train("DecPDV")

# COMMAND ----------

y = df["Valeur"]
X = df.drop(["DT_VALR","Valeur"],axis = 1)

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.5,shuffle = True)

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

# COMMAND ----------

import pickle
pickle_in = open('/dbfs/tmp/models/XGBRegressorDecPDV.sav', 'rb')

Xgb_model = pickle.load(pickle_in)

Gb_model = GradientBoostingRegressor(loss="squared_error",n_estimators = 1000,
                                                      learning_rate = 0.01)
Lgbm_model = LGBMRegressor(objective='mse')


# COMMAND ----------

models = [Xgb_model,Gb_model,Lgbm_model]

# COMMAND ----------

for model in models:
  model.fit(X_train,y_train)

# COMMAND ----------

X_test_1,X_test_2,y_test_1,y_test_2 = train_test_split(X_test,y_test,test_size = 0.5)

# COMMAND ----------

predict_set = []
for model in models :
  predict_set.append(model.predict(X_test_1))

# COMMAND ----------

ag = aggregation_IC(predict_set,y_test_1)

# COMMAND ----------

ag.faboa(0.5)

# COMMAND ----------

def aggreg(pred_set,pi_t_j):
  pred = []
  for t in range(len(pi_t_j)):
    pred.append(np.dot([pi_t_j[t][i] for i in range(len(pred_set))],[pred_set[i][t] for i in range(len(pred_set))]))
  return pred

# COMMAND ----------

pred_aggreg = aggreg(predict_set,ag.pi_t_j)

# COMMAND ----------

def mse(pred,true):
    return(np.sum([(t-p)**2 for t,p in zip(true,pred)])/len(true))


# COMMAND ----------

mse(predict_set,y_test_1)

# COMMAND ----------



# COMMAND ----------

fig = go.Figure()
fig.add_trace(go.Scatter( y=predict_set[0],
                mode='lines',
                name=f'xgbdecpdv'))

fig.add_trace(go.Scatter( y=predict_set[1],
                mode='lines',
                name=f'gb'))

fig.add_trace(go.Scatter( y=predict_set[2],
              mode='lines',
              name=f"lgbm",))

fig.add_trace(go.Scatter( y=pred_aggreg,
              mode='lines',
              name=f'aggreg'))

fig.add_trace(go.Scatter( y=y_test_1,
              mode='lines',
              name=f'y_true'))

fig.update_traces(mode='lines')

fig.show()

# COMMAND ----------

#[Xgb_model,Gb_model,Lgbm_model]

# COMMAND ----------

for pred in predict_set:
  print(mse(pred,y_test_1))
print(mse(pred_aggreg,y_test_1))