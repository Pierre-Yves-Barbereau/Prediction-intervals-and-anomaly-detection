# Databricks notebook source
# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/preprocessing

# COMMAND ----------

from datetime import datetime
def pinball(X:list ,q:float, alpha = 0.05):
  """Compute the alpha-pinball loss between x and q"""
  loss = 0
  for i,x in enumerate(X):
    if x <= q : 
      loss += (alpha - 1)*(x-q)
    else :
      loss += alpha*(x-q)
  return loss/(i+1)

# COMMAND ----------

import pandas as pd
def quantile_by_day(df : pd.DataFrame,alphas : list):
  """return pandas Dataframe of alphas quantiles of df by weekday"""
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

def weekday_quantile(DT_VALR,df,alpha):
  """return the alpha conditional quantile by weekday"""
  df_conditionnel = df[df["DT_VALR"].dt.weekday == DT_VALR.weekday()]
  q = df_conditionnel["Valeur"].quantile(alpha)
  return q

# COMMAND ----------

def plot_anomaly(df,bolean,title = ""):
  if "fig" in locals():
    plt.close(fig)
  fig,ax = plt.subplots()
  ax.plot(df["DT_VALR"],df["Valeur"],color="blue",label = f"Valeur",zorder=1)
  x_anomaly = [df["DT_VALR"][i] for i in range(len(df["Valeur"])) if bolean[i]]
  y_anomaly = [df["Valeur"][i] for i in range(len(df["Valeur"])) if bolean[i]]
  ax.scatter(x_anomaly,y_anomaly,color = "red",zorder=2,alpha=0.5)
  ax.legend()
  ax.set_title(title)


# COMMAND ----------

def huber_approx_of_pinball_loss05(y_pred,y_true,nu = 0.1,alpha = 0.05):
  grad = []
  hess = []
  for t in y_pred - y_true:
    if t < -nu:
      grad.append(1-alpha)
      hess.append(0)
    elif -nu <= t < 0:
      grad.append((alpha-1)*(t/nu))
      hess.append((alpha-1)/nu)
    elif 0 <= t < nu :
      grad.append(alpha*(t/nu))
      hess.append(t/nu)
    elif t > nu :
      grad.append(alpha)
      hess.append(0)
  return grad,hess

def huber_approx_of_pinball_loss95(y_pred,y_true,nu = 0.1,alpha = 0.95):
  grad = []
  hess = []
  for t in y_pred - y_true:
    if t < -nu:
      grad.append(1-alpha)
      hess.append(0)
    elif -nu <= t < 0:
      grad.append((alpha-1)*(t/nu))
      hess.append((alpha-1)/nu)
    elif 0 <= t < nu :
      grad.append(alpha*(t/nu))
      hess.append(t/nu)
    elif t > nu :
      grad.append(alpha)
      hess.append(0)
  return grad,hess

# COMMAND ----------

class Anomaly_Detection():
  def __init__(self,models_names = ["IsolationForest","Xgb","Lgbm","Gb","OneClassSVM","LocalOutlierFactor"],confidence = 0.9,n_predict = 2):
    self.n_predict = n_predict
    self.confidence = confidence
    self.models_names = []
    self.models = []
    self.distance_based_models = ["Xgb","Lgbm","Gb"]
    
    for model_name in models_names:
      if model_name == "IsolationForest":
        self.models.append(IsolationForest(n_estimators = 1000, contamination=1-confidence))
        self.models_names.append(model_name)
      elif model_name == "OneClassSVM":
        for kernel in ["poly", "rbf", "sigmoid"]: #precomputed possible
          if kernel == "poly":
            for i in range(3,10):
              self.models.append(OneClassSVM(kernel = kernel,degree = i,max_iter = 3000,gamma = 'scale',nu = 1-self.confidence))
              self.models_names.append(model_name+"_"+kernel+str(i))
          else :
            self.models.append(OneClassSVM(kernel = kernel,max_iter = 3000,gamma = 'scale',nu = 1-self.confidence))
            self.models_names.append(model_name+"_"+kernel)
          
      elif model_name == "LocalOutlierFactor":
        for n_neighbors in [10,100,1000,3000]:
          self.models.append(LocalOutlierFactor(n_neighbors=n_neighbors,contamination = 1-self.confidence))
          self.models_names.append(model_name+"_"+str(n_neighbors))

      elif model_name == "Xgb":
        self.models.append(xgb.XGBRegressor())
        self.models_names.append(model_name)

      elif model_name == "Lgbm":
        self.models.append(LGBMRegressor())
        self.models_names.append(model_name)

      elif model_name == "Gb":
        self.models.append(GradientBoostingRegressor())
        self.models_names.append(model_name)

  def fit_predict(self,historique,plot = False):  #A tester sur dftrain entier
    from sklearn import datasets, linear_model
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import GradientBoostingRegressor
    from lightgbm import LGBMRegressor
    import xgboost as xgb
    """
    lasso = linear_model.Lasso()
    y_pred = cross_val_predict(lasso, X, y, cv=3)
    """
    self.df = historique
    self.y = np.array([[self.df["Valeur"][i]] for i in range(self.df["Valeur"].shape[0])])
    self.X = self.df.drop(["Valeur","DT_VALR"],axis = 1)
    self.AD_bolean_set = []
    
    for i in range(len(self.models_names)) :
      print(i)
      print("model _ name = ",self.models_names[i])
      if self.models_names[i] in self.distance_based_models :
        y_pred = cross_val_predict(self.models[i], self.X, self.y, cv=50)
        print("y_pred : ",y_pred)
        print(np.array(np.abs(y_pred-self.y) > np.quantile(np.abs(y_pred-self.y),1-confidence)))
        print(self.models_names[i])
        self.df[self.models_names[i]] = np.multiply(np.array(np.abs(y_pred-self.y) > np.quantile(np.abs(y_pred-self.y),confidence))[0],1)
        self.AD_bolean_set.append(self.df[self.models_names[i]])

      else:
        print(self.models[i])
        print( self.models_names[i], np.multiply(self.models[i].fit_predict(self.y)==-1, 1))
        self.df[self.models_names[i]] = np.multiply(self.models[i].fit_predict(self.y)==-1, 1)
        print(self.df[self.models_names[i]])
        self.AD_bolean_set.append(self.df[self.models_names[i]])
    
    self.AD_aggreg_score = np.sum(self.AD_bolean_set,axis = 0)
    self.df["AD_aggreg_score"] = self.AD_aggreg_score
    self.AD_aggreg_score = np.sum(self.AD_bolean_set,axis = 0)
    self.df["AD_aggreg_score"] = self.AD_aggreg_score
    self.frac_anomaly_aggreg = np.sum([1 for i in self.AD_aggreg_score if i>0])/len(self.AD_aggreg_score)
    rm = regles_metier()
    rm.data_jour
    self.df = self.df.merge(rm.data_jour, left_on ="DT_VALR",right_on="Date")
    self.df["AD_aggreg_score"][np.sum(self.df.iloc[:,-3:],axis = 1).values !=0 ] = 0

    if plot:
      fig1 = px.line(self.df, x="DT_VALR", y="Valeur")
      fig1.update_traces(line=dict(color = 'rgba(50,50,50,0.2)'))
      fig2 = px.scatter(self.df, x="DT_VALR", y="Valeur",size ="AD_aggreg_score",color ="AD_aggreg_score")
      fig = go.Figure(data=fig1.data + fig2.data)
      print("frac_anomaly_aggreg = ",self.frac_anomaly_aggreg)
      fig.update_layout(title = f" AD aggreg score")    
      #fig.update_traces(mode='lines')                  
      fig.show()
    return self.AD_aggreg_score
  
  def plot(self,model):
    fig1 = px.line(self.df, x="DT_VALR", y="Valeur")
    fig1.update_traces(line=dict(color = 'rgba(50,50,50,0.2)'))
    fig2 = px.scatter(self.df, x="DT_VALR", y="Valeur",size =model,color =model)
    fig = go.Figure(data=fig1.data + fig2.data)
    print("frac_anomaly_aggreg = ",self.frac_anomaly_aggreg)
    fig.update_layout(title = f"{model} AD aggreg score")    
    #fig.update_traces(mode='lines')                  
    fig.show()
    return self.AD_aggreg_score

# COMMAND ----------

class LOF():
  def __init__(self,df):
    import numpy as np
    self.df = df
    self.X = self.df["Valeur"]

  def k_dist(self,x,k):
    return([np.abs(x-xi) for xi in self.X][k])
  
  def N_voisinage(self,x,n):
    return([np.abs(x-xi) <= self.k_dist(x,n) for xi in self.X])

  def reach_distance(self,a,b,k):
    return(np.max([self.k_dist(b,k),np.abs(a-b)]))
  
  def local_reachability_density(self,p,minpts):
    return(np.sum(self.N_voisinage(p,minpts))/np.sum([self.reach_distance(p,o,minpts) for o in self.X[self.N_voisinage(p,minpts)]]))

  def local_outlier_factor(self,p,minpts):
    return(sum([self.local_reachability_density(o,minpts)/self.local_reachability_density(p,minpts) for o in self.X[self.N_voisinage(p,minpts)] ])/np.sum(self.N_voisinage(p,minpts)))

# COMMAND ----------

class regles_metier:
  def __init__(self):
    self.data_jour= spark.sql("select * from jours_target_csv") \
                      .withColumn("Date", to_timestamp("Date", "yyyy-MM-dd")) \
                      .drop("_c5", 'Jour_target_UK', 'Jour_target_US').toPandas()
    self.data_jour["Is Weekend"] = (self.data_jour["Date"].dt.day_name() == "Sunday").values + (self.data_jour["Date"].dt.day_name() == "Saturday").values


# COMMAND ----------

class distance_based_ad():
  def __init(self,model,confidence):
    self.model = model
    self.confidence = self.confidence
  def fit_predict(y):
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import GradientBoostingRegressor
    from lightgbm import LGBMRegressor
    import xgboost as xgb
    y_pred = cross_val_predict(model, self.X, y, cv=50)
    return np.array(np.abs(y_pred-y) > np.quantile(np.abs(y_pred-y),1-confidence))-1