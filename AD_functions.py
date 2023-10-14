# Databricks notebook source
class Anomaly_Detection_Dataset_score_cuted2: 
  """Anomaly detection for dataset"""
  def __init__(self,labels_str,labels):
    self.labels_str = labels_str
    self.labels = labels
    self.score = {}
    self.dico = {}

  def fit_predict(self,historique,plot = False,seuil = 0.9): #ticks 10% d'anomalies
    self.model = IsolationForest(n_estimators = 1000)
    self.df = historique
    DT_VALR = historique["DT_VALR"]
    labels = list(set(historique["label"]))
    warnings.filterwarnings("ignore")
    self.df["AD_score"] = 0
    for label,label_str in zip(self.labels,self.labels_str):
      self.X = self.df[self.df["label"] == label]
      self.X = self.X.drop(["DT_VALR"],axis = 1)
      self.model.fit(self.X)
      self.score = -self.model.score_samples(self.X)
      pred = len(self.df[self.df["label"] == label])*self.score/np.sum(self.score)
      self.df.loc[self.df["label"] == label,["AD_score"]] = pred

    self.AD_mean_score = self.df["AD_score"]
    self.AD_score = (self.AD_mean_score - np.min(self.AD_mean_score))/(np.max(self.AD_mean_score) - np.min(self.AD_mean_score))
    self.df["AD_score"] = self.AD_score
    self.df.loc[self.df["Valeur"]==0,["AD_score"]] = 0
    self.df.loc[self.df["AD_score"] <= np.quantile(self.df["AD_score"],seuil),["AD_score"]] = 0
    self.df['DT_VALR'] = DT_VALR

    if plot:
      fig1 = px.line(self.df, x = "DT_VALR", y="Valeur")
      fig1.update_traces(line = dict(color = 'rgba(50,50,50,0.2)'))
      fig2 = px.scatter(self.df, x = "DT_VALR", y="Valeur",size = "AD_score",color = "AD_score")
      fig = go.Figure(data = fig1.data + fig2.data)
      fig.update_layout(title = f"AD_score")    
      #fig.update_traces(mode='lines')                  
      fig.show()
    return self.AD_mean_score

# COMMAND ----------

class Anomaly_Detection_score_cuted(): 
  """anomaly detection with reconstruction methods best for "Valeur" target"""
  def __init__(self,groupe , labels_str,labels , models_names = ["IsolationForest"],confidence = 0.9,dico_hyperparametres = None,gridsearch = False,param_grid_gb= None,param_grid_lgbm = None): #["IsolationForest","LGBM","GradientBoosting","LocalOutlierFactor","OneClassSVM"]
    self.groupe = groupe
    self.labels_str = labels_str
    self.labels = labels
    self.confidence = confidence
    self.models_names = models_names
    self.param_grid_gb = param_grid_gb
    self.param_grid_lgbm = param_grid_lgbm
    self.models = []
    self.distance_based_models = ["LGBM","GradientBoosting","QuantileRandomForest"]
    self.alpha_bonferonni = (1- confidence)/len(models_names)
    self.confidence_bonferonni = 1 - self.alpha_bonferonni
    self.dico_hyperparametres = dico_hyperparametres
    if dico_hyperparametres == None :
      self.dico_hyperparametres = {}
    self.gridsearch = gridsearch
    self.anomaly_score_set = []
    self.score = {}
    self.dico = {}
    for model_name in self.models_names:
      if model_name == "IsolationForest":
        self.dico[model_name] = IsolationForest(n_estimators = 1000)
        self.models.append(IsolationForest(n_estimators = 1000))

  def fit_predict(self,historique,plot = False,seuil = 0.7):  #A tester sur dftrain entier
    """
    lasso = linear_model.Lasso()
    y_pred = cross_val_predict(lasso, X, y, cv=3)
    """
    DT_VALR = historique["DT_VALR"]
    self.hist = historique.copy(deep = True)
    self.df = historique.copy(deep = True)
    labels = list(set(historique["label"]))
    warnings.filterwarnings("ignore")

    for label,label_str in zip(self.labels,self.labels_str):
      if self.gridsearch:
        self.dico_hyperparametres[label_str] = {}
      self.y = [[self.df["Valeur"][i]] for i in range(self.df["Valeur"].shape[0]) if self.df['label'][i] == label]
      self.X = self.hist.copy(deep = True)[self.df['label'] == label].drop(["Valeur","DT_VALR","label"],axis = 1)
      for i,model in enumerate(self.models_names) :
        #print(int(100*t/len(labels) + 100*i/(len(labels)*len(self.models_names))),"%")
        #print(i)
        #print("model _ name = ",self.models_names[i])
        if model in self.distance_based_models :
          distancemodel_score = distance_model_score(model = model,confidence = self.confidence_bonferonni,label_str = label_str,dico_hyperparametres = self.dico_hyperparametres,gridsearch = self.gridsearch,param_grid_gb = self.param_grid_gb,param_grid_lgbm = self.param_grid_gb)
          #y_pred = cross_val_predict(self.models[i], self.X, self.y, cv=3)
          #print("y_pred : ",y_pred)
          #print(np.array(np.abs(y_pred-self.y) > np.quantile(np.abs(y_pred-self.y),confidence)))
          pred = list(distancemodel_score.fit_predict(self.X,self.y))
          self.anomaly_score_set.append(pred)
          self.df.loc[self.df["label"] == label,model] = pred
        else:
          #print(self.models[i])
          #print(self.models_names[i], np.multiply(self.models[i].fit_predict(self.y)==-1, 1))
          self.dico[model].fit(self.y)
          if model == "IsolationForest":
            self.score[model] = -self.dico[model].score_samples(self.y)
          else :
            self.score[model] =  self.dico[model].score_samples(self.y)
          pred = (self.score[model]/np.linalg.norm(self.score[model]))
          self.anomaly_score_set.append(pred)
          self.df.loc[self.df["label"] == label,model] = pred*len(self.df.loc[self.df["label"] == label])
          #self.AD_bolean_set.append(self.df.loc[self.df["label"] == label,self.models_names[i]])
    if self.gridsearch :
            self.dico_hyperparametres[groupe] = distancemodel_score.dico_hyperparametres
            
    for model in self.models_names:
      self.df[model] = sorted(range(len(self.df[model])), key=lambda k: self.df[model][k])
    self.AD_mean_score = np.mean(self.df[self.models_names],axis = 1)
    self.AD_score = (self.AD_mean_score - np.min(self.AD_mean_score))/np.max(self.AD_mean_score)
    self.df["AD_score"] = self.AD_score
    self.df.loc[self.df["Valeur"]==0,["AD_score"]] = 0
    self.df.loc[self.df["AD_score"] <= np.quantile(self.df["AD_score"],seuil),["AD_score"]] = 0


    if plot:
      fig1 = px.line(self.df, x = "DT_VALR", y="Valeur")
      fig1.update_traces(line = dict(color = 'rgba(50,50,50,0.2)'))
      fig2 = px.scatter(self.df, x = "DT_VALR", y="Valeur",size = "AD_score",color = "AD_score")
      fig = go.Figure(data = fig1.data + fig2.data)
      fig.update_layout(title = f"AD_score")    
      #fig.update_traces(mode='lines')                  
      fig.show()
    self.hist["DT_VALR"] = DT_VALR
    df_output = self.hist[["DT_VALR","Valeur"]]
    df_output["AD_score"] = self.AD_mean_score
    return(df_output)


# COMMAND ----------

class distance_model_score:
  def __init__(self,model,confidence = 0.9,label_str = None,dico_hyperparametres = None,gridsearch = False,param_grid_gb = None,param_grid_lgbm = None):
    self.label_str = label_str
    """
    retourne un score d'anomaly pour les methodes de reconstruction
    model_names = ["IsolationForest","Lgbm","Gb","LocalOutlierFactor","quantile regression forest"]
    """
    self.model = model
    self.confidence = confidence
    self.alpha_down = (1-confidence)/2 
    self.alpha_up =  1 - (1-confidence)/2 
    self.dico_hyperparametres = dico_hyperparametres
    self.gridsearch = gridsearch
    if gridsearch:
      
      self.dico_hyperparametres[self.label_str][self.model] = {}
      self.dico_hyperparametres[self.label_str][self.model]["up"] = {}
      self.dico_hyperparametres[self.label_str][self.model]["down"] = {}
      
  def fit_predict(self,X,y):
    temp=[]
    for i in range(len(y)):
      temp.append(y[i][0])
    y = temp

    if self.model == "LGBM":
      if self.gridsearch :
        mqloss_scorer_up = make_scorer(mqloss, alpha=0.90) #quantile scorer pour le gridsearch
        mqloss_scorer_down = make_scorer(mqloss, alpha=0.05) #quantile scorer pour le gridsearch
        gridsearch_up = GridSearchCV(
                          estimator=LGBMRegressor(alpha = self.alpha_up,objective = "quantile"),
                          param_grid=self.param_grid_lgbm,
                          cv=3,
                          scoring=mqloss_scorer_up,
                          n_jobs=-1,
                          verbose=1
                          )
        model_up = gridsearch_up.fit(X, y).best_estimator_
        print("model_up.best_params = ", gridsearch_up.best_params_)
        self.dico_hyperparametres[self.label_str][self.model]["up"] =  gridsearch_up.best_params_

        gridsearch_down = GridSearchCV(
                          estimator=LGBMRegressor(alpha = self.alpha_down,objective = "quantile"),
                          param_grid=self.param_grid_lgbm,
                          cv=3,
                           scoring=mqloss_scorer_down,
                          n_jobs=-1,
                          verbose=1
                          )
        model_down = gridsearch_down.fit(X, y).best_estimator_
        print("model_down.best_params = ", gridsearch_down.best_params_)
        self.dico_hyperparametres[self.label_str][self.model]["down"] =  gridsearch_down.best_params_

      else:

        model_down = LGBMRegressor(alpha = self.alpha_down,objective='quantile',**self.dico_hyperparametres[self.label_str][self.model]["down"])
        model_up = LGBMRegressor(alpha = self.alpha_up,objective = "quantile",**self.dico_hyperparametres[self.label_str][self.model]["up"])



    elif self.model == "GradientBoosting":
      if self.gridsearch :
        mqloss_scorer_up = make_scorer(mqloss, alpha=0.90)
        mqloss_scorer_down = make_scorer(mqloss, alpha=0.05)
        mqloss_scorer_median = make_scorer(mqloss, alpha=0.5)
        gridsearch_up = GridSearchCV(
                          estimator = GradientBoostingRegressor(loss="quantile", alpha=self.alpha_up,n_estimators = 200,
                                                        learning_rate = 0.01,max_depth = 5),
                          param_grid=self.param_grid_gb,
                          cv=3,
                          scoring=mqloss_scorer_up,
                          n_jobs=-1,
                          verbose=1
                          )
        model_up = gridsearch_up.fit(X, y).best_estimator_
        print("model_up.best_params = ", gridsearch_up.best_params_)
        self.dico_hyperparametres[self.label_str][self.model]["up"] =  gridsearch_up.best_params_

        gridsearch_down = GridSearchCV(
                          estimator = GradientBoostingRegressor(loss="quantile", alpha=self.alpha_down,n_estimators = 200,
                                                        learning_rate = 0.01,max_depth = 5),
                          param_grid=self.param_grid_gb,
                          cv=3,
                          scoring=mqloss_scorer_down,
                          n_jobs=-1,
                          verbose=1
                          )
        model_down = gridsearch_down.fit(X, y).best_estimator_
        print("model_down.best_params = ", gridsearch_up.best_params_)
        self.dico_hyperparametres[self.label_str][self.model]["down"] =  gridsearch_down.best_params_

      else:
        model_down = GradientBoostingRegressor(loss="quantile", alpha=self.alpha_down,**self.dico_hyperparametres[self.label_str][self.model]["down"])
        model_up = GradientBoostingRegressor(loss="quantile", alpha=self.alpha_up,**self.dico_hyperparametres[self.label_str][self.model]["up"])
      
      
    else:
      print('Please enter a correct model in LGBM,QuantileRandomForest or GradientBoosting')

    self.pred_up = cross_val_predict(model_up, X, y, cv=10) #cross validation prediction
    self.pred_down = cross_val_predict(model_down, X, y, cv=10) #cross validation prediction
    """
    self.ad_down = (self.pred_down - np.array(y)) #score d'anomalie
    print("self.ad_down = ",self.ad_down)
    self.ad_down = self.ad_down - np.min(self.ad_down)
    print("self.ad_down positive = ",self.ad_down)
    self.ad_up = (np.array(y) - self.pred_up)
    print("self.ad_up = ",self.ad_up)
    self.ad_up = self.ad_up - np.min(self.ad_up)  
    print("self.ad_up positive = ",self.ad_up)"""

    score = np.maximum((self.pred_down - np.array(y)),(np.array(y) - self.pred_up))/(self.pred_up-self.pred_down)
    score = score - np.min(score)
    score = score/np.linalg.norm(score)
    return(score)
  
    output = self.ad_down + self.ad_up #addition des scores d'anomalies
    
    return(output/np.linalg.norm(output))

# COMMAND ----------

class Anomaly_Detection_Dataset_score_cuted(): 
  def __init__(self,groupe , labels_str,labels , models_names = ["IsolationForest"],confidence = 0.9,dico_hyperparametres = None,gridsearch = False,param_grid_gb= None,param_grid_lgbm = None): #["IsolationForest","LGBM","GradientBoosting","LocalOutlierFactor","OneClassSVM"]
    self.groupe = groupe
    self.labels_str = labels_str
    self.labels = labels
    self.confidence = confidence
    self.models_names = models_names
    self.param_grid_gb
    self.param_grid_lgbm
    self.models = []
    self.distance_based_models = ["LGBM","GradientBoosting","QuantileRandomForest"]
    self.alpha_bonferonni = (1- confidence)/len(models_names)
    self.confidence_bonferonni = 1 - self.alpha_bonferonni
    if dico_hyperparametres == None :
      self.dico_hyperparametres = {}
      self.dico_hyperparametres[groupe] = {}
    else : 
       self.dico_hyperparametres = dico_hyperparametres
  
    self.gridsearch = gridsearch
    self.anomaly_score_set = []
    self.score = {}
    self.dico = {}
    for model_name in self.models_names:
      if model_name == "IsolationForest":
        self.dico[model_name] = IsolationForest(n_estimators = 2000, contamination=self.alpha_bonferonni)

      elif model_name == "OneClassSVM":
        self.dico[model_name] = OneClassSVM(kernel = "rbf",max_iter = 3000,gamma = 'scale',nu =self.alpha_bonferonni)
          
      elif model_name == "LocalOutlierFactor":
        self.dico[model_name] = LocalOutlierFactor(n_neighbors=100,contamination = self.alpha_bonferonni)

      elif model_name == "SGDOneClassSVM":
        self.dico[model_name] = SGDOneClassSVM(nu=self.alpha_bonferonni, shuffle=True, fit_intercept=True, random_state=42, tol=1e-4)
      
      elif model_name == "eif":
        self.dico[model_name] = eif.iForest(ntrees=10000, contamination = self.alpha_bonferonni, ExtensionLevel=1)

    """
      if model_name == "IsolationForest":
        self.models.append(IsolationForest(n_estimators = 1000, contamination=1-confidence))
        self.models_names.append(model_name)

      elif model_name == "OneClassSVM":
        for kernel in ["poly", "rbf", "sigmoid"]: #precomputed possible
          if kernel == "poly":
            for i in range(3,10):
              self.models.append(OneClassSVM(kernel = kernel,degree = i,max_iter = 3000,gamma = 'scale',nu = 1 - self.confidence))
              self.models_names.append(model_name+"_"+kernel+str(i))
          else :
            self.models.append(OneClassSVM(kernel = kernel,max_iter = 3000,gamma = 'scale',nu = 1 - self.confidence))
            self.models_names.append(model_name+"_"+kernel)
          
      elif model_name == "LocalOutlierFactor":
        for n_neighbors in [31,93,365]:
          self.models.append(LocalOutlierFactor(n_neighbors=n_neighbors,contamination = 1-self.confidence))
          self.models_names.append(model_name+"_"+str(n_neighbors))

      elif model_name == "Lgbm":
        self.models.append(LGBMRegressor())
        self.models_names.append(model_name)

      elif model_name == "Gb":
        self.models.append(GradientBoostingRegressor())
        self.models_names.append(model_name)
        """

  def fit_predict(self,historique,plot = False):  #A tester sur dftrain entier
    """
    lasso = linear_model.Lasso()
    y_pred = cross_val_predict(lasso, X, y, cv=3)
    """
    normalizer = preprocessing.normalize()
    self.df = normalizer(historique.copy(deep = True))
    labels = list(set(historique["label"]))
    warnings.filterwarnings("ignore")

    for label,label_str in zip(self.labels,self.labels_str):
      if self.gridsearch:
        self.dico_hyperparametres[groupe] = {}
        self.dico_hyperparametres[groupe][label_str] = {}
      self.X = self.df[self.df['label'] == label].drop(["DT_VALR","label"],axis = 1)
      for i,model in enumerate(self.models_names) :
        #print(int(100*t/len(labels) + 100*i/(len(labels)*len(self.models_names))),"%")
        print(model)
        #print(i)
        #print("model _ name = ",self.models_names[i])
        """
        if model in self.distance_based_models :
          distancemodel_score = distance_model_score(model = model,confidence = self.confidence_bonferonni,label_str = label_str,dico_hyperparametres = self.dico_hyperparametres[groupe],gridsearch = self.gridsearch,param_grid_gb = self.param_grid_gb,param_grid_lgbm = self.param_grid_gb)
          #y_pred = cross_val_predict(self.models[i], self.X, self.y, cv=3)
          #print("y_pred : ",y_pred)
          #print(np.array(np.abs(y_pred-self.y) > np.quantile(np.abs(y_pred-self.y),confidence)))
          X_distance = self.X.drop(["Valeur"]axis = 1)
          pred = list(distancemodel_score.fit_predict(X_distance,self.hist["Valeur"]))
          print("anomaly_score = ",pred)
          self.anomaly_score_set.append(pred)
          self.df.loc[self.df["label"] == label,model] = pred
      else:
        """
        #print(self.models[i])
        #print(self.models_names[i], np.multiply(self.models[i].fit_predict(self.y)==-1, 1))
        self.dico[model].fit(self.y)
        if model == "LocalOutlierFactor":
          self.score[model] =  -self.dico[model].negative_outlier_factor_

        elif model == "IsolationForest":
          self.score[model] = -self.dico[model].score_samples(self.X)

        elif model == "eif":
          self.score[model] = self.dico[model].compute_paths(X_in=self.X)

        else :
          self.score[model] =  self.dico[model].score_samples(self.X)
          
        pred = (self.score[model]/np.linalg.norm(self.score[model]))
        self.anomaly_score_set.append(pred)
        self.df.loc[self.df["label"] == label,model] = pred*len(self.df.loc[self.df["label"] == label])
        #self.AD_bolean_set.append(self.df.loc[self.df["label"] == label,self.models_names[i]])

    for model in model_names:
      self.df[model] = sorted(range(len(self.df[model])), key=lambda k: self.df[model][k])
    self.AD_mean_score = np.mean(self.df[self.models_names],axis = 1)**2
    self.AD_score = (self.AD_mean_score - np.min(self.AD_mean_score))/np.max(self.AD_mean_score)
    self.df["AD_score"] = self.AD_score
    self.df.loc[self.df["Valeur"]==0,["AD_score"]] = 0
    self.df.loc[self.df["Valeur"]==1,["AD_score"]] = 0

    if plot:
      fig1 = px.line(self.df, x = "DT_VALR", y="Valeur")
      fig1.update_traces(line = dict(color = 'rgba(50,50,50,0.2)'))
      fig2 = px.scatter(self.df, x = "DT_VALR", y="Valeur",size = "AD_score",color = "AD_score")
      fig = go.Figure(data = fig1.data + fig2.data)
      fig.update_layout(title = f"AD_score")    
      #fig.update_traces(mode='lines')                  
      fig.show()
    return self.AD_mean_score

# COMMAND ----------

class Anomaly_Detection_Dataset_score_cuted(): 
  def __init__(self,groupe , labels_str,labels , models_names = ["IsolationForest"],confidence = 0.9,dico_hyperparametres = None,gridsearch = False,param_grid_gb= None,param_grid_lgbm = None): #["IsolationForest","LGBM","GradientBoosting","LocalOutlierFactor","OneClassSVM"]
    self.groupe = groupe
    self.labels_str = labels_str
    self.labels = labels
    self.confidence = confidence
    self.models_names = models_names
    self.param_grid_gb
    self.param_grid_lgbm
    self.models = []
    self.distance_based_models = ["LGBM","GradientBoosting","QuantileRandomForest"]
    self.alpha_bonferonni = (1- confidence)/len(models_names)
    self.confidence_bonferonni = 1 - self.alpha_bonferonni
    if dico_hyperparametres == None :
      self.dico_hyperparametres = {}
      self.dico_hyperparametres[groupe] = {}
    else : 
       self.dico_hyperparametres = dico_hyperparametres
  
    self.gridsearch = gridsearch
    self.anomaly_score_set = []
    self.score = {}
    self.dico = {}
    for model_name in self.models_names:
      if model_name == "IsolationForest":
        self.dico[model_name] = IsolationForest(n_estimators = 2000, contamination=self.alpha_bonferonni)

      elif model_name == "OneClassSVM":
        self.dico[model_name] = OneClassSVM(kernel = "rbf",max_iter = 3000,gamma = 'scale',nu =self.alpha_bonferonni)
          
      elif model_name == "LocalOutlierFactor":
        self.dico[model_name] = LocalOutlierFactor(n_neighbors=100,contamination = self.alpha_bonferonni)

      elif model_name == "SGDOneClassSVM":
        self.dico[model_name] = SGDOneClassSVM(nu=self.alpha_bonferonni, shuffle=True, fit_intercept=True, random_state=42, tol=1e-4)
      
      elif model_name == "eif":
        self.dico[model_name] = eif.iForest(ntrees=10000, contamination = self.alpha_bonferonni, ExtensionLevel=1)

    """
      if model_name == "IsolationForest":
        self.models.append(IsolationForest(n_estimators = 1000, contamination=1-confidence))
        self.models_names.append(model_name)

      elif model_name == "OneClassSVM":
        for kernel in ["poly", "rbf", "sigmoid"]: #precomputed possible
          if kernel == "poly":
            for i in range(3,10):
              self.models.append(OneClassSVM(kernel = kernel,degree = i,max_iter = 3000,gamma = 'scale',nu = 1 - self.confidence))
              self.models_names.append(model_name+"_"+kernel+str(i))
          else :
            self.models.append(OneClassSVM(kernel = kernel,max_iter = 3000,gamma = 'scale',nu = 1 - self.confidence))
            self.models_names.append(model_name+"_"+kernel)
          
      elif model_name == "LocalOutlierFactor":
        for n_neighbors in [31,93,365]:
          self.models.append(LocalOutlierFactor(n_neighbors=n_neighbors,contamination = 1-self.confidence))
          self.models_names.append(model_name+"_"+str(n_neighbors))

      elif model_name == "Lgbm":
        self.models.append(LGBMRegressor())
        self.models_names.append(model_name)

      elif model_name == "Gb":
        self.models.append(GradientBoostingRegressor())
        self.models_names.append(model_name)
        """

  def fit_predict(self,historique,plot = False):  #A tester sur dftrain entier
    """
    lasso = linear_model.Lasso()
    y_pred = cross_val_predict(lasso, X, y, cv=3)
    """
    normalizer = preprocessing.normalize()
    self.df = normalizer(historique.copy(deep = True))
    labels = list(set(historique["label"]))
    warnings.filterwarnings("ignore")

    for label,label_str in zip(self.labels,self.labels_str):
      if self.gridsearch:
        self.dico_hyperparametres[groupe] = {}
        self.dico_hyperparametres[groupe][label_str] = {}
      self.X = self.df[self.df['label'] == label].drop(["DT_VALR","label"],axis = 1)
      for i,model in enumerate(self.models_names) :
        #print(int(100*t/len(labels) + 100*i/(len(labels)*len(self.models_names))),"%")
        print(model)
        #print(i)
        #print("model _ name = ",self.models_names[i])
        """
        if model in self.distance_based_models :
          distancemodel_score = distance_model_score(model = model,confidence = self.confidence_bonferonni,label_str = label_str,dico_hyperparametres = self.dico_hyperparametres[groupe],gridsearch = self.gridsearch,param_grid_gb = self.param_grid_gb,param_grid_lgbm = self.param_grid_gb)
          #y_pred = cross_val_predict(self.models[i], self.X, self.y, cv=3)
          #print("y_pred : ",y_pred)
          #print(np.array(np.abs(y_pred-self.y) > np.quantile(np.abs(y_pred-self.y),confidence)))
          X_distance = self.X.drop(["Valeur"]axis = 1)
          pred = list(distancemodel_score.fit_predict(X_distance,self.hist["Valeur"]))
          print("anomaly_score = ",pred)
          self.anomaly_score_set.append(pred)
          self.df.loc[self.df["label"] == label,model] = pred
      else:
        """
        #print(self.models[i])
        #print(self.models_names[i], np.multiply(self.models[i].fit_predict(self.y)==-1, 1))
        self.dico[model].fit(self.y)
        if model == "LocalOutlierFactor":
          self.score[model] =  -self.dico[model].negative_outlier_factor_

        elif model == "IsolationForest":
          self.score[model] = -self.dico[model].score_samples(self.X)

        elif model == "eif":
          self.score[model] = self.dico[model].compute_paths(X_in=self.X)

        else :
          self.score[model] =  self.dico[model].score_samples(self.X)
          
        pred = (self.score[model]/np.linalg.norm(self.score[model]))
        self.anomaly_score_set.append(pred)
        self.df.loc[self.df["label"] == label,model] = pred*len(self.df.loc[self.df["label"] == label])
        #self.AD_bolean_set.append(self.df.loc[self.df["label"] == label,self.models_names[i]])

    for model in model_names:
      self.df[model] = sorted(range(len(self.df[model])), key=lambda k: self.df[model][k])
    self.AD_mean_score = np.mean(self.df[self.models_names],axis = 1)**2
    self.AD_score = (self.AD_mean_score - np.min(self.AD_mean_score))/np.max(self.AD_mean_score)
    self.df["AD_score"] = self.AD_score
    self.df.loc[self.df["Valeur"]==0,["AD_score"]] = 0
    self.df.loc[self.df["Valeur"]==1,["AD_score"]] = 0

    if plot:
      fig1 = px.line(self.df, x = "DT_VALR", y="Valeur")
      fig1.update_traces(line = dict(color = 'rgba(50,50,50,0.2)'))
      fig2 = px.scatter(self.df, x = "DT_VALR", y="Valeur",size = "AD_score",color = "AD_score")
      fig = go.Figure(data = fig1.data + fig2.data)
      fig.update_layout(title = f"AD_score")    
      #fig.update_traces(mode='lines')                  
      fig.show()
    return self.AD_mean_score

# COMMAND ----------

import numpy as np
np.array([1,2,3])**2

# COMMAND ----------



# COMMAND ----------

class Anomaly_Detection_score_cuted_old(): 
  def __init__(self,groupe,labels = None,models_names = ["OneClassSVM","LocalOutlierFactor","IsolationForest","GradientBoosting","LGBM",],confidence = 0.9): #["IsolationForest","LGBM","GradientBoosting","LocalOutlierFactor","OneClassSVM"]
    self.confidence = confidence
    self.models_names = models_names
    self.models = []
    self.distance_based_models = ["LGBM","GradientBoosting","QuantileRandomForest"]
    self.alpha_bonferonni = (1- confidence)/len(models_names)
    self.confidence_bonferonni = 1 - self.alpha_bonferonni
    self.groupe = groupe
    self.labels = labels

    self.dico = {}
    
    for model_name in self.models_names:
      if model_name == "IsolationForest":
        self.dico[model_name] = IsolationForest(n_estimators = 1000, contamination=self.alpha_bonferonni)
        self.models.append(IsolationForest(n_estimators = 1000, contamination=self.alpha_bonferonni))

      elif model_name == "OneClassSVM":
        self.dico[model_name] = OneClassSVM(kernel = "rbf",max_iter = 3000,gamma = 'scale',nu =self.alpha_bonferonni)
        self.models.append(OneClassSVM(kernel = "rbf",max_iter = 3000,gamma = 'scale',nu =self.alpha_bonferonni))
          
      elif model_name == "LocalOutlierFactor":
        self.dico[model_name] = LocalOutlierFactor(n_neighbors=100,contamination = self.alpha_bonferonni)
        self.models.append(LocalOutlierFactor(n_neighbors=100,contamination = self.alpha_bonferonni))


    """
      if model_name == "IsolationForest":
        self.models.append(IsolationForest(n_estimators = 1000, contamination=1-confidence))
        self.models_names.append(model_name)

      elif model_name == "OneClassSVM":
        for kernel in ["poly", "rbf", "sigmoid"]: #precomputed possible
          if kernel == "poly":
            for i in range(3,10):
              self.models.append(OneClassSVM(kernel = kernel,degree = i,max_iter = 3000,gamma = 'scale',nu = 1 - self.confidence))
              self.models_names.append(model_name+"_"+kernel+str(i))
          else :
            self.models.append(OneClassSVM(kernel = kernel,max_iter = 3000,gamma = 'scale',nu = 1 - self.confidence))
            self.models_names.append(model_name+"_"+kernel)
          
      elif model_name == "LocalOutlierFactor":
        for n_neighbors in [31,93,365]:
          self.models.append(LocalOutlierFactor(n_neighbors=n_neighbors,contamination = 1-self.confidence))
          self.models_names.append(model_name+"_"+str(n_neighbors))

      elif model_name == "Lgbm":
        self.models.append(LGBMRegressor())
        self.models_names.append(model_name)

      elif model_name == "Gb":
        self.models.append(GradientBoostingRegressor())
        self.models_names.append(model_name)
        """

  def fit_predict(self,historique,plot = False,gridsearch = False,param_grid_gb= None,param_grid_lgbm = None,dico_hyperparametres = None):  #A tester sur dftrain entier
    
    """
    lasso = linear_model.Lasso()
    y_pred = cross_val_predict(lasso, X, y, cv=3)
    """
    if dico_hyperparametres == None:
      self.dico_hyperparametres = {}
    self.hist = historique.copy(deep = True)
    self.df = historique.copy(deep = True)
    self.anomaly_score_set = []
    warnings.filterwarnings("ignore")
    self.score = {}
    if gridsearch:
      self.dico_hyperparametres[groupe] = {}

    for t,label in enumerate(self.labels):
      self.y = [[self.df["Valeur"][i]] for i in range(self.df["Valeur"].shape[0]) if self.df['label'][i] == label]
      print("self.y = ", self.y)
      self.X = self.hist.copy(deep = True)[self.df['label'] == label].drop(["Valeur","DT_VALR","label"],axis = 1)
      if gridsearch:
        self.dico_hyperparametres[label] = {}
      for i,model in enumerate(self.models_names) :
        print(int(100*t/len(labels) + 100*i/(len(labels)*len(self.models_names))),"%")
        print(model)
        #print(i)
        #print("model _ name = ",self.models_names[i])
        if model in self.distance_based_models :
          distancemodel_score = distance_model_score(model = model,confidence = self.confidence_bonferonni,label = label,dico_hyperparametres = self.dico_hyperparametres[groupe])
          #y_pred = cross_val_predict(self.models[i], self.X, self.y, cv=3)
          #print("y_pred : ",y_pred)
          #print(np.array(np.abs(y_pred-self.y) > np.quantile(np.abs(y_pred-self.y),confidence)))
          pred = list(distancemodel_score.fit_predict(self.X,self.y,gridsearch = gridsearch,param_grid_gb= param_grid_gb,param_grid_lgbm = param_grid_lgbm))
          self.anomaly_score_set.append(pred)
          self.df.loc[self.df["label"] == label,model] = pred
          if gridsearch:
            self.dico_hyperparametres[groupe][label]= distancemodel_score.dico_hyperparametres[label]
        else:
          #print(self.models[i])
          #print(self.models_names[i], np.multiply(self.models[i].fit_predict(self.y)==-1, 1))
          print(self.y)
          print(self.dico[model])
          self.dico[model].fit(self.y)
          if model == "LocalOutlierFactor":
            self.score[model] =  -self.dico[model].negative_outlier_factor_
          elif model == "IsolationForest":
            self.score[model] = -self.dico[model].score_samples(self.y)
          else :
            self.score[model] =  self.dico[model].score_samples(self.y)
          pred = (self.score[model]/np.linalg.norm(self.score[model]))
          self.anomaly_score_set.append(pred)
          self.df.loc[self.df["label"] == label,model] = pred
          #self.AD_bolean_set.append(self.df.loc[self.df["label"] == label,self.models_names[i]])

     
    self.AD_mean_score = np.mean(self.df[self.models_names],axis = 1)
    self.AD_score = self.AD_mean_score/np.max(self.AD_mean_score)
    self.df["AD_score"] = self.AD_score
    self.df.loc[self.df["Valeur"]==0,["AD_score"]] = 0
    self.df.loc[self.df["Valeur"]==1,["AD_score"]] = 0

    if plot:
      fig1 = px.line(self.df, x = "DT_VALR", y="Valeur")
      fig1.update_traces(line = dict(color = 'rgba(50,50,50,0.2)'))
      fig2 = px.scatter(self.df, x = "DT_VALR", y="Valeur",size = "AD_score",color = "AD_score")
      fig = go.Figure(data = fig1.data + fig2.data)
      fig.update_layout(title = f"AD_score_{self.groupe}")    
      #fig.update_traces(mode='lines')                  
      fig.show()
    return self.AD_mean_score
  

# COMMAND ----------

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

def weekday_quantile(DT_VALR,df,alpha):
  """return the alpha conditional quantile by weekday"""
  df_conditionnel = df[df["DT_VALR"].dt.weekday == DT_VALR.weekday()]
  q = df_conditionnel["Valeur"].quantile(alpha)
  return q

# COMMAND ----------

def plot_anomaly(df,bolean,title = ""):
  """
  plot anomaly of a df with bolean of anomalies
  """
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

class Anomaly_Detection_old():
  """
  Detect anomalies 
  """
  def __init__(self,models_names = ["IsolationForest","Xgb","Lgbm","Gb","OneClassSVM","LocalOutlierFactor"],confidence = 0.9,n_predict = 2):
    self.n_predict = n_predict
    self.confidence = confidence
    self.models_names = []
    self.models = []
    self.distance_based_models = ["Xgb","Lgbm","Gb"]
    

  def fit_predict(self,historique,plot = False,gridsearch = False):  
    """
    fit the model and return a bolean of anomalies
    """
    self.df = historique
    self.y = np.array([[self.df["Valeur"][i]] for i in range(self.df["Valeur"].shape[0])])
    self.X = self.df.drop(["Valeur","DT_VALR"],axis = 1)
    self.AD_bolean_set = []


    for model_name in models_names:
      if model_name == "IsolationForest":
        self.models.append(IsolationForest(n_estimators = 1000, contamination=1-confidence))
        self.models_names.append(model_name)
      elif model_name == "OneClassSVM":
        for kernel in ["linear","poly", "rbf", "sigmoid"]: #precomputed possible
          if kernel == "poly":
            for i in range(3,10):
              self.models.append(OneClassSVM(kernel = kernel,degree = i,max_iter = 3000,gamma = 'scale',nu = 1-self.confidence))
              self.models_names.append(model_name+"_"+kernel+str(i))
          else :
            self.models.append(OneClassSVM(kernel = kernel,max_iter = 3000,gamma = 'scale',nu = 1-self.confidence))
            self.models_names.append(model_name+"_"+kernel)
          
      elif model_name == "LocalOutlierFactor":
        for n_neighbors in [7,31,93,365]:
          self.models.append(LocalOutlierFactor(n_neighbors=n_neighbors,contamination = 1-self.confidence))
          self.models_names.append(model_name+"_"+str(n_neighbors))

      elif model_name == "Lgbm":
        if gridsearch:
          gridsearch_lgbm = GridSearchCV(
                          estimator = LGBMRegressor(loss="mse"),
                          param_grid={'n_estimators': [10,50,100,200,1000,2000],
                                      'learning_rate' : [0.01,0.05,0.1,0.5],
                                      'max_depth' : [3,5,10]
                          },
                          cv=3,
                          n_jobs=-1,
                          verbose=1
                          )
          print("gridsearch xgb best params = ", gridsearch_lgbm.best_params_)
          self.models.append(gridsearch_lgbm.fit(self.X, self.y).best_estimator_)

        else :
          self.models.append(LGBMRegressor())
          self.models_names.append(model_name)

      elif model_name == "Gb":
        if gridsearch:
          gridsearch_gb = GridSearchCV(
                          estimator = GradientBoostingRegressor(loss="mse"),
                          param_grid={'n_estimators': [10,50,100,200,1000,2000],
                                      'learning_rate' : [0.01,0.05,0.1,0.5],
                                      'max_depth' : [3,5,10]
                          },
                          cv=3,
                          n_jobs=-1,
                          verbose=1
                          )
          print("gridsearch xgb best params = ", gridsearch_gb.best_params_)
          self.models.append(gridsearch_gb.fit(self.X, self.y).best_estimator_)
        else:
          self.models.append(GradientBoostingRegressor())
          self.models_names.append(model_name)
      
      else:
        pass
    
    for i in range(len(self.models_names)) :
      print(i)
      print("model _ name = ",self.models_names[i])
      if self.models_names[i] in self.distance_based_models :
        print("GB_up.best_params = ", GB_up.best_params_)
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

class Anomaly_Detection_cuted():
  def __init__(self,models_names = ["IsolationForest","LGBM","GradientBoosting","LocalOutlierFactor","OneClassSVM"],confidence = 0.9,n_predict = 2): #["IsolationForest","LGBM","GradientBoosting","LocalOutlierFactor"]
    self.n_predict = n_predict
    self.confidence = confidence
    self.models_names = models_names
    self.models = []
    self.distance_based_models = ["LGBM","GradientBoosting","QuantileRandomForest"]
    self.alpha_bonferonni = (1- confidence)/len(models_names) #correction de bonferronni
    self.confidence_bonferonni = 1 - self.alpha_bonferonni #correction de bonferronni

    self.dico = {} #dictionnaire contenant les modèles classiques
    
    for model_name in self.models_names: #Initialisation des modèles classiques
      if model_name == "IsolationForest":
        self.dico[model_name] = IsolationForest(n_estimators = 1000, contamination=self.alpha_bonferonni)
        self.models.append(IsolationForest(n_estimators = 1000, contamination=self.alpha_bonferonni))

      elif model_name == "OneClassSVM":
        self.dico[model_name] = OneClassSVM(kernel = "rbf",max_iter = 3000,gamma = 'scale',nu =self.alpha_bonferonni)
        self.models.append(OneClassSVM(kernel = "rbf",max_iter = 3000,gamma = 'scale',nu =self.alpha_bonferonni))
          
      elif model_name == "LocalOutlierFactor":
        self.dico[model_name] = LocalOutlierFactor(n_neighbors=100,contamination = self.alpha_bonferonni)
        self.models.append(LocalOutlierFactor(n_neighbors=100,contamination = self.alpha_bonferonni))


  def fit_predict(self,historique,plot = False,gridsearch = False):  #A tester sur dftrain entier
    
    self.hist = historique.copy(deep = True)
    self.df = historique.copy(deep = True)
    self.AD_bolean_set = []
    labels = list(set(historique["label"]))
    warnings.filterwarnings("ignore")

    for t,label in enumerate(labels):
      self.y = [[self.df["Valeur"][i]] for i in range(self.df["Valeur"].shape[0]) if self.df['label'][i] == label] #Création de la target conditionnée au label
      self.X = self.hist.copy(deep = True)[self.df['label'] == label].drop(["Valeur","DT_VALR","label"],axis = 1) #création des features conditionnées au label
      for i,model in enumerate(self.models_names) :
        print(int(100*t/len(labels) + 100*i/(len(labels)*len(self.models_names))),"%") #Avancement de l'entrainement
        
        if model in self.distance_based_models :
          distancemodel = distance_model(model = model,confidence = self.confidence_bonferonni) #Anomaly detection par methode de reconstruction
          pred = list(distancemodel.fit_predict(self.X,self.y,gridsearch = gridsearch))
          self.AD_bolean_set.append(pred)
          self.df.loc[self.df["label"] == label,model] = pred #Ajout du resultats dans la colonne du model correspondant pour les lignes du label correspondant
        else:
          pred = self.dico[model].fit_predict(self.y) #fit_predict des modèles classiques 
          pred =  np.multiply(self.dico[model].fit_predict(self.y)==-1, 1) # normalisation par convention 0 = normal, 1 = anormal
          self.AD_bolean_set.append(pred)
          self.df.loc[self.df["label"] == label,model] = pred #Ajout du resultats dans la colonne du model correspondant pour les lignes du label correspondant
    self.AD_aggreg_score = np.sum(self.df[self.models_names],axis = 1) #liste contenant le nombre d'algorithme ayant détécté chaque valeur
    self.df["AD_aggreg_score"] = self.AD_aggreg_score
    self.df.loc[self.df["Valeur"]==0,["AD_aggred_score"]] = 0 #suppression des anomalies pour les jours fermés
    self.df.loc[self.df["Valeur"]==1,["AD_aggred_score"]] = 0 #suppression des anomalies pour les jours fermés
    self.df["AD_aggreg_score"] = self.df["AD_aggreg_score"]/len(self.models_names)

    if plot:
      fig1 = px.line(self.df, x = "DT_VALR", y="Valeur")
      fig1.update_traces(line = dict(color = 'rgba(50,50,50,0.2)'))
      fig2 = px.scatter(self.df, x = "DT_VALR", y="Valeur",size = "AD_aggreg_score",color = "AD_aggreg_score")
      fig = go.Figure(data = fig1.data + fig2.data)
      fig.update_layout(title = f" AD aggreg score")    
      #fig.update_traces(mode='lines')                  
      fig.show()
    return self.AD_aggreg_score

# COMMAND ----------

class distance_model:  
  def __init__(self,model,confidence = 0.9):
    """
    model_names = ["IsolationForest","Lgbm","Gb","LocalOutlierFactor","quantile regression forest"]
    """
    self.model = model
    self.confidence = confidence
    self.alpha_down = (1-confidence)/2
    self.alpha_up =  1 - (1-confidence)/2

  def fit_predict(self,X,y,gridsearch = False):
    temp=[]
    for i in range(len(y)):
      temp.append(y[i][0])
    y = temp
    if self.model == "LGBM":  #Recherche des hyperparametres et initialisation des modèles
      if gridsearch :
        mqloss_scorer_up = make_scorer(mqloss, alpha=0.90) #quantile scorer pour le gridsearch
        mqloss_scorer_down = make_scorer(mqloss, alpha=0.05) #quantile scorer pour le gridsearch
        gridsearch_up = GridSearchCV(
                          estimator=LGBMRegressor(alpha = self.alpha_up,objective = "quantile"),
                          param_grid={'n_estimators': [100,200],
                                      'learning_rate' : [0.01,0.05,0.1,0.5]
                          },
                          cv=3,
                          scoring=mqloss_scorer_up,
                          n_jobs=-1,
                          verbose=1
                          )
        model_up = gridsearch_up.fit(X, y).best_estimator_
        print("model_up.best_params = ", gridsearch_up.best_params_)

        gridsearch_down = GridSearchCV(
                          estimator=LGBMRegressor(alpha = self.alpha_down,objective = "quantile"),
                          param_grid={'n_estimators': [100,200],
                                      'learning_rate' : [0.01,0.05,0.1,0.5]
                          },
                          cv=3,
                           scoring=mqloss_scorer_down,
                          n_jobs=-1,
                          verbose=1
                          )
        model_down = gridsearch_down.fit(X, y).best_estimator_
        print("model_down.best_params = ", gridsearch_down.best_params_)

      else:
        model_down = LGBMRegressor(alpha = self.alpha_down,objective='quantile',learning_rate = 0.5,n_estimators = 200)
        model_up = LGBMRegressor(alpha = self.alpha_up,objective = "quantile",learning_rate = 0.5,n_estimators = 200)



    elif self.model == "GradientBoosting":  #Recherche des hyperparametres et initialisation des modèles
      if gridsearch :
        mqloss_scorer_up = make_scorer(mqloss, alpha=0.90)
        mqloss_scorer_down = make_scorer(mqloss, alpha=0.05)
        mqloss_scorer_median = make_scorer(mqloss, alpha=0.5)
        gridsearch_up = GridSearchCV(
                          estimator = GradientBoostingRegressor(loss="quantile", alpha=self.alpha_up,n_estimators = 200,
                                                        learning_rate = 0.01,max_depth = 5),
                          param_grid={'n_estimators': [10,50,100,200,1000,2000],
                                      'learning_rate' : [0.01,0.05,0.1,0.5]
                          },
                          cv=3,
                          scoring=mqloss_scorer_up,
                          n_jobs=-1,
                          verbose=1
                          )
        model_up = gridsearch_up.fit(X, y).best_estimator_
        print("model_up.best_params = ", gridsearch_up.best_params_)

        gridsearch_down = GridSearchCV(
                          estimator = GradientBoostingRegressor(loss="quantile", alpha=self.alpha_down,n_estimators = 200,
                                                        learning_rate = 0.01,max_depth = 5),
                          param_grid={'n_estimators': [100,200],
                                      'learning_rate' : [0.01,0.05,0.1,0.5]
                          },
                          cv=3,
                          scoring=mqloss_scorer_down,
                          n_jobs=-1,
                          verbose=1
                          )
        model_down = gridsearch_down.fit(X, y).best_estimator_
        print("model_down.best_params = ", gridsearch_up.best_params_)

      else:
        model_down = GradientBoostingRegressor(loss="quantile", alpha=self.alpha_down,n_estimators = 200,
                                                        learning_rate = 0.01,max_depth = 5)
        model_up = GradientBoostingRegressor(loss="quantile", alpha=self.alpha_up,n_estimators = 200,
                                                        learning_rate = 0.01,max_depth = 5)
      
      
    else:
      print('Please enter a correct model in LGBM,QuantileRandomForest or GradientBoosting')



    self.pred_up = cross_val_predict(model_up, X, y, cv=10) #cross validation prediction
    self.pred_down = cross_val_predict(model_down, X, y, cv=10) #cross validation prediction
    self.ad_down = (np.array(y) - self.pred_down) < np.quantile(np.array(y) - np.array(self.pred_down),q = self.alpha_down) #booleen alpha quantile hors des prediction quantiles
    self.ad_up = (np.array(y) - self.pred_up) > np.quantile(np.array(y) - np.array(self.pred_up),q = self.alpha_up) #booleen alpha quantile hors des prediction quantiles
    output = self.ad_down + self.ad_up #addition des deux scores
    output[output == 2] = 1 #retransforme output en boleen apres la somme
    return( output )

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

def mqloss(y_true, y_pred, alpha):   #scorer quantile pour le gridsearch
  if (alpha > 0) and (alpha < 1):
    residual = y_true - y_pred 
    return np.mean(residual * (alpha - (residual<0)))
  else:
    return np.nan

# COMMAND ----------

import numpy as np


# COMMAND ----------

np.max([1,2,3])


# COMMAND ----------

class Anomaly_Detection_score():
  """
  Detect anomalies 
  """
  def __init__(self,models_names = ["IsolationForest","Xgb","Lgbm","Gb","OneClassSVM","LocalOutlierFactor"],confidence = 0.9,n_predict = 2):
    self.n_predict = n_predict
    self.confidence = confidence
    self.models_names = []
    self.models = []
    self.distance_based_models = ["Xgb","Lgbm","Gb"]
    

  def fit_predict(self,historique,plot = False,gridsearch = False):  
    """
    fit the model and return a bolean of anomalies
    """
    self.df = historique
    self.y = np.array([[self.df["Valeur"][i]] for i in range(self.df["Valeur"].shape[0])])
    self.X = self.df.drop(["Valeur","DT_VALR"],axis = 1)
    self.anomaly_score_set = []


    for model_name in models_names:
      if model_name == "IsolationForest":
        self.models.append(IsolationForest(n_estimators = 1000, contamination=1-confidence))
        self.models_names.append(model_name)
      elif model_name == "OneClassSVM":
        for kernel in ["linear","poly", "rbf", "sigmoid"]: #precomputed possible
          if kernel == "poly":
            for i in range(3,10):
              self.models.append(OneClassSVM(kernel = kernel,degree = i,max_iter = 3000,gamma = 'scale',nu = 1-self.confidence))
              self.models_names.append(model_name+"_"+kernel+str(i))
          else :
            self.models.append(OneClassSVM(kernel = kernel,max_iter = 3000,gamma = 'scale',nu = 1-self.confidence))
            self.models_names.append(model_name+"_"+kernel)
          
      elif model_name == "LocalOutlierFactor":
        for n_neighbors in [7,31,93,365]:
          self.models.append(LocalOutlierFactor(n_neighbors=n_neighbors,contamination = 1-self.confidence))
          self.models_names.append(model_name+"_"+str(n_neighbors))

      elif model_name == "Lgbm":
        if gridsearch:
          gridsearch_lgbm = GridSearchCV(
                          estimator = LGBMRegressor(loss="mse"),
                          param_grid={'n_estimators': [10,50,100,200,1000,2000],
                                      'learning_rate' : [0.01,0.05,0.1,0.5],
                                      'max_depth' : [3,5,10]
                          },
                          cv=3,
                          n_jobs=-1,
                          verbose=1
                          )
          print("gridsearch xgb best params = ", gridsearch_lgbm.best_params_)
          self.models.append(gridsearch_lgbm.fit(self.X, self.y).best_estimator_)

        else :
          self.models.append(LGBMRegressor())
          self.models_names.append(model_name)

      elif model_name == "Gb":
        if gridsearch:
          gridsearch_gb = GridSearchCV(
                          estimator = GradientBoostingRegressor(loss="mse"),
                          param_grid={'n_estimators': [10,50,100,200,1000,2000],
                                      'learning_rate' : [0.01,0.05,0.1,0.5],
                                      'max_depth' : [3,5,10]
                          },
                          cv=3,
                          n_jobs=-1,
                          verbose=1
                          )
          print("gridsearch xgb best params = ", gridsearch_gb.best_params_)
          self.models.append(gridsearch_gb.fit(self.X, self.y).best_estimator_)
        else:
          self.models.append(GradientBoostingRegressor())
          self.models_names.append(model_name)
      
      else:
        pass
    
    for i in range(len(self.models_names)) :
      print(i)
      print("model _ name = ",self.models_names[i])
      if self.models_names[i] in self.distance_based_models :
        distancemodel_score = distance_model_score(model = model,confidence = self.confidence_bonferonni)
        #y_pred = cross_val_predict(self.models[i], self.X, self.y, cv=3)
        #print("y_pred : ",y_pred)
        #print(np.array(np.abs(y_pred-self.y) > np.quantile(np.abs(y_pred-self.y),confidence)))
        pred = list(distancemodel_score.fit_predict(self.X,self.y,gridsearch = gridsearch))
        self.anomaly_score_set.append(pred)
        self.df[self.models_names[i]] = pred

      else:
        if model == "LocalOutlierFactor":
          score_sample =  self.models[i].fit(self.y).negative_outlier_factor_
        else :
          score_sample =  self.models[i].fit(self.y).score_samples(self.y)
        pred = -(score_sample/np.linalg.norm(score_sample))
        self.anomaly_score_set.append(pred)
        #self.AD_bolean_set.append(self.df.loc[self.df["label"] == label,self.models_names[i]])
        self.df[self.models_names[i]] = pred
    
    self.anomaly_score = np.mean(self.anomaly_score_set,axis = 0)
    self.df["anomaly_score"] = self.anomaly_score

    if plot:
      fig1 = px.line(self.df, x="DT_VALR", y="Valeur")
      fig1.update_traces(line=dict(color = 'rgba(50,50,50,0.2)'))
      fig2 = px.scatter(self.df, x="DT_VALR", y="Valeur",size ="anomaly_score",color ="anomaly_score")
      fig = go.Figure(data=fig1.data + fig2.data)
      fig.update_layout(title = f"anomaly_score")    
      #fig.update_traces(mode='lines')                  
      fig.show()
    return self.AD_aggreg_score
  

# COMMAND ----------

class Anomaly_Detection_Dataset(): 
  def __init__(self,models_names = ["OneClassSVM","LocalOutlierFactor","IsolationForest"],confidence = 0.9,n_predict = 2): #["IsolationForest","LGBM","GradientBoosting","LocalOutlierFactor","OneClassSVM"]
    self.n_predict = n_predict
    self.confidence = confidence
    self.models_names = models_names
    self.models = []
    self.distance_based_models = ["LGBM","GradientBoosting","QuantileRandomForest"]
    self.alpha_bonferonni = (1- confidence)/len(models_names)
    self.confidence_bonferonni = 1 - self.alpha_bonferonni

    self.dico = {}
    
    for model_name in self.models_names:
      if model_name == "IsolationForest":
        self.dico[model_name] = IsolationForest(n_estimators = 1000, contamination=self.alpha_bonferonni)
        self.models.append(IsolationForest(n_estimators = 1000, contamination=self.alpha_bonferonni))

      elif model_name == "OneClassSVM":
        self.dico[model_name] = OneClassSVM(kernel = "rbf",max_iter = 3000,gamma = 'scale',nu =self.alpha_bonferonni)
        self.models.append(OneClassSVM(kernel = "rbf",max_iter = 3000,gamma = 'scale',nu =self.alpha_bonferonni))
          
      elif model_name == "LocalOutlierFactor":
        self.dico[model_name] = LocalOutlierFactor(n_neighbors=100,contamination = self.alpha_bonferonni)
        self.models.append(LocalOutlierFactor(n_neighbors=100,contamination = self.alpha_bonferonni))


    """
      if model_name == "IsolationForest":
        self.models.append(IsolationForest(n_estimators = 1000, contamination=1-confidence))
        self.models_names.append(model_name)

      elif model_name == "OneClassSVM":
        for kernel in ["poly", "rbf", "sigmoid"]: #precomputed possible
          if kernel == "poly":
            for i in range(3,10):
              self.models.append(OneClassSVM(kernel = kernel,degree = i,max_iter = 3000,gamma = 'scale',nu = 1 - self.confidence))
              self.models_names.append(model_name+"_"+kernel+str(i))
          else :
            self.models.append(OneClassSVM(kernel = kernel,max_iter = 3000,gamma = 'scale',nu = 1 - self.confidence))
            self.models_names.append(model_name+"_"+kernel)
          
      elif model_name == "LocalOutlierFactor":
        for n_neighbors in [31,93,365]:
          self.models.append(LocalOutlierFactor(n_neighbors=n_neighbors,contamination = 1-self.confidence))
          self.models_names.append(model_name+"_"+str(n_neighbors))

      elif model_name == "Lgbm":
        self.models.append(LGBMRegressor())
        self.models_names.append(model_name)

      elif model_name == "Gb":
        self.models.append(GradientBoostingRegressor())
        self.models_names.append(model_name)
        """

  def fit_predict(self,historique,plot = False,gridsearch = False,param_grid_gb= None,param_grid_lgbm = None):  #A tester sur dftrain entier
    
    """
    lasso = linear_model.Lasso()
    y_pred = cross_val_predict(lasso, X, y, cv=3)
    """
    self.hist = historique.copy(deep = True)
    self.df = historique.copy(deep = True)
    self.anomaly_score_set = []
    labels = list(set(historique["label"]))
    warnings.filterwarnings("ignore")
    self.score = {}

    for t,label in enumerate(labels):
      
      self.X = self.hist.copy(deep = True)[self.df['label'] == label].drop(["DT_VALR","label"],axis = 1)
      for i,model in enumerate(self.models_names) :
        print(int(100*t/len(labels) + 100*i/(len(labels)*len(self.models_names))),"%")
        print(model)
        #print(i)
        #print("model _ name = ",self.models_names[i])
        
        #print(self.models[i])
        #print(self.models_names[i], np.multiply(self.models[i].fit_predict(self.y)==-1, 1))
        self.dico[model].fit(self.X)
        if model == "LocalOutlierFactor":
          self.score[model] =  -self.dico[model].negative_outlier_factor_
        elif model == "IsolationForest":
          self.score[model] = -self.dico[model].score_samples(self.X)
        else :
          self.score[model] =  self.dico[model].score_samples(self.X)
        pred = (self.score[model]/np.linalg.norm(self.score[model]))
        print("anomaly_score= ",pred)
        self.anomaly_score_set.append(sorted(range(len(pred)), key=lambda k: score[pred]))
        self.df.loc[self.df["label"] == label,model] = pred
        #self.AD_bolean_set.append(self.df.loc[self.df["label"] == label,self.models_names[i]])




    self.AD_mean_score = np.mean(self.df[self.models_names],axis = 1)
    self.AD_score = self.AD_mean_score/np.max(self.AD_mean_score)
    self.df["AD_score"] = self.AD_score
    self.df.loc[self.df["Valeur"]==0,["AD_score"]] = 0
    self.df.loc[self.df["Valeur"]==1,["AD_score"]] = 0

    if plot:
      fig1 = px.line(self.df, x = "DT_VALR", y="Valeur")
      fig1.update_traces(line = dict(color = 'rgba(50,50,50,0.2)'))
      fig2 = px.scatter(self.df, x = "DT_VALR", y="Valeur",size = "AD_score",color = "AD_score")
      fig = go.Figure(data = fig1.data + fig2.data)
      fig.update_layout(title = f"AD_score")    
      #fig.update_traces(mode='lines')                  
      fig.show()
    return self.AD_mean_score
  

# COMMAND ----------

def load_creditcard():
  names = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
  data_loaded = pd.read_csv("/dbfs/FileStore/creditcard.csv",sep = ",",names = names) 
  data_loaded = data_loaded.iloc[1:,:]
  for i in range(len(data_loaded)):
    if i%1000 == 0:
      print(100*i/284808,"%")
    string = data_loaded["Time"][i]
    for name,val in zip(names,string.split(",")):
      data_loaded[name][i] = val
  data_loaded[data_loaded["Class"] == 1] = -1
  data_loaded[data_loaded["Class"] == 0] = 1
  return data_loaded