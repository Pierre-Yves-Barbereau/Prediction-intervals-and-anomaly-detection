# Databricks notebook source
def IC(regressor,alphas,df):
  
  target = df["Valeur"]
  df2 = pd.DataFrame([])
  df2["DT_VALR"] = pd.to_datetime(df["DT_VALR"])
  df2["Valeur"] = df["Valeur"]
  df2["Jour_de_la_semaine"] = pd.to_datetime(df['DT_VALR']).dt.weekday
  df2["Jour"] = pd.to_datetime(df['DT_VALR']).dt.day
  df2["Mois"] = pd.to_datetime(df['DT_VALR']).dt.month

  X_train, X_test, y_train, y_test = train_test_split(df2, target,train_size = 0.8,shuffle = False)
  xX_train = X_train["DT_VALR"]
  X_train = X_train.drop(["DT_VALR"],axis = 1)
  X_train = X_train.drop(["Valeur"],axis = 1)
  xX_test = X_test["DT_VALR"]
  X_test = X_test.drop(["DT_VALR"],axis = 1)
  X_test = X_test.drop(["Valeur"],axis = 1)

  df_output = pd.DataFrame([])
  df_output["DT_VALR"] = xX_test
  df_output["Valeur"] = y_test
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                      mode='lines',
                      name='True'))
  for alpha in alphas :
    model = regressor(loss="quantile", alpha=alpha)
    model.fit(X_train,y_train)
    df_output[f"Q_{alpha}"] = model.predict(X_test)
    
    fig.add_trace(go.Scatter(x=df_output["DT_VALR"], y=df_output[f"Q_{alpha}"],
                        mode='lines',
                        name=f'gbr_{alpha}'))
                        
  error = (np.sum(df_output["Valeur"]<df_output["Q_0.05"]) + np.sum(df_output["Valeur"]>df_output["Q_0.95"]))/df_output["Valeur"].shape[0]             
  fig.update_layout(title = f"IC predict {regressor=} confidence = {1-error} ")                      
  fig.show()

# COMMAND ----------



# COMMAND ----------

def original_quantile_grad(y_true,y_pred,alpha):
    x = y_true - y_pred
    grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
    hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta 
    return grad,hess

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

# COMMAND ----------

def pred_quantiles(regressor,alphas,df):
  

  target = df["Valeur"]
  df2 = pd.DataFrame([])
  df2["DT_VALR"] = pd.to_datetime(df["DT_VALR"])
  df2["Valeur"] = df["Valeur"]
  df2["Jour_de_la_semaine"] = pd.to_datetime(df['DT_VALR']).dt.weekday
  df2["Jour"] = pd.to_datetime(df['DT_VALR']).dt.day
  df2["Mois"] = pd.to_datetime(df['DT_VALR']).dt.month

  X_train, X_test, y_train, y_test = train_test_split(df2, target,train_size = 0.8,shuffle = False)
  xX_train = X_train["DT_VALR"]
  X_train = X_train.drop(["DT_VALR"],axis = 1)
  X_train = X_train.drop(["Valeur"],axis = 1)
  xX_test = X_test["DT_VALR"]
  X_test = X_test.drop(["DT_VALR"],axis = 1)
  X_test = X_test.drop(["Valeur"],axis = 1)

  df_output = pd.DataFrame([])
  df_output["DT_VALR"] = xX_test
  df_output["Valeur"] = y_test
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=xX_test, y=y_test,
                      mode='lines',
                      name='True'))
  for alpha in alphas :
    model = regressor(loss="quantile", alpha=alpha)
    model.fit(X_train,y_train)
    df_output[f"Q_{alpha}"] = model.predict(X_test)
    
    fig.add_trace(go.Scatter(x=df_output["DT_VALR"], y=df_output[f"Q_{alpha}"],
                        mode='lines',
                        name=f'gbr_{alpha}'))
                        
  error = (np.sum(df_output["Valeur"]<df_output["Q_0.05"]) + np.sum(df_output["Valeur"]>df_output["Q_0.95"]))/df_output["Valeur"].shape[0]             
  fig.update_layout(title = f"IC predict {regressor=} confidence = {1-error} ")                      
  fig.show()

# COMMAND ----------

def multi_quantiles_loss(y_trues,y_preds,alphas): #crossing version
    grad = []
    for y_true,y_pred,alpha in y_trues,y_preds,alphas:
      x = y_true - y_pred
      grad.append(x<(alpha-1.0))*(1.0-alpha)-((x>=(alpha-1.0))& (x<alpha) )*x-alpha*(x>alpha)
    return grad

# COMMAND ----------

def train_cal_val_test_split(df,train_size,cal_size,val_size):
  target = df["Valeur"]
  X_train, X_test, y_train, y_test = train_test_split(df,target,train_size = train_size,shuffle = False)
  print("first")
  xX_train = X_train["DT_VALR"]
  X_train = X_train.drop(["DT_VALR","Valeur"],axis = 1)
  X_cal, X_test , y_cal, y_test = train_test_split(X_test, y_test,train_size = cal_size,shuffle = False)
  X_val, X_test , y_val, y_test = train_test_split(X_test, y_test,train_size = val_size,shuffle = False)
  print("X_cal")
  print(type(X_cal["Year"]))
  xX_cal = X_cal["DT_VALR"]
  print("xX_cal")
  xX_val = X_val["DT_VALR"]
  print("xX_val")
  xX_test = X_test["DT_VALR"]
  print("xX_test")
  X_cal = X_cal.drop(["DT_VALR","Valeur"], axis = 1)
  X_val = X_val.drop(["DT_VALR","Valeur"], axis = 1)
  X_test = X_test.drop(["DT_VALR","Valeur"],axis = 1)
  return xX_train, xX_cal, xX_val, xX_test, X_train, X_cal, X_val, X_test, y_train, y_cal, y_val, y_test

# COMMAND ----------

def f_conformity_score(pred_down_cal,pred_up_cal,y_cal):
  return np.max([pred_down_cal-y_cal,y_cal-pred_up_cal],axis = 0)

def f_conformity_score_low(pred_down_cal,y_cal):
  return [pred_down_cal-y_cal]

def f_conformity_score_high(pred_up_cal,y_cal):
  return [y_cal-pred_up_cal]

def miscoverage_rate(pred_down_cal,pred_up_cal,pred_down_val,pred_up_val,conformity_score,alpha):
  csq = np.quantile(conformity_score,1-alpha)
  return(np.sum(np.max([pred_down_val-y_val,y_val-pred_up_val],axis = 0)>csq)/len(y_val))

def miscoverage_rate_low(pred_down_val,conformity_score_low,alpha):
  csq = np.quantile(conformity_score_low,1-alpha)
  return(np.sum((pred_down_val-y_val)>csq)/len(y_val))

def miscoverage_rate_high(pred_up_val,conformity_score_high,alpha):
  csq = np.quantile(conformity_score_high,1-alpha)
  return(np.sum((y_val-pred_up_val)>csq)/len(y_val))

# COMMAND ----------

def err_t(y_true,pred_up,pred_down,t):
  return (list(y_true)[-t]>list(pred_up)[-t] or list(y_true)[-t]<list(pred_down)[-t])

def err_t_low(y_true,pred_down,t):
  return list(y_true)[-t]<list(pred_down)[-t]

def err_t_high(y_true,pred_up,t):
  return list(y_true)[-t]>list(pred_up)[-t]

# COMMAND ----------

def pinball_loss(beta,theta,alpha):
  return (alpha*(beta-theta) - min(0,beta-theta))

# COMMAND ----------

def boa(X,y,loss,eta,functions_set,alpha): #not finite decalage t+1
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

class confidence_interval():

  def __init__(self,df_train,alphas, cal_size , val_size ,test_size ,n_predict = 1,mode = "train"):
    self.alphas = alphas
    self.confidence = self.alphas[1] - self.alphas[0]
    self.df_train = df_train
    self.cal_size = cal_size
    self.val_size = val_size
    self.test_size = test_size
    self.n_predict = n_predict
    self.target_train = self.df_train["Valeur"]
    if mode == "train":
      self.train_cal_val_split()
    if mode == "test":
      self.train_cal_val_test_split()



  def train_cal_val_split(self):
    
    self.X_train, self.X_val , self.y_train, self.y_val = train_test_split(self.df_train, self.target_train,test_size = self.val_size,shuffle = False)
    self.X_train, self.X_cal , self.y_train, self.y_cal = train_test_split(self.X_train, self.y_train,test_size = self.cal_size,shuffle = False)
    self.xX_train = self.X_train["DT_VALR"]
    self.X_train = self.X_train.drop(["DT_VALR","Valeur"],axis = 1)
    self.xX_cal = self.X_cal["DT_VALR"]
    self.xX_val = self.X_val["DT_VALR"]
    self.X_cal = self.X_cal.drop(["DT_VALR","Valeur"], axis = 1)
    self.X_val = self.X_val.drop(["DT_VALR","Valeur"], axis = 1)
    
    #return self.xX_train, self.xX_cal, self.xX_val, self.xX_test, self.X_train, self.X_cal, self.X_val, self.X_test, self.y_train, self.y_cal, self.y_val, self.y_test
    self.y_train = list(self.y_train)
    self.X_calval = pd.concat([self.X_cal,self.X_val], sort = False)
    self.y_calval = list(self.y_cal) + list(self.y_val)
    print("train_size = ", self.X_train.shape[0])

  def train_cal_val_test_split(self):
    self.X_train, self.X_test , self.y_train, self.y_test = train_test_split(self.df_train, self.target_train,test_size = self.test_size,shuffle = False)
    self.X_train, self.X_val , self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,test_size = self.val_size,shuffle = False)
    self.X_train, self.X_cal , self.y_train, self.y_cal = train_test_split(self.X_train, self.y_train,test_size = self.cal_size,shuffle = False)
    self.xX_train = self.X_train["DT_VALR"]
    self.X_train = self.X_train.drop(["DT_VALR","Valeur"],axis = 1)
    self.xX_cal = self.X_cal["DT_VALR"]
    self.xX_val = self.X_val["DT_VALR"]
    self.X_cal = self.X_cal.drop(["DT_VALR","Valeur"], axis = 1)
    self.X_val = self.X_val.drop(["DT_VALR","Valeur"], axis = 1)
    self.xX_test = self.X_test["DT_VALR"]
    self.X_test = self.X_test.drop(["DT_VALR","Valeur"], axis = 1)
    
    
    self.y_train = list(self.y_train)
    self.X_calval = pd.concat([self.X_cal,self.X_val], sort = False)
    self.y_calval = list(self.y_cal) + list(self.y_val)
    print("train_size = ", self.X_train.shape[0])
    return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_val,self.X_val, self.y_val, self.xX_test, self.X_test, self.y_test
  
  def predict_aggreg(self):
    self.model_down_set = []
    self.model_up_set = []
    self.pred_down_cal_set = []
    self.pred_up_cal_set = []
    self.pred_down_val_set = []
    self.pred_up_val_set = []
    self.pred_down_calval_set = []
    self.pred_up_calval_set = []

    for _ in range(self.n_predict):
      print("aggregation : ", _+1,"/",self.n_predict)

      #GradientBoosting
      self.model_up_set.append(GradientBoostingRegressor(loss="quantile", alpha=alphas[1],n_estimators = 1000,
                                                        learning_rate = 0.01).fit(self.X_train,self.y_train))
      self.model_down_set.append(GradientBoostingRegressor(loss="quantile", alpha=alphas[0],n_estimators = 1000,
                                                          learning_rate = 0.01).fit(self.X_train,self.y_train))
      self.pred_down_cal_set.append(self.model_down_set[-1].predict(self.X_cal))
      self.pred_up_cal_set.append(self.model_up_set[-1].predict(self.X_cal))
      self.pred_down_val_set.append(self.model_down_set[-1].predict(self.X_val))
      self.pred_up_val_set.append(self.model_up_set[-1].predict(self.X_val))
      self.pred_down_calval_set.append(self.model_down_set[-1].predict(self.X_calval))
      self.pred_up_calval_set.append(self.model_up_set[-1].predict(self.X_calval))

      #LGBM
      self.model_down_set.append(LGBMRegressor(alpha=self.alphas[0],  objective='quantile').fit(self.X_train,self.y_train))
      self.model_up_set.append(LGBMRegressor(alpha=self.alphas[1],  objective='quantile').fit(self.X_train,self.y_train))
      self.pred_down_cal_set.append(self.model_down_set[-1].predict(self.X_cal))
      self.pred_up_cal_set.append(self.model_up_set[-1].predict(self.X_cal))
      self.pred_down_val_set.append(self.model_down_set[-1].predict(self.X_val))
      self.pred_up_val_set.append(self.model_up_set[-1].predict(self.X_val))
      self.pred_down_calval_set.append(self.model_down_set[-1].predict(self.X_calval))
      self.pred_up_calval_set.append(self.model_up_set[-1].predict(self.X_calval))

      
      

    #Aggregation
    self.ag_up = aggregation_IC(self.pred_up_calval_set,self.y_calval)
    self.ag_down = aggregation_IC(self.pred_down_calval_set,self.y_calval)
    self.ag_down.faboa(alphas[0])
    self.ag_up.faboa(alphas[1])

    self.pred_down_cal = np.dot(self.ag_down.pi_t_j[-1],[self.pred_down_cal_set[i] for i in range(len(self.pred_down_cal_set))])
    self.pred_up_cal = np.dot(self.ag_up.pi_t_j[-1],[self.pred_up_cal_set[i] for i in range(len(self.pred_up_cal_set))])
    self.pred_down_val = np.dot(self.ag_down.pi_t_j[-1],[self.pred_down_val_set[i] for i in range(len(self.pred_down_val_set))])
    self.pred_up_val = np.dot(self.ag_up.pi_t_j[-1],[self.pred_up_val_set[i] for i in range(len(self.pred_up_val_set))])

    self.conformity_score = self.f_conformity_score()
    self.conformity_score_low = self.f_conformity_score_low()
    self.conformity_score_high = self.f_conformity_score_high()


  def f_conformity_score(self):
    return np.max([self.pred_down_cal-self.y_cal,self.y_cal-self.pred_up_cal],axis = 0)
  
  def f_conformity_score_low(self):
    return [self.pred_down_cal-self.y_cal]

  def f_conformity_score_high(self):
    return [self.y_cal-self.pred_up_cal]
  
  def miscoverage_rate(self,alpha):
    self.csq = np.quantile(self.conformity_score,1 - alpha)
    return(np.sum(np.max([self.pred_down_val-self.y_val,self.y_val-self.pred_up_val],axis = 0)>self.csq)/len(self.y_val))

  def miscoverage_rate_low(self,alpha):
    self.csq_low = np.quantile(self.conformity_score_low,1-alpha)
    return(np.sum((self.pred_down_val-self.y_val)>self.csq_low)/len(self.y_val))

  def miscoverage_rate_high(self,alpha):
    self.csq_up = np.quantile(self.conformity_score_high,1 - alpha)
    return(np.sum((self.y_val-self.pred_up_val)>self.csq_up)/len(self.y_val))

  def asymetric_conformal_IC(self,X_predict,plot = False):
    self.X_predict = X_predict
    self.predict_aggreg()

    self.pred_down_predict_set = []
    self.pred_up_predict_set = []
    

    for _ in range(len(self.model_down_set)):
      
      self.pred_down_predict_set.append(self.model_down_set[_].predict(self.X_predict))
      self.pred_up_predict_set.append(self.model_up_set[_].predict(self.X_predict))


    self.pred_down_predict = np.dot(self.ag_down.pi_t_j[-1],[self.pred_down_predict_set[i] 
                                                        for i in range(len(self.pred_down_predict_set))])
    self.pred_up_predict = np.dot(self.ag_up.pi_t_j[-1],[self.pred_up_predict_set[i] 
                                                         for i in range(len(self.pred_up_predict_set))])


    self.betas = np.arange(0,1,0.0001)
    self.alpha_star_low = np.max([b for b in np.arange(0,1,0.0001) if (self.miscoverage_rate_low(alpha = b) < self.alphas[0])])
    self.alpha_star_high = np.max([b for b in np.arange(0,1,0.0001) if (self.miscoverage_rate_high(alpha = b) < 1 - self.alphas[1])])
    self.alpha_star_low_updated = self.alpha_star_low
    self.alpha_star_high_updated = self.alpha_star_high
    self.pred_down_predict_asymetric_conformal = self.pred_down_predict - np.quantile(self.conformity_score_low,1-self.alpha_star_low)
    self.pred_up_predict_asymetric_conformal = self.pred_up_predict + np.quantile(self.conformity_score_high,1-self.alpha_star_high)
    
  
    if plot :
      fig = go.Figure()
      fig.add_trace(go.Scatter( y=self.pred_up_predict_asymetric_conformal,
                      mode='lines',
                      name=f'q_{alphas[1]}',
                      line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                      showlegend = False))

      fig.add_trace(go.Scatter( y=self.pred_down_predict_asymetric_conformal,
                      mode='lines',
                      name=f'{int(np.round(100*(self.alphas[1]-self.alphas[0])))}% Confidence Interval',
                      line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                      fill='tonexty',
                      fillcolor='rgba(0,176,246,0.2)',
                      line_color='rgba(255,255,255,0)'))

      fig.update_traces(mode='lines')
      fig.update_layout(title = f"{int(np.round(100*(self.alphas[1]-self.alphas[0])))}% Confidence Interval Prediction")                      
      fig.show()
    
    return self.pred_down_predict_asymetric_conformal,self.pred_up_predict_asymetric_conformal
  

  def FACI(self, gammas : list):
    self.gammas_FACI = gammas
  
  def plot_by_flux(self,fluxs):
    fig = make_subplots(rows=len(fluxs), cols=1)

    for flux in fluxs :
      preproc = preprocessing(NAME_ADB_BASE_CFM_IC = NAME_ADB_BASE_CFM_IC,
                            NAME_ADB_VIEW_FLUX_HISTORIQUE = NAME_ADB_VIEW_FLUX_HISTORIQUE,
                            NAME_ADB_VIEW_FLUX_FIN_TRAN = NAME_ADB_VIEW_FLUX_FIN_TRAN,
                            path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing,
                            path_dataframe_train = path_dataframe_train,
                            path_dataframe_predict = path_dataframe_predict,
                            groupe = flux
                            )

    dataframe_dectrain, dataframe_decpredict = preproc.load_train_predict()
    ci_test = confidence_interval(df_train = dataframe_dectrain, alphas = alphas)
    ci_test.train_cal_val_test_split()
    xX_test = ci_test.xX_test
    X_test = ci_test.X_test
    y_test = ci_test.y_test
    asymetric_conformal_ic_test = ci_test.asymetric_conformal_IC(X_test)


    fig.append_trace(go.Scatter( y=asymetric_conformal_ic_test[0],
                          mode='lines',
                          name=f'q_{alphas[0]}'))
    fig.add_trace(go.Scatter( y=asymetric_conformal_ic_test[1],
                          mode='lines',
                          name=f'q_{alphas[1]}'))


    fig.add_trace(go.Scatter(
        y=asymetric_conformal_ic_test[0],
        fill='tonexty',
        fillcolor='rgba(0,176,246,0.2)',
        line_color='rgba(255,255,255,0)',
        name='Premium',
        showlegend=False,
    ))

    fig.update_traces(mode='lines')
    error = error = (np.sum(ci_test.y_test<asymetric_conformal_ic_test[0]) + np.sum(ci_test.y_test>asymetric_conformal_ic_test[1]))/len(ci_test.y_test) 
    fig.update_layout(title = f"{flux} Conformal IC test, confidence = {1 - error}, expected = {alphas[1]-alphas[0]}")                    
    fig.show()

  

# COMMAND ----------



# COMMAND ----------

class aggregation_IC():

  def __init__(self,pred_set,y_true):
    self.normalised = False
    self.coef_factor = np.max(y_true)
    self.pred_set = pred_set
    self.y = list(y_true)
    self.J = len(pred_set)
    self.len_y = len(y_true)
    self.normalisation()

  def normalisation(self):
    if self.normalised == False:
      self.normalisation_coef = np.max(self.y)
      self.y = self.y/self.normalisation_coef
      self.pred_set = [pred/self.normalisation_coef for pred in self.pred_set]
      self.normalised = True

  def denormalisation(self):
    if self.normalised == True:
      self.y = self.y*self.normalisation_coef
      self.pred_set = self.pred_set*self.normalisation_coef
      self.normalised = False

  def pinball_loss(self,true,pred,alpha):
    return (alpha*(true-pred)*(true-pred >= 0) + (alpha-1)*(true-pred)*(true-pred < 0))

  def boa(self,eta,alpha): 
    self.eta_boa = eta
    self.pi_t_j = [np.ones(self.J)/self.J]
    self.l_t_j = []
    self.epi = None
    for t in range(self.len_y):
      self.epi = np.dot(self.pi_t_j[-1],[self.pinball_loss(self.y[t],self.pred_set[i][t],alpha = alpha) for i in range(self.J)])
      self.l_t_j.append([self.pinball_loss(self.y[t],self.pred_set[i][t],alpha = alpha) - self.epi for i in range(self.J)])
      self.regularisation = np.sum([np.exp(-self.eta_boa*self.l_t_j[-1][j]*(1 + self.eta_boa*self.l_t_j[-1][j]))*self.pi_t_j[-1][j] for j in range(self.J)])
      self.pi_t_j.append([np.exp(-self.eta_boa*self.l_t_j[-1][j]*(1 + self.eta_boa*self.l_t_j[-1][j]))*self.pi_t_j[-1][j] / self.regularisation for j in range(self.J)])
      return self.pi_t_j[-1]

  def faboa(self,alpha): 
    np.seterr(divide='ignore', invalid='ignore')
    self.pi_t_j = [np.ones(self.J)/self.J]
    self.L_t_j = [np.zeros(self.J)]
    self.n_t_j = [np.zeros(self.J)]
    self.E_t_j = 2*np.ones(self.J)
    self.l_t_j = []
    self.epi = None
    for t in range(self.len_y):
      self.epi = np.dot(self.pi_t_j[-1],[self.pinball_loss(self.y[t],self.pred_set[i][t],alpha = alpha) for i in range(self.J)])
      self.l_t_j.append([self.pinball_loss(self.y[t],self.pred_set[i][t],alpha = alpha) - self.epi for i in range(self.J)])

      self.L_t_j.append([self.L_t_j[-1][j] + self.l_t_j[-1][j]*(1 + self.n_t_j[-1][j]*self.l_t_j[-1][j])/2 + self.E_t_j[j]*(self.n_t_j[-1][j]*self.l_t_j[-1][j]>0.5) for j in range(self.J)])

      for j in range(self.J):
        if self.l_t_j[-1][j] > self.E_t_j[j]:
          k=0
          while self.l_t_j[-1][j] <= 2**k:
            k=k+1
          self.E_t_j[j] = 2**(k+1)

      self.n_t_j.append([np.min([1/self.E_t_j[j], math.sqrt(math.log(1/self.pi_t_j[0][j])/np.sum([self.l_t_j[s][j]**2 for s in range(t)]))]) for j in range(self.J)])
      
      self.regularisation = np.sum([np.exp(-self.n_t_j[-1][j]*self.l_t_j[-1][j]*(1 + self.n_t_j[-1][j]*self.l_t_j[-1][j]))*self.pi_t_j[-1][j] for j in range(self.J)])
      self.pi_t_j.append([np.exp(-self.n_t_j[-1][j]*self.l_t_j[-1][j]*(1 + self.n_t_j[-1][j]*self.l_t_j[-1][j]))*self.pi_t_j[-1][j] / self.regularisation for j in range(self.J)])

    return self.pi_t_j[-1]
  

def AgACI(self, gammas : list, window = 20):
  
  for t in y_test_2:
    low_t = []
    high_t = []
    omega_t_low = []
    omega_t_high = []
    low = pred_down_test_2 - np.quantile(f_conformity_score_low(pred_down_test_1,y_test_1),1-alpha_star_low_updated)
    high = pred_up_test_2 + np.quantile(f_conformity_score_high(pred_up_test_1,y_test_1),1-alpha_star_high_updated)
    for gamma in gammas:

      alpha_star_low_upd = alpha_star_low_updated + gamma*(alphas[0] - np.mean([err_t_low(y_test_1,pred_down_test_1,i) for i in range(window)]))
      if 0<alpha_star_low_upd <1:
        low = pred_down_test_2 - np.quantile(f_conformity_score_low(pred_down_test_1,y_test_1),1-alpha_star_low_upd)
      else :
        print("alpha_star_low_updated out of 0-1")

      alpha_star_high_upd = alpha_star_high_updated + gamma*(1-alphas[1] - np.mean([err_t_high(y_test_1,pred_up_test_1,i) for i in range(window)]))
      if 0<alpha_star_high_upd <1:
        high = pred_up_test_2 + np.quantile(f_conformity_score_high(pred_up_test_1,y_test_1),1-alpha_star_high_upd)
      else :
        print("alpha_star_high_updated out of 0-1")

      low_t.append(low)
      high_t.append(high)


    C_low = np.mean(low_t,axis = 0)
    C_high = np.mean(high_t,axis = 0)
  return C_low,C_high
