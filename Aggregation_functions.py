# Databricks notebook source
# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/preprocessing

# COMMAND ----------

dataframe_decpredict

# COMMAND ----------

xX_train, xX_cal, xX_val, xX_test, X_train, X_cal, X_val, X_test, y_train, y_cal, y_val, y_test = train_cal_val_test_split(dataframe_dectrain,0.5,0.33,0.5)


# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
model_name = "Gradient_Boosting"
#Entrainement des modèles de regression quantile pour chacune des bornes de l'intervalle
model_up = GradientBoostingRegressor(loss="quantile", alpha=alphas[1],n_estimators = 1000, learning_rate = 0.01).fit(X_train,y_train)
model_down = GradientBoostingRegressor(loss="quantile", alpha=alphas[0],n_estimators = 1000, learning_rate = 0.01).fit(X_train,y_train)
pred_down_cal = model_down.predict(X_cal)
pred_up_cal = model_up.predict(X_cal)

# COMMAND ----------

pred_set = [pred_down_cal,pred_up_cal]

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
      self.pred_set = self.pred_set/self.normalisation_coef
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


# COMMAND ----------

pred_set = [pred_up_cal,pred_down_cal]

# COMMAND ----------



# COMMAND ----------

ag = aggregation_IC(predict,y_cal)

# COMMAND ----------

ag.boa(1,0.05)

# COMMAND ----------

predict = []
for alpha in np.arange(0.1,1,0.1):
  model = GradientBoostingRegressor(loss="quantile", alpha=alpha,n_estimators = 1000, learning_rate = 0.01).fit(X_train,y_train)
  predict.append(model.predict(X_cal))

# COMMAND ----------



# COMMAND ----------

ag.faboa(0.5)

# COMMAND ----------

fig = go.Figure()
fig.add_trace(go.Scatter(x=xX_cal, y=y_cal,
                    mode='lines',
                    name='True'))
for alpha in np.arange(0.1,1,0.1):
  fig.add_trace(go.Scatter(x=xX_cal, y=np.dot(ag.faboa(alpha),predict),
                      mode='lines',
                      name=f'q_{alpha}'))
fig.show() 

# COMMAND ----------

class aggregation_pred():

  def __init__(self,pred_set,y_true,loss):
    self.loss = loss
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
      self.pred_set = self.pred_set/self.normalisation_coef
      self.normalised = True

  def denormalisation(self):
    if self.normalised == True:
      self.y = self.y*self.normalisation_coef
      self.pred_set = self.pred_set*self.normalisation_coef
      self.normalised = False

  def boa(self,eta,alpha): 
    self.eta_boa = eta
    self.pi_t_j = [np.ones(self.J)/self.J]
    self.l_t_j = []
    self.epi = None
    for t in range(self.len_y):
      self.epi = np.dot(self.pi_t_j[-1],[self.loss(self.y[t],self.pred_set[i][t]) for i in range(self.J)])
      self.l_t_j.append([self.loss(self.y[t],self.pred_set[i][t]) - self.epi for i in range(self.J)])
      self.regularisation = np.sum([np.exp(-self.eta_boa*self.l_t_j[-1][j]*(1 + self.eta_boa*self.l_t_j[-1][j]))*self.pi_t_j[-1][j] for j in range(self.J)])
      self.pi_t_j.append([np.exp(-self.eta_boa*self.l_t_j[-1][j]*(1 + self.eta_boa*self.l_t_j[-1][j]))*self.pi_t_j[-1][j] / self.regularisation for j in range(self.J)])
      return self.pi_t_j[-1]

  def faboa(self): 
    np.seterr(divide='ignore', invalid='ignore')
    self.pi_t_j = [np.ones(self.J)/self.J]
    self.L_t_j = [np.zeros(self.J)]
    self.n_t_j = [np.zeros(self.J)]
    self.E_t_j = 2*np.ones(self.J)
    self.l_t_j = []
    self.epi = None
    for t in range(self.len_y):
      self.epi = np.dot(self.pi_t_j[-1],[self.loss(self.y[t],self.pred_set[i][t]) for i in range(self.J)])
      self.l_t_j.append([self.pinball_loss(self.y[t],self.pred_set[i][t]) - self.epi for i in range(self.J)])

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
  

def AgACI(self,gammas : list, window :int):
  
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


# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
model_name = "Gradient_Boosting"
#Entrainement des modèles de regression quantile pour chacune des bornes de l'intervalle
model = GradientBoostingRegressor(loss="ls",n_estimators = 1000, learning_rate = 0.01)
pred_mse_cal_1 = model.fit(X_train,y_train).predict(X_cal)
pred_mse_cal_2 = model.fit(X_train,y_train).predict(X_cal)

# COMMAND ----------

pred_mse_cal_1 - pred_mse_cal_2

# COMMAND ----------



# COMMAND ----------

