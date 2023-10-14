# Databricks notebook source
class Prediction_intervals():
    def __init__(self,alphas, cal_size,groupe,models = ["LGBM","GB","QRF"], test_size = None,val_size = None,mode = "normal",dico_hyperparametres = None,gridsearch = False, param_grid_gb=None, param_grid_lgbm=None ):
        """
        alphas = [quantile_bottom,quantile_up]
        mode = ("normal" ou "test")
        """
        self.models = models
        self.gridsearch = gridsearch
        self.param_grid_gb = param_grid_gb
        self.param_grid_lgbm = param_grid_lgbm
        self.alphas = alphas
        self.confidence = self.alphas[1] - self.alphas[0]
        self.cal_size = cal_size
        self.test_size = test_size
        self.fitted = False
        self.mode = mode
        self.val_size = val_size
        self.adaptative = False
        self.groupe = groupe
        #Initialisation des modèles
        if dico_hyperparametres == None :
            self.dico_hyperparametres = {}
            self.dico_hyperparametres[groupe] = {}
            self.model = IC_model(alphas = self.alphas,groupe= self.groupe,models = self.models,gridsearch = True)
        else : 
            self.dico_hyperparametres = dico_hyperparametres
            self.model = IC_model(alphas = self.alphas,groupe= self.groupe,models = self.models, dico_hyperparametres = self.dico_hyperparametres,param_grid_gb = param_grid_gb,param_grid_lgbm = param_grid_lgbm,gridsearch = self.gridsearch)


    def split_conformal_ic(self,df_train,df_predict = None,plot = True):
        """
        Compute Asymetric Conformal Inference for predict set,
        X_predict = None,plot = False
        return self.pred_down_predict_asymetric_conformal,self.pred_up_predict_asymetric_conformal
        """

        self.target_train = df_train["Valeur"]
        self.df_train = df_train

        # On crée les sets de calibration et de validation
        if self.mode == "normal":
            self.train_cal_split()

        # En mode test on crée les sets de calibration de validation et de test
        if self.mode == "test":
            self.train_cal_test_split()

        #On fit chacun des modèles
        self.model.fit(self.X_train,self.y_train)
        #if self.gridsearch == True:
          #np.save(f'/dbfs/FileStore/IC_hyperparametres.npy',self.model.dico_hyperparametres) 

        #On effectue les prediction sur les sets de calibration et de validation
        self.predict_cal()

        self.X_predict = df_predict
        if self.mode == "test":
            self.X_predict = self.X_test
        #Prediction du set de test
        self.pred_down_predict,self.pred_up_predict,self.pred_median_predict = self.predict_test(self.X_predict)
        self.pred_down = np.empty(0)
        self.pred_up = np.empty(0)
        self.pred_median = np.empty(0)

        for i in range(len(self.X_predict)):
            try:
                self.pred_down = np.append(self.pred_down,self.pred_down_predict[i] - np.quantile(self.f_conformity_score_down(self.pred_down_cal[self.X_cal["label"] == list(self.X_predict["label"])[i]], self.y_cal[self.X_cal["label"] == list(self.X_predict["label"])[i]]),1 - self.alphas[0]))
            except Exception as e:
                print(e)
                self.pred_down = np.append(self.pred_down,self.pred_down_predict[i])
            try:
                self.pred_up = np.append(self.pred_up,self.pred_up_predict[i] + np.quantile(self.f_conformity_score_up(self.pred_up_cal[self.X_cal["label"] == list(self.X_predict["label"])[i]],self.y_cal[self.X_cal["label"] == list(self.X_predict["label"])[i]]),self.alphas[1]))
            except Exception as e:
                print(e)
                self.pred_up = np.append(self.pred_up,self.pred_up_predict[i])
            try:
                self.pred_median = np.append(self.pred_median,self.pred_median_predict[i] + np.quantile(self.f_conformity_score_up(self.pred_median_cal[self.X_cal["label"] == list(self.X_predict["label"])[i]],self.y_cal[self.X_cal["label"] == list(self.X_predict["label"])[i]]),0.5))
            except Exception as e:
                print(e)
                self.pred_median = np.append(self.pred_median,self.pred_median_predict[i])

        if plot:
            self.plot()
        self.X_predict[f"{np.round(self.alphas[0],decimals = 3)}_quantile"] = self.pred_down
        self.X_predict[f"{np.round(self.alphas[1],decimals = 3)}_quantile"] = self.pred_up
        self.X_predict[f"0.5_quantile"] = self.pred_median
        return(self.X_predict)


    def adaptative_ci(self,X_predict = None,plot = False,gamma = 0.1,adaptative_window_length = 50):
        self.gamma = gamma
        self.adaptative_window_length = adaptative_window_length
        self.target_train = df_train["Valeur"]
        self.df_train = df_train

        # On crée les sets de calibration et de validation
        if self.mode == "normal" or self.adaptative:
            self.train_cal_val_split()

        # En mode test on crée les sets de calibration de validation et de test
        if self.mode == "test" and not self.adaptative:
            self.train_cal_val_test_split()

        #On fit chacun des modèles
        self.model.fit(self.X_train,self.y_train)
        #if self.gridsearch == True:
          #np.save(f'/dbfs/FileStore/IC_hyperparametres.npy',self.model.dico_hyperparametres) 

        #On effectue les prediction sur les sets de calibration et de validation
        self.predict_cal_val()
        print("len(self.X_cal) = ",len(self.X_cal))
        print("len(self.y_cal) = ",len(self.y_cal))
        print("conformal fitted")
        self.fitted = True
        self.X_predict = X_predict
        if self.mode == "test":
            self.X_predict = self.X_test
        self.pred_down_predict,self.pred_up_predict,self.pred_median_predict = self.predict_test(self.X_predict)
        #initialize adaptative alphas
        alpha_star_down_updated = self.alphas[0]
        alpha_star_up_updated = self.alphas[1]
        alpha_star_median_updated = 0.5
        pred_down_adaptative = copy.deepcopy(self.pred_down_cal[-len(self.X_predict):])
        pred_up_adaptative = copy.deepcopy(self.pred_up_cal[-len(self.X_predict):])
        pred_median_adaptative = copy.deepcopy(self.pred_median_cal[-len(self.X_predict):])
        y_cal_increased = copy.deepcopy(self.y_cal[-len(self.X_predict):])
        y_val = self.y_val.copy()
        #pred_up_val = self.pred_up_val.copy()
        #pred_down_val = self.pred_down_val.copy()
        pred_median_cal = copy.deepcopy(self.pred_median_cal)
        pred_median_val = copy.deepcopy(self.pred_median_val)

        self.alpha_star_down_updated_list = []
        self.alpha_star_up_updated_list = []
        self.alpha_star_median_updated_list = []
  
        #init alpha star
        try:
            alpha_star_down_updated = np.max([b for b in np.arange(0,1,0.0001) if (self.f_miscoverage_rate_down(self.pred_down_cal,self.pred_down_val,self.y_cal,self.y_val,alpha = b) < self.alphas[0])])
            alpha_star_up_updated = np.max([b for b in np.arange(0,1,0.0001) if (self.f_miscoverage_rate_up(self.pred_up_cal,self.pred_up_cal,self.y_cal,self.y_val,alpha = b) < 1 - self.alphas[1])])
            alpha_star_median_updated = np.max([b for b in np.arange(0,1,0.0001) if (self.f_miscoverage_rate_median(self.pred_median_cal,self.pred_median_val,self.y_cal,self.y_val,alpha = b) < 0.5)])
        except ValueError:
            print("no quantile found to cover the shift")
            print("setting default value")
            alpha_star_down_updated = 0.05
            alpha_star_up_updated = 0.95
            alpha_star_median_updated = 0.5

        for it in range(len(self.X_val)):
            #compute alpha star updated
            alpha_star_up_updated = alpha_star_up_updated + self.gamma*(1 - self.alphas[1] - np.sum([self.err_t_up(y_cal_increased,pred_up_adaptative,w)*math.exp(-w) for w in range(self.adaptative_window_length)])/np.sum([math.exp(k) for k in range(self.adaptative_window_length)]))
            alpha_star_down_updated = alpha_star_down_updated + self.gamma*(self.alphas[0] - np.sum([self.err_t_down(y_cal_increased,pred_down_adaptative,w)*math.exp(-w) for w in range(self.adaptative_window_length)])/np.sum([math.exp(k) for k in range(self.adaptative_window_length)]))
            alpha_star_median_updated = alpha_star_median_updated + self.gamma*(0.5 - np.sum([self.err_t_median(y_cal_increased,pred_median_adaptative,w)*math.exp(-w) for w in range(self.adaptative_window_length)])/np.sum([math.exp(k) for k in range(self.adaptative_window_length)]))
            self.alpha_star_down_updated_list.append(alpha_star_down_updated)
            self.alpha_star_up_updated_list.append(alpha_star_up_updated)
            self.alpha_star_median_updated_list.append(alpha_star_median_updated)

            if 0<alpha_star_up_updated <1:
                pred_up_adaptative = np.append(pred_up_adaptative,self.pred_up_val[it] + np.quantile(self.f_conformity_score_up(self.y_cal,self.pred_up_cal),1-alpha_star_up_updated))
            elif alpha_star_up_updated <= 0:
                pred_up_adaptative = np.append(pred_up_adaptative,self.pred_up_val[it] + np.min(self.f_conformity_score_up(self.y_cal,self.pred_up_cal)))
                alpha_star_up_updated = 0.001
                print(f" i = {it} alpha_star_up_updated < 0")
            elif alpha_star_up_updated >=1:
                pred_up_adaptative = np.append(pred_up_adaptative,self.pred_up_val[it] + np.max(self.f_conformity_score_up(self.y_cal,self.pred_up_cal)))
                alpha_star_up_updated = 0.999
                print(f" i = {it} alpha_star_up_updated >1")
                
          #lower bound 
            if 0<alpha_star_down_updated <1:
                pred_down_adaptative = np.append(pred_down_adaptative,self.pred_down_val[it] - np.quantile(self.f_conformity_score_down(self.y_cal,self.pred_up_cal),1-alpha_star_down_updated))
            elif alpha_star_down_updated <= 0:
                pred_down_adaptative = np.append(pred_down_adaptative,self.pred_down_val[it] - np.max(self.f_conformity_score_down(self.y_cal,self.pred_up_cal)))
                alpha_star_down_updated = 0.001
                print(f" i = {it} alpha_star_down_updated < 0")
            elif alpha_star_down_updated >=1:
                pred_down_adaptative = np.append(pred_down_adaptative,self.pred_down_val[it] - np.min(self.f_conformity_score_down(self.y_cal,self.pred_up_cal)))
                alpha_star_down_updated = 0.999
                print(f" i = {it} alpha_star_down_updated >1")
                
          #Median 
            if 0<alpha_star_median_updated <1:
                pred_median_adaptative = np.append(pred_median_adaptative,self.pred_median_val[it] - np.quantile(self.f_conformity_score_median(self.y_cal,self.pred_up_cal),1-alpha_star_median_updated))
            elif alpha_star_median_updated <= 0:
                pred_median_adaptative = np.append(pred_median_adaptative,self.pred_median_val[it] - np.max(self.f_conformity_score_median(self.y_cal,self.pred_up_cal)))
                alpha_star_median_updated = 0.1
                print(f" i = {it} alpha_star_down_updated < 0")
            elif alpha_star_median_updated >=1:
                pred_median_adaptative = np.append(pred_median_adaptative,self.pred_median_val[it] - np.min(self.f_conformity_score_median(self.y_cal,self.pred_up_cal)))
                alpha_star_median_updated = 0.9
                print(f" i = {it} alpha_star_down_updated >1")

            y_cal_increased = np.append(y_cal_increased,self.y_val[it])
            y_cal_increased = np.delete(y_cal_increased,0)
            pred_up_adaptative = np.delete(pred_up_adaptative,0)
            pred_down_adaptative = np.delete(pred_down_adaptative,0)
            pred_median_adaptative = np.delete(pred_median_adaptative,0)
            print(it , "alpha_star_down_updated : ",alpha_star_down_updated)
            print(it , "alpha_star_up_updated : ",alpha_star_up_updated)
            print(it , "alpha_star_median_updated : ",alpha_star_median_updated)

        if self.mode =="normal":
            pred_down_adaptative = self.pred_down_predict - np.quantile(self.f_conformity_score_down(self.y_cal,self.pred_up_cal), 1-alpha_star_down_updated)
            pred_up_adaptative = self.pred_up_predict + np.quantile(self.f_conformity_score_up(self.y_cal,self.pred_up_cal),1-alpha_star_up_updated)
            pred_median_adaptative = self.pred_median_predict - np.quantile(self.f_conformity_score_median(self.y_cal,self.pred_up_cal),1-alpha_star_median_updated)
        
        self.pred_up = pred_up_adaptative
        self.pred_down = pred_down_adaptative
        self.pred_median = pred_median_adaptative
        if plot:
            self.plot()
        return(self.pred_down,self.pred_up,self.pred_median)
    
    def FACI(self,alphas,sigma, eta,gammas : list,alphas_i : list,X_predict = None,plot = True):
        self.alphas = alphas
        """Perform a fully adaptative ACI"""
        self.alphas_i = alphas_i
        self.target_train = df_train["Valeur"]
        self.df_train = df_train

        # On crée les sets de calibration et de validation
        if self.mode == "normal":
            self.train_cal_val_split()

        # En mode test on crée les sets de calibration de validation et de test
        if self.mode == "test":
            self.train_cal_val_test_split()

        #On fit chacun des modèles
        self.model.fit(self.X_train,self.y_train)

        #On effectue les prediction sur les sets de calibration et de validation
        self.predict_cal_val()
        print("len(self.X_cal) = ",len(self.X_cal))
        print("len(self.y_cal) = ",len(self.y_cal))
        print("conformal fitted")
        if self.mode == "test":
            self.X_predict = self.X_test
        self.pred_down_predict,self.pred_up_predict,self.pred_median_predict = self.predict_test(self.X_predict)
        
        ## FACI UP
        p_values = [np.max([b for b in np.arange(0,1,0.001) if (self.pred_up_val[t] + np.quantile(self.f_conformity_score_up(self.pred_up_cal,self.y_cal),1-b))>=list(self.y_val)[t] ]) for t in range(len(self.pred_up_val))]
        k = len(p_values)
        omegas = np.ones(len(gammas))
        pred_up = []
        for t in range(k):
            p = omegas / np.sum(omegas)
            alpha_barre = np.dot(p,alphas_i)
            pred_up.append(alpha_barre)
            omegas_barre = []
            for i in range(len(alphas_i)):
                omegas_barre.append(omegas[i]*np.exp(-eta*(self.alphas[1]*(p_values[t]-alphas_i[i]) - min(0,p_values[t]-alphas_i[i]))))
            omegas = (np.multiply((1-sigma),omegas_barre) + np.mean(omegas_barre)*sigma)
            err_t_i = [(self.pred_up_cal[t] + np.quantile(self.f_conformity_score_up(self.pred_up_cal,self.y_cal),1-alpha_i))>list(self.y_cal)[t] for alpha_i in alphas_i]
            for i in range(len(alphas_i)):
                alphas_i[i] = max(0,min(1,alphas_i[i] + gammas[i]*(self.alphas[1] - err_t_i[i])))
        self.pred_up = self.pred_up_predict + np.quantile(self.f_conformity_score_up(self.pred_up_cal,self.y_cal),alpha_barre)
        
        ## FACI Down
        alphas_i = self.alphas_i
        p_values = [np.max([b for b in np.arange(0,1,0.001) if (self.pred_down_val[t] - np.quantile(self.f_conformity_score_down(self.pred_down_cal,self.y_cal),1-b))<=list(self.y_val)[t] ]) for t in range(len(self.pred_up_val))]
        k = len(p_values)
        omegas = np.ones(len(gammas))
        pred_down = []
        for t in range(k):
            p = omegas / np.sum(omegas)
            alpha_barre = np.dot(p,alphas_i)
            pred_down.append(alpha_barre)
            omegas_barre = []
            for i in range(len(alphas_i)):
                omegas_barre.append(omegas[i]*np.exp(-eta*(self.alphas[0]*(p_values[t]-alphas_i[i]) - min(0,p_values[t]-alphas_i[i]))))
            omegas = (np.multiply((1-sigma),omegas_barre) + np.mean(omegas_barre)*sigma)
            err_t_i = [(self.pred_up_cal[t] - np.quantile(self.f_conformity_score_down(self.pred_down_cal,self.y_cal),1-alpha_i))>list(self.y_cal)[t] for alpha_i in alphas_i]
            for i in range(len(alphas_i)):
                alphas_i[i] = max(0,min(1,alphas_i[i] + gammas[i]*(self.alphas[0] - err_t_i[i])))
        
        self.pred_down = self.pred_down_predict - np.quantile(self.f_conformity_score_down(self.pred_down_cal,self.y_cal),1-alpha_barre)
        if plot:
            self.plot()
        return(self.pred_up,self.pred_down)
    
    
    def AgACI(self, gammas : list, window = 20): #Aggregative ACI
        """perform an aggregation of multiples ACI"""
        self.alphas = alphas
        """Perform a fully adaptative ACI"""
        self.alphas_i = alphas_i
        self.target_train = df_train["Valeur"]
        self.df_train = df_train

        # On crée les sets de calibration et de validation
        if self.mode == "normal":
            self.train_cal_val_split()

        # En mode test on crée les sets de calibration de validation et de test
        if self.mode == "test":
            self.train_cal_val_test_split()

        #On fit chacun des modèles
        self.model.fit(self.X_train,self.y_train)

        #On effectue les prediction sur les sets de calibration et de validation
        self.predict_cal_val()
        print("len(self.X_cal) = ",len(self.X_cal))
        print("len(self.y_cal) = ",len(self.y_cal))
        print("conformal fitted")
        if self.mode == "test":
            self.X_predict = self.X_test
        self.pred_down_predict,self.pred_up_predict,self.pred_median_predict = self.predict_test(self.X_predict)
        
        for t in y_test_2:
            low_t = []
            high_t = []
            omega_t_low = []
            omega_t_high = []
            low = pred_down_test_2 - np.quantile(f_conformity_score_low(pred_down_test_1,y_test_1),1-alpha_star_low_updated)
            high = pred_up_test_2 + np.quantile(f_conformity_score_high(pred_up_test_1,y_test_1),alpha_star_high_updated)
            for gamma in gammas:
                alpha_star_low_upd = alpha_star_low_updated + gamma*(alphas[0] - np.mean([err_t_low(y_test_1,pred_down_test_1,i) for i in range(window)]))
                if 0<alpha_star_low_upd <1:
                    low = pred_down_test_2 - np.quantile(f_conformity_score_low(pred_down_test_1,y_test_1),1-alpha_star_low_upd)
                else :
                    print("alpha_star_low_updated out of 0-1")
                    alpha_star_high_upd = alpha_star_high_updated + gamma*(1-alphas[1] - np.mean([err_t_high(y_test_1,pred_up_test_1,i) for i in range(window)]))
                if 0<alpha_star_high_upd <1:
                    high = pred_up_test_2 + np.quantile(f_conformity_score_high(pred_up_test_1,y_test_1),alpha_star_high_upd)
                else :
                    print("alpha_star_high_updated out of 0-1")
                low_t.append(low)
                high_t.append(high)
                C_low = np.mean(low_t,axis = 0)
                C_high = np.mean(high_t,axis = 0)
            return C_low,C_high
        
        
    def enbpi(self,model,df_train,alphas,df_predict=None,n_bootstrap = 100,bach_size = 1):
        """
        Perform a bootstrap method to construct an interval for prediction model
        model = regression model
        """
        # En mode test on crée les sets de calibration de validation et de test
        if self.mode == "test":
            self.df_train,self.df_predict = train_test_split(df_train,test_size = 0.2)
        self.n_bootstrap = n_bootstrap
        self.batch_size = batch_size
        self.confidence = alphas[1]-alphas[0]
        self.model = model
        self.X_train=self.df_train.drop(["Valeur","DT_VALR"],axis = 1)
        self.y_train = self.df_train["Valeur"]
        self.T = len(self.df_train)
        self.len_train = len(self.df_train)
        self.S_b = []
        self.f_b = []
        
        #On effectue les prédictions bootstrap
        for b in range(self.n_bootstrap):
            print("1/3 Train 1/2 : ",int(100*b/self.n_bootstrap),"%")
            self.S_b.append(np.random.choice(list(self.X_train.index),self.T,replace = True))
            self.f_b.append(self.model.fit(self.X_train.loc[self.S_b[-1]],np.take(self.y_train, self.S_b[-1])))
        self.eps =[]
        self.f_phi_i = []
        
        for i in range(self.len_train):
            print("2/3 train 2/2 : ",int(100*i/self.len_train),"%")
            self.f_phi_temp = []
            for bi,b in enumerate(self.S_b):
                if i not in b:
                      self.f_phi_temp.append(self.f_b[bi].predict(self.X_train.iloc[i:i+1]))
            self.f_phi_i.append(np.mean(self.f_phi_temp))
            self.eps_phi_i = np.abs(self.y_train[i]-self.f_phi_i[i])
            if math.isnan(self.eps_phi_i) == False:
                self.eps.append(self.eps_phi_i)
        self.X_test = df_predict.drop(["DT_VALR"],axis = 1)
        self.len_test = len(self.X_test)
        self.C_t_low = []
        self.C_t_high = []
        self.f_phi_t = [] 
        for t in range(len(self.X_test)):
            print(t,"/",self.len_test)
            print(int(100*t/self.len_test),"%")
            self.f_phi_temp = []
            for bi,b in enumerate(self.S_b):
                if t not in b:
                    self.f_phi_temp.append(self.f_b[bi].predict(self.X_test.iloc[t:t+1]))
            self.f_phi_t.append(np.quantile(self.f_phi_temp,self.confidence))
            self.w_phi_t = np.quantile(self.eps,self.confidence)
            self.C_t_low.append(self.f_phi_t - self.w_phi_t)
            self.C_t_high.append(self.f_phi_t + self.w_phi_t)
          
        if t % self.batch_size == 0:
            for j in range(self.batch_size):
                self.e_phi_j = float(np.abs(self.y_test[j]-self.f_phi_t[j]))
                self.eps.append(self.e_phi_j)
                del self.eps[0]
        
        return self.C_t_low, self.C_t_high

    def train_cal_split(self):
        if self.cal_size < 1:
            cal_size = int(self.cal_size*len(self.df_train))
        else : 
            cal_size = int(self.cal_size)
        "split data into train, calibration, used in normal mode"
        self.X_train, self.X_cal , self.y_train, self.y_cal = train_test_split(self.df_train, self.target_train,test_size = cal_size,shuffle = False)
        self.xX_train = self.X_train["DT_VALR"]
        self.X_train = self.X_train.drop(["DT_VALR","Valeur"],axis = 1)
        self.xX_cal = self.X_cal["DT_VALR"]
        self.X_cal = self.X_cal.drop(["DT_VALR","Valeur"], axis = 1)
        #return self.xX_train, self.xX_cal, self.xX_val, self.xX_test, self.X_train, self.X_cal, self.X_val, self.X_test, self.y_train, self.y_cal, self.y_val, self.y_test
        self.y_train = np.array(self.y_train)
        self.y_cal = np.array(self.y_cal)


    def train_cal_test_split(self):
        """
        split data into train, calibration validation and test set, used in test mode
        return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_val,self.X_val, self.y_val, self.xX_test, self.X_test, self.y_test
        """
        if self.cal_size <= 1:
            cal_size = int(self.cal_size*len(self.df_train))
        else : 
            cal_size = int(self.cal_size)

        if self.test_size == None :
            test_size = cal_size
        else : 
            test_size = test_size

        self.X_train, self.X_test , self.y_train, self.y_test = train_test_split(self.df_train, self.target_train,test_size = test_size,shuffle = False)
        self.X_train, self.X_cal , self.y_train, self.y_cal = train_test_split(self.X_train, self.y_train,test_size = cal_size,shuffle = False)
        self.xX_train = self.X_train["DT_VALR"]
        self.X_train = self.X_train.drop(["DT_VALR","Valeur"],axis = 1)
        self.xX_cal = self.X_cal["DT_VALR"]
        self.X_cal = self.X_cal.drop(["DT_VALR","Valeur"], axis = 1)
        self.xX_test = self.X_test["DT_VALR"]
        self.X_test = self.X_test.drop(["DT_VALR","Valeur"], axis = 1)
        self.y_train = np.array(self.y_train)
        self.y_cal = np.array(self.y_cal)
        print("train_size = ", self.X_train.shape[0])
        return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_test, self.X_test, self.y_test

    def train_cal_val_test_split(self):
        """
        split data into train, calibration validation and test set, used in test mode
        return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_val,self.X_val, self.y_val, self.xX_test, self.X_test, self.y_test
        """
        cal_size = int(self.cal_size*len(self.df_train))
        val_size = int(self.val_size*len(self.df_train))
        if self.test_size == None :
            test_size = int(self.val_size*len(self.df_train))
        else :
            test_size = int(self.test_size*len(self.df_train))
        self.X_train, self.X_test , self.y_train, self.y_test = train_test_split(self.df_train, self.target_train,test_size = test_size,shuffle = False)
        self.X_train, self.X_val , self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,test_size = val_size,shuffle = False)
        self.X_train, self.X_cal , self.y_train, self.y_cal = train_test_split(self.X_train, self.y_train,test_size = cal_size,shuffle = False)
        self.xX_train = self.X_train["DT_VALR"]
        self.X_train = self.X_train.drop(["DT_VALR","Valeur"],axis = 1)
        self.xX_cal = self.X_cal["DT_VALR"]
        self.xX_val = self.X_val["DT_VALR"]
        self.X_cal = self.X_cal.drop(["DT_VALR","Valeur"], axis = 1)
        self.X_val = self.X_val.drop(["DT_VALR","Valeur"], axis = 1)
        self.xX_test = self.X_test["DT_VALR"]
        self.X_test = self.X_test.drop(["DT_VALR","Valeur"], axis = 1)
        self.y_train = list(self.y_train)
        self.y_cal = list(self.y_cal)
        self.y_val = list(self.y_val)
        self.X_calval = pd.concat([self.X_cal,self.X_val], sort = False)
        self.y_calval = list(self.y_cal) + list(self.y_val)
        print("train_size = ", self.X_train.shape[0])
        return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_val,self.X_val, self.y_val, self.xX_test, self.X_test, self.y_test

    def predict_cal_val(self):
        "predict on calibration and validation set"
        pred_cal = self.model.predict(self.X_cal)
        pred_val = self.model.predict(self.X_val)
        self.pred_down_cal = np.array(pred_cal[0])
        self.pred_up_cal = np.array(pred_cal[1])
        self.pred_median_cal = np.array(pred_cal[2])
        self.pred_down_val = np.array(pred_val[0])
        self.pred_up_val = np.array(pred_val[1])
        self.pred_median_val = np.array(pred_val[2])


    def predict_cal(self):
        "predict on calibration and validation set"
        pred_cal = self.model.predict(self.X_cal)
        self.pred_down_cal = np.array(pred_cal[0])
        self.pred_up_cal = np.array(pred_cal[1])
        self.pred_median_cal = np.array(pred_cal[2])

    def predict_test(self,X):
        "predict on predict_set"
        #Initialisation des predictions
        predict = self.model.predict(X)
        #Aggregation des predictions
        pred_down = np.array(predict[0])
        pred_up = np.array(predict[1])
        pred_median = np.array(predict[2])
        return(pred_down,pred_up,pred_median)
    
    def plot(self):
        #Show plot pred test 
        if self.mode == "normal":
            fig = go.Figure()
            fig.add_trace(go.Scatter( y=self.pred_up,
                          mode='lines',
                          name=f'q_{alphas[1]}',
                          line=dict(
                              color='rgb(0, 256, 0)',
                              width=0),
                          showlegend = False))

            fig.add_trace(go.Scatter( y=self.pred_down,
                          mode='lines',
                          name=f'{int(np.round(100*(self.alphas[1]-self.alphas[0])))}% Confidence Interval',
                          line=dict(
                              color='rgb(0, 256, 0)',
                              width=0),
                          fill='tonexty',
                          fillcolor='rgba(0,176,246,0.2)',
                          line_color='rgba(255,255,255,0)'))
      
            fig.add_trace(go.Scatter( y=self.pred_median,
                        mode='lines',
                        name=f'y_median',
                        line=dict(
                            color='rgb(256,0, 0)',
                            width=1),
                        showlegend = True))

            fig.update_layout(title = f"{self.groupe}, {int(np.round(100*self.confidence))}% confidence prediction interval")
            fig.show()

        if self.mode == "test":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.xX_test, y=self.y_test,
                        mode='lines',
                        name=f'y_true',
                        line=dict(
                            color='rgb(0,0, 256)',
                            width=1),
                        showlegend = True))

            fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_up,
                          mode='lines',
                          name=f'q_{alphas[1]}',
                          line=dict(
                              color='rgb(0, 256, 0)',
                              width=0),
                          showlegend = False))

            fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_down,
                          mode='lines',
                          name=f'{int(np.round(100*(self.alphas[1]-self.alphas[0])))}% Confidence Interval',
                          line=dict(
                              color='rgb(0, 256, 0)',
                              width=0),
                          fill='tonexty',
                          fillcolor='rgba(0,176,246,0.2)',
                          line_color='rgba(255,255,255,0)'))
            """
            fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_median,
                        mode='lines',
                        name=f'y_median',
                        line=dict(
                            color='rgb(256,0, 0)',
                            width=1),
                        showlegend = True))
            """

            error = (np.sum(self.y_test<self.pred_down) + np.sum(self.y_test>self.pred_up))/len(self.y_test)
            fig.update_traces(mode='lines')
            fig.update_layout(title = f"Test : {self.groupe}, {1-error}% confidence")                      
            fig.show()


    def f_conformity_score(self,pred_down_cal,pred_up_cal,y_cal):
        """
        Compute the symetric conformity score
        """
        return np.max([pred_down_cal-y_cal,y_cal-pred_up_cal],axis = 0)
  
    def f_conformity_score_down(self,pred_down_cal,y_cal):
        """
        Compute the asymetric conformity score for down bound
        """
        return [pred_down_cal-y_cal]

    def f_conformity_score_up(self,pred_up_cal,y_cal):
        """
        Compute the asymetric conformity score for upper bound
        """
        return [y_cal-pred_up_cal]
  
    def f_conformity_score_median(self,pred_median_cal,y_cal):
        """
        Compute the asymetric conformity score for upper bound
        """
        return [pred_median_cal-y_cal]
  
    def f_miscoverage_rate(self,pred_down_cal,pred_up_cal,pred_down_val,pred_up_val,y_cal,y_val,alpha):
        """
        Compute the miscoverage rate
        """
        csq = np.quantile(self.f_conformity_score(pred_down_cal,pred_up_cal,y_cal),1-alpha)
        return(np.sum(np.max([pred_down_val-y_val,y_val-pred_up_val],axis = 0)>csq)/len(y_val))

    def f_miscoverage_rate_down(self,pred_down_cal,pred_down_val,y_cal,y_val,alpha):
        """
        Compute the asymetric miscoverage rate for down bound
        """
        csq_low = np.quantile(self.f_conformity_score_down(pred_down_cal,y_cal),1-alpha)
        return(np.sum((pred_down_val-y_val)>csq_low)/len(y_val))

    def f_miscoverage_rate_median(self,pred_median_cal,pred_median_val,y_cal,y_val,alpha):
        """
        Compute the asymetric miscoverage rate for median bound
        """
        csq_median = np.quantile(self.f_conformity_score_median(pred_median_cal,y_cal),1-alpha)
        return(np.sum((pred_median_val-y_val)>csq_median)/len(y_val))

    def f_miscoverage_rate_up(self,pred_up_cal,pred_up_val,y_cal,y_val,alpha):
        """
        Compute the asymetric miscoverage rate for upper bound
        """
        csq_up = np.quantile(self.f_conformity_score_up(pred_up_cal,y_cal),1 - alpha)
        return(np.sum((y_val-pred_up_val)>csq_up)/len(y_val))
    
    def err_t(self,y_true,pred_up,pred_down,t):
        """
        Compute the adaptative error of adaptative conformal inference at time t
        """
        return (list(y_true)[-t]>list(pred_up)[-t] or list(y_true)[-t]<list(pred_down)[-t])

    def err_t_down(self,y_true,pred_down,t):
        """
        Compute the adaptative error of asymetric adaptative conformal inference for down bound at time t
        """
        return list(y_true)[-t]<=list(pred_down)[-t]

    def err_t_up(self,y_true,pred_up,t):
        """
        Compute the adaptative error of asymetric adaptative conformal inference for upper down bound at time t
        """
        return list(y_true)[-t]>=list(pred_up)[-t]
  
    def err_t_median(self,y_true,pred_median,t):
        """
        Compute the adaptative error of asymetric adaptative conformal inference for down bound at time t
        """
        return list(y_true)[-t]<=list(pred_median)[-t]
    
    


# COMMAND ----------

class Conformal_Inference():
  def __init__(self,alphas, cal_size,groupe, test_size = None,mode = "normal",dico_hyperparametres = None,gridsearch = False, param_grid_gb=None, param_grid_lgbm=None ):
    """
    alphas = [quantile_bottom,quantile_up]
    mode = ("normal" ou "test")
    """
    self.gridsearch = gridsearch
    self.param_grid_gb = param_grid_gb
    self.param_grid_lgbm = param_grid_lgbm
    self.alphas = alphas
    self.confidence = self.alphas[1] - self.alphas[0]
    self.cal_size = cal_size
    self.test_size = test_size
    self.fitted = False
    self.mode = mode
    self.adaptative = False
    self.groupe = groupe
    #Initialisation des modèles
    if dico_hyperparametres == None :
      self.dico_hyperparametres = {}
      self.dico_hyperparametres[groupe] = {}
      self.model = IC_model(alphas = self.alphas,groupe= groupe,gridsearch = gridsearch)
    else : 
       self.dico_hyperparametres = dico_hyperparametres
       self.model = IC_model(alphas = self.alphas,groupe= groupe, dico_hyperparametres = self.dico_hyperparametres,param_grid_gb = param_grid_gb,param_grid_lgbm = param_grid_lgbm,gridsearch = self.gridsearch)

  def fit(self,df_train):
    """
    Fit the model
    """
    self.target_train = df_train["Valeur"]
    self.df_train = df_train


    # On crée les sets de calibration et de validation
    if self.mode == "normal" or self.adaptative:
      self.train_cal_split()

    # En mode test on crée les sets de calibration de validation et de test
    if self.mode == "test" and not self.adaptative:
      self.train_cal_test_split()

    #On fit chacun des modèles
    self.model.fit(self.X_train,self.y_train)
    if self.gridsearch == True:
      self.model.gridsearchCV()
      np.save(f'/dbfs/FileStore/IC_hyperparametres.npy',self.model.dico_hyperparametres) 

    #On effectue les prediction sur les sets de calibration et de validation
    self.predict_cal()
    print("len(self.X_cal) = ",len(self.X_cal))
    print("len(self.y_cal) = ",len(self.y_cal))
    print("conformal fitted")
    self.fitted = True

  def predict(self,X_predict = None,plot = False):
    """
    Compute Asymetric Conformal Inference for predict set,
    X_predict = None,plot = False
    return self.pred_down_predict_asymetric_conformal,self.pred_up_predict_asymetric_conformal
    """
    self.X_predict = X_predict
    if self.mode == "test":
      self.X_predict = self.X_test
    #Prediction du set de test
    self.pred_down_predict,self.pred_up_predict,self.pred_median_predict = self.predict_test(self.X_predict)
    print(self.pred_down_predict)
    try:
      self.pred_down_predict_asymetric_conformal = self.pred_down_predict - np.quantile(self.f_conformity_score_down(self.pred_down_cal,self.y_cal),1- self.alphas[0])
      self.pred_up_predict_asymetric_conformal = self.pred_up_predict + np.quantile(self.f_conformity_score_up(self.pred_up_cal,self.y_cal),self.alphas[1])
      self.pred_median_predict_asymetric_conformal = self.pred_median_predict - np.quantile(self.f_conformity_score_median(self.pred_median_cal,self.y_cal),0.5)
    except:
      self.pred_down_predict_asymetric_conformal,self.pred_up_predict_asymetric_conformal,self.pred_median_predict_asymetric_conformal = self.pred_down_predict,self.pred_up_predict,self.pred_median_predict
    if plot:
      self.plot()
    self.X_predict[f"{np.round(self.alphas[0],decimals = 3)}_quantile"] = self.pred_down_predict_asymetric_conformal
    self.X_predict[f"{np.round(self.alphas[1],decimals = 3)}_quantile"] = self.pred_up_predict_asymetric_conformal
    self.X_predict[f"0.5_quantile"] = self.pred_median_predict_asymetric_conformal
    return(self.X_predict)

  def train_cal_split(self):
    if self.cal_size < 1:
      cal_size = int(self.cal_size*len(self.df_train))
    else : 
      cal_size = int(self.cal_size)
    "split data into train, calibration, used in normal mode"
    self.X_train, self.X_cal , self.y_train, self.y_cal = train_test_split(self.df_train, self.target_train,test_size = cal_size,shuffle = False)
    self.xX_train = self.X_train["DT_VALR"]
    self.X_train = self.X_train.drop(["DT_VALR","Valeur"],axis = 1)
    self.xX_cal = self.X_cal["DT_VALR"]
    self.X_cal = self.X_cal.drop(["DT_VALR","Valeur"], axis = 1)
    #return self.xX_train, self.xX_cal, self.xX_val, self.xX_test, self.X_train, self.X_cal, self.X_val, self.X_test, self.y_train, self.y_cal, self.y_val, self.y_test
    self.y_train = list(self.y_train)
    self.y_cal = list(self.y_cal)
    print("train_size = ", self.X_train.shape[0])
  """
  def train_cal_val_split(self):
    "split data into train, calibration and validation set, used in normal mode"
    self.X_train, self.X_val , self.y_train, self.y_val = train_test_split(self.df_train, self.target_train,test_size = self.val_size,shuffle = False)
    self.X_train, self.X_cal , self.y_train, self.y_cal = train_test_split(self.X_train, self.y_train ,test_size = self.cal_size,shuffle = False)
    self.xX_train = self.X_train["DT_VALR"]
    self.X_train = self.X_train.drop(["DT_VALR","Valeur"],axis = 1)
    self.xX_cal = self.X_cal["DT_VALR"]
    self.xX_val = self.X_val["DT_VALR"]
    self.X_cal = self.X_cal.drop(["DT_VALR","Valeur"], axis = 1)
    self.X_val = self.X_val.drop(["DT_VALR","Valeur"], axis = 1)
    #return self.xX_train, self.xX_cal, self.xX_val, self.xX_test, self.X_train, self.X_cal, self.X_val, self.X_test, self.y_train, self.y_cal, self.y_val, self.y_test
    self.y_train = list(self.y_train)
    self.y_cal = list(self.y_cal)
    self.y_val = list(self.y_val)
    self.X_calval = pd.concat([self.X_cal,self.X_val], sort = False)
    self.y_calval = list(self.y_cal) + list(self.y_val)
    print("train_size = ", self.X_train.shape[0])
  """
  def train_cal_test_split(self):
    """
    split data into train, calibration validation and test set, used in test mode
    return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_val,self.X_val, self.y_val, self.xX_test, self.X_test, self.y_test
    """
    if self.cal_size <= 1:
      cal_size = int(self.cal_size*len(self.df_train))
    else : 
      cal_size = int(self.cal_size)

    if self.test_size == None :
      test_size = cal_size
    else : 
      test_size = test_size

    self.X_train, self.X_test , self.y_train, self.y_test = train_test_split(self.df_train, self.target_train,test_size = test_size,shuffle = False)
    self.X_train, self.X_cal , self.y_train, self.y_cal = train_test_split(self.X_train, self.y_train,test_size = cal_size,shuffle = False)
    self.xX_train = self.X_train["DT_VALR"]
    self.X_train = self.X_train.drop(["DT_VALR","Valeur"],axis = 1)
    self.xX_cal = self.X_cal["DT_VALR"]
    self.X_cal = self.X_cal.drop(["DT_VALR","Valeur"], axis = 1)
    self.xX_test = self.X_test["DT_VALR"]
    self.X_test = self.X_test.drop(["DT_VALR","Valeur"], axis = 1)
    self.y_train = list(self.y_train)
    self.y_cal = list(self.y_cal)
    print("train_size = ", self.X_train.shape[0])
    return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_test, self.X_test, self.y_test

  def train_cal_val_test_split(self):
    """
    split data into train, calibration validation and test set, used in test mode
    return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_val,self.X_val, self.y_val, self.xX_test, self.X_test, self.y_test
    """
    cal_size = int(self.cal_size*len(self.df_train))
    val_size = int(self.val_size*len(self.df_train))
    if self.test_size == None :
      test_size = int(self.val_size*len(self.df_train))
    else :
      test_size = int(self.test_size*len(self.df_train))
    self.X_train, self.X_test , self.y_train, self.y_test = train_test_split(self.df_train, self.target_train,test_size = test_size,shuffle = False)
    self.X_train, self.X_val , self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,test_size = val_size,shuffle = False)
    self.X_train, self.X_cal , self.y_train, self.y_cal = train_test_split(self.X_train, self.y_train,test_size = cal_size,shuffle = False)
    self.xX_train = self.X_train["DT_VALR"]
    self.X_train = self.X_train.drop(["DT_VALR","Valeur"],axis = 1)
    self.xX_cal = self.X_cal["DT_VALR"]
    self.xX_val = self.X_val["DT_VALR"]
    self.X_cal = self.X_cal.drop(["DT_VALR","Valeur"], axis = 1)
    self.X_val = self.X_val.drop(["DT_VALR","Valeur"], axis = 1)
    self.xX_test = self.X_test["DT_VALR"]
    self.X_test = self.X_test.drop(["DT_VALR","Valeur"], axis = 1)
    self.y_train = list(self.y_train)
    self.y_cal = list(self.y_cal)
    self.y_val = list(self.y_val)
    self.X_calval = pd.concat([self.X_cal,self.X_val], sort = False)
    self.y_calval = list(self.y_cal) + list(self.y_val)
    print("train_size = ", self.X_train.shape[0])
    return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_val,self.X_val, self.y_val, self.xX_test, self.X_test, self.y_test

  def predict_cal_val(self):
    "predict on calibration and validation set"
    pred_cal = self.model.predict(self.X_cal)
    pred_val = self.model.predict(self.X_val)
    self.pred_down_cal = pred_cal[0]
    self.pred_up_cal = pred_cal[1]
    self.pred_median_cal = pred_cal[2]
    self.pred_down_val = pred_val[0]
    self.pred_up_val = pred_val[1]
    self.pred_median_val = pred_val[2]
    """
    #Conformity score conformal inference
    self.conformity_score = self.f_conformity_score(self.pred_down_cal,self.pred_up_cal,self.y_cal)
    self.conformity_score_down = self.f_conformity_score_down(self.pred_down_cal,self.y_cal)
    self.conformity_score_up = self.f_conformity_score_up(self.pred_up_cal,self.y_cal)
    """

  def predict_cal(self):
    "predict on calibration and validation set"
    pred_cal = self.model.predict(self.X_cal)
    self.pred_down_cal = pred_cal[0]
    self.pred_up_cal = pred_cal[1]
    self.pred_median_cal = pred_cal[2]

  def predict_test(self,X):
    "predict on predict_set"
    #Initialisation des predictions
    predict = self.model.predict(X)
    #Aggregation des predictions
    pred_down = predict[0]
    pred_up = predict[1]
    pred_median = predict[2]
    return(pred_down,pred_up,pred_median)

  def plot(self) :
    #Show plot pred test 
    if self.mode == "normal":
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
      
      fig.add_trace(go.Scatter( y=self.pred_median_predict_asymetric_conformal,
                    mode='lines',
                    name=f'y_median',
                    line=dict(
                        color='rgb(256,0, 0)',
                        width=1),
                    showlegend = True))
      
      fig.update_layout(title = f"{int(np.round(100*self.confidence))}% Asymetric conformal prediction interval")
      fig.show()

    if self.mode == "test":
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=self.xX_test, y=self.y_test,
                    mode='lines',
                    name=f'y_true',
                    line=dict(
                        color='rgb(0,0, 256)',
                        width=1),
                    showlegend = True))
              
      fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_up_predict_asymetric_conformal,
                      mode='lines',
                      name=f'q_{alphas[1]}',
                      line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                      showlegend = False))

      fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_down_predict_asymetric_conformal,
                      mode='lines',
                      name=f'{int(np.round(100*(self.alphas[1]-self.alphas[0])))}% Confidence Interval',
                      line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                      fill='tonexty',
                      fillcolor='rgba(0,176,246,0.2)',
                      line_color='rgba(255,255,255,0)'))
      
      fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_median_predict_asymetric_conformal,
                    mode='lines',
                    name=f'y_median',
                    line=dict(
                        color='rgb(256,0, 0)',
                        width=1),
                    showlegend = True))
      
      error = (np.sum(self.y_test<self.pred_down_predict_asymetric_conformal) + np.sum(self.y_test>self.pred_up_predict_asymetric_conformal))/len(self.y_test)
      fig.update_traces(mode='lines')
      fig.update_layout(title = f"Test : {1-error}% asymetric conformal prediction test")                      
      fig.show()


  def f_conformity_score(self,pred_down_cal,pred_up_cal,y_cal):
    """
    Compute the symetric conformity score
    """
    return np.max([pred_down_cal-y_cal,y_cal-pred_up_cal],axis = 0)
  
  def f_conformity_score_down(self,pred_down_cal,y_cal):
    """
    Compute the asymetric conformity score for down bound
    """
    return [pred_down_cal-y_cal]

  def f_conformity_score_up(self,pred_up_cal,y_cal):
    """
    Compute the asymetric conformity score for upper bound
    """
    return [y_cal-pred_up_cal]
  
  def f_conformity_score_median(self,pred_median_cal,y_cal):
    """
    Compute the asymetric conformity score for upper bound
    """
    return [pred_median_cal-y_cal]
  
  def f_miscoverage_rate(self,pred_down_cal,pred_up_cal,pred_down_val,pred_up_val,y_cal,y_val,alpha):
    """
    Compute the miscoverage rate
    """
    csq = np.quantile(self.f_conformity_score(pred_down_cal,pred_up_cal,y_cal),1-alpha)
    return(np.sum(np.max([pred_down_val-y_val,y_val-pred_up_val],axis = 0)>csq)/len(y_val))

  def f_miscoverage_rate_down(self,pred_down_cal,pred_down_val,y_cal,y_val,alpha):
    """
    Compute the asymetric miscoverage rate for down bound
    """
    csq_low = np.quantile(self.f_conformity_score_down(pred_down_cal,y_cal),1-alpha)
    return(np.sum((pred_down_val-y_val)>csq_low)/len(y_val))
  
  def f_miscoverage_rate_median(self,pred_median_cal,pred_median_val,y_cal,y_val,alpha):
    """
    Compute the asymetric miscoverage rate for median bound
    """
    csq_median = np.quantile(self.f_conformity_score_median(pred_median_cal,y_cal),1-alpha)
    return(np.sum((pred_median_val-y_val)>csq_median)/len(y_val))

  def f_miscoverage_rate_up(self,pred_up_cal,pred_up_val,y_cal,y_val,alpha):
    """
    Compute the asymetric miscoverage rate for upper bound
    """
    csq_up = np.quantile(self.f_conformity_score_up(pred_up_cal,y_cal),1 - alpha)
    return(np.sum((y_val-pred_up_val)>csq_up)/len(y_val))

# COMMAND ----------

class Conformal_Inference_qrf():
  def __init__(self,groupe, cal_size ,mode = "normal",test_size = None):
    """
    alphas = [quantile_bottom,quantile_up]
    mode = ("normal" ou "test")
    """
    self.groupe = groupe
    self.test_size = test_size
    self.cal_size = cal_size
    self.fitted = False
    self.mode = mode
    self.adaptative = False
    

  def fit(self,df_train):
    self.model = RandomForestQuantileRegressor(n_estimators= 64, min_samples_leaf = int(math.log(len(df_train))), warm_start = True,bootstrap = True,max_depth = 5)
    """
    Fit the model
    """
    self.target_train = df_train["Valeur"]
    self.DT_VALR_train = df_train["DT_VALR"]
    df_train = df_train.drop(["DT_VALR",["Valeur"]],axis = 1)
    self.scaler = MinMaxScaler()
    self.scaler.fit(df_train)
    df_array = self.scaler.transform(df_train)
    
    self.df_train = pd.DataFrame(df_array)
    self.df_train = df_train
    self.df_train["DT_VALR"] = self.DT_VALR_train
    print(df_train)
    # On crée les sets de calibration et de validation
    if self.mode == "normal" or self.adaptative:
      self.train_cal_split()

    # En mode test on crée les sets de calibration de validation et de test
    if self.mode == "test" and not self.adaptative:
      self.train_cal_test_split()

    #On fit chacun des modèles
    self.model.fit(self.X_train, self.y_train)

    #On effectue les prediction sur les sets de calibration 
    print("len(self.X_cal) = ", len(self.X_cal))
    print("len(self.y_cal) = ", len(self.y_cal))
    print("conformal fitted")
    self.fitted = True

  def pred_all_quantiles(self,X_predict = None):
    self.df_output = pd.DataFrame()
    self.X_predict = X_predict
    if self.mode == "test":
      self.X_predict = self.X_test
    if self.mode == 'normal':
      self.X_predict_dtvalr = preproc.set_DT_VALR(X_predict, self.df_train)
      self.df_output["DT_VALR"] = self.X_predict_dtvalr["DT_VALR"]
    elif self.mode =="test":
      self.df_output["DT_VALR"] = self.xX_test
    if X_predict is not None:
      self.X_predict = X_predict
      self.X_predict = self.scaler.transform(self.X_predict)
    if self.mode == "test":
      self.X_predict = self.X_test

    for alpha in np.arange(0.01,0.5,0.01):
      self.alphas = [alpha,1-alpha]
      self.predict_cal()
      self.pred_down_predict = self.model.predict(self.X_predict, quantiles = self.alphas[0])
      self.pred_up_predict = self.model.predict(self.X_predict, quantiles = self.alphas[1])
      try:
        self.pred_down_predict_asymetric_conformal = self.pred_down_predict - np.quantile(self.f_conformity_score_down(self.pred_down_cal,self.y_cal),1- self.alphas[0])
      except Exception as e:
        print("down non conformal")
        print(e)
        self.pred_down_predict_asymetric_conformal = self.pred_down_predict
      try:
        self.pred_up_predict_asymetric_conformal = self.pred_up_predict + np.quantile(self.f_conformity_score_up(self.pred_up_cal,self.y_cal),self.alphas[1])
      except Exception as e:
        print("up non conformal")
        print(e)
        self.pred_down_predict_asymetric_conformal,self.pred_up_predict_asymetric_conformal = self.pred_down_predict,self.pred_up_predict
      self.df_output[f"{np.round(self.alphas[0],decimals = 3)}_quantile"] = self.pred_down_predict_asymetric_conformal
      self.df_output[f"{np.round(self.alphas[1],decimals = 3)}_quantile"] = self.pred_up_predict_asymetric_conformal
    self.pred_median_predict = self.model.predict(self.X_predict,quantiles = 0.5)  
    try:
      self.pred_median_predict_asymetric_conformal = self.pred_median_predict - np.quantile(self.f_conformity_score_median(self.pred_median_cal,self.y_cal),0.5)
    except Exception as e:
      print("down non conformal")
      print(e)
      self.pred_median_predict_asymetric_conformal = self.pred_median_predict
    self.df_output[f"{0.5}_quantile"] = self.pred_up_predict_asymetric_conformal
    return(self.df_output)

  def asymetric_conformal_IC(self,X_predict,alphas,plot = False,show_median = True):
    self.alphas = alphas
    """
    Compute Asymetric Conformal Inference for predict set,
    X_predict = None,plot = False
    return self.pred_down_predict_asymetric_conformal,self.pred_up_predict_asymetric_conformal
    """
    self.predict_cal()
    self.X_predict = X_predict
    if self.mode == "test":
      self.X_predict = self.X_test
    #Prediction du set de test
    self.pred_down_predict = self.model.predict(self.X_predict,quantiles = alphas[0])
    self.pred_up_predict = self.model.predict(self.X_predict,quantiles = alphas[1])
    if show_median:
      self.pred_median_predict = self.model.predict(self.X_predict,quantiles = 0.5)
    try:
      self.pred_down_predict_asymetric_conformal = self.pred_down_predict - np.quantile(self.f_conformity_score_down(self.pred_down_cal,self.y_cal),1- self.alphas[0])
    except Exception as error:
      print("pred_down_nonconformel",error)
      print(np.quantile(self.f_conformity_score_down(self.pred_down_cal,self.y_cal),1- self.alphas[0]))
      self.pred_down_predict_asymetric_conformal = self.pred_down_predict
    try:
      self.pred_up_predict_asymetric_conformal = self.pred_up_predict + np.quantile(self.f_conformity_score_up(self.pred_up_cal,self.y_cal),self.alphas[1])
    except Exception as error :
      print("pred_up_nonconformel")
      self.pred_up_predict_asymetric_conformal = self.pred_up_predict
    if show_median:
      try:
        self.pred_median_predict_asymetric_conformal = self.pred_median_predict - np.quantile(self.f_conformity_score_median(self.pred_median_cal,self.y_cal),0.5)
      except:
        self.pred_median_predict_asymetric_conformal = self.pred_median_predict
    if plot:
      self.plot(show_median = show_median)
    self.df_output = pd.DataFrame([self.X_predict])
    self.df_output[f"{np.round(self.alphas[0],decimals = 3)}_quantile"] = self.pred_down_predict_asymetric_conformal
    self.df_output[f"{np.round(self.alphas[1],decimals = 3)}_quantile"] = self.pred_up_predict_asymetric_conformal
    if show_median:
      self.df_output[f"{0.5}_quantile"] = self.pred_median_predict_asymetric_conformal
    return(self.df_output)

  def train_cal_split(self):
    if self.cal_size < 1:
      cal_size = int(self.cal_size*len(self.df_train))
    else : 
      cal_size = int(self.cal_size)
    "split data into train, calibration, used in normal mode"
    self.X_train, self.X_cal , self.y_train, self.y_cal = train_test_split(self.df_train, self.target_train,test_size = cal_size,shuffle = False)
    self.xX_train = self.X_train["DT_VALR"]
    self.X_train = self.X_train.drop(["DT_VALR","Valeur"],axis = 1)
    self.xX_cal = self.X_cal["DT_VALR"]
    self.X_cal = self.X_cal.drop(["DT_VALR","Valeur"], axis = 1)
    #return self.xX_train, self.xX_cal, self.xX_val, self.xX_test, self.X_train, self.X_cal, self.X_val, self.X_test, self.y_train, self.y_cal, self.y_val, self.y_test
    self.y_train = list(self.y_train)
    self.y_cal = list(self.y_cal)
    print("train_size = ", self.X_train.shape[0])


  def train_cal_test_split(self):
    """
    split data into train, calibration validation and test set, used in test mode
    return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_val,self.X_val, self.y_val, self.xX_test, self.X_test, self.y_test
    """
    if self.cal_size <= 1:
      cal_size = int(self.cal_size*len(self.df_train))
    else : 
      cal_size = int(self.cal_size)

    if self.test_size == None :
      test_size = cal_size
    else : 
      if self.test_size <= 1:
        test_size = int(self.test_size*len(self.df_train))
      else : 
        test_size = int(self.test_size)

    self.X_train, self.X_test , self.y_train, self.y_test = train_test_split(self.df_train, self.target_train,test_size = test_size,shuffle = False)
    self.X_train, self.X_cal , self.y_train, self.y_cal = train_test_split(self.X_train, self.y_train,test_size = cal_size,shuffle = False)
    self.xX_train = self.X_train["DT_VALR"]
    self.X_train = self.X_train.drop(["DT_VALR","Valeur"],axis = 1)
    self.xX_cal = self.X_cal["DT_VALR"]
    self.X_cal = self.X_cal.drop(["DT_VALR","Valeur"], axis = 1)
    self.xX_test = self.X_test["DT_VALR"]
    self.X_test = self.X_test.drop(["DT_VALR","Valeur"], axis = 1)
    self.y_train = list(self.y_train)
    self.y_cal = list(self.y_cal)
    print("train_size = ", self.X_train.shape[0])
    return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_test, self.X_test, self.y_test

  def predict_cal(self):
    "predict on calibration and validation set"
    self.pred_down_cal = self.model.predict(self.X_cal,quantiles = self.alphas[0])
    self.pred_up_cal = self.model.predict(self.X_cal,quantiles = self.alphas[1])
    self.pred_median_cal = self.model.predict(self.X_cal,quantiles = 0.5)

  def predict_test(self,X):
    "predict on predict_set"
    #Initialisation des predictions
    #Aggregation des predictions
    pred_down = self.model.predict(X,quantiles = self.alphas[0])
    pred_up = self.model.predict(X,quantiles = self.alphas[1])
    pred_median = self.model.predict(X,quantiles = 0.5)
    return(pred_down,pred_up,pred_median)
  
  def predict(self,alphas,plot = True,show_median = False,X_predict = None):
    self.df_output = pd.DataFrame()
    if self.mode == "test":
      self.X_predict = self.X_test
    if X_predict is not None:
      self.X_predict = X_predict
    if self.mode == 'normal':
      self.X_predict_dtvalr = preproc.set_DT_VALR(X_predict,self.df_train)
      self.df_output["DT_VALR"] = self.X_predict_dtvalr["DT_VALR"]
    elif self.mode =="test":
      self.df_output["DT_VALR"] = self.xX_test
    if X_predict is not None:
      self.X_predict = X_predict
      self.X_predict = self.scaler.transform(self.X_predict)
    if self.mode == "test":
      self.X_predict = self.X_test
    self.alphas = alphas
    self.confidence = self.alphas[1]-self.alphas[0]
    self.predict_cal()
    self.pred_down_predict = self.model.predict(self.X_predict,quantiles = alphas[0])
    self.pred_up_predict = self.model.predict(self.X_predict,quantiles = alphas[1])
    if show_median:
      self.pred_median_predict = self.model.predict(self.X_predict,quantiles = 0.5)
    try:
      self.pred_down_predict_asymetric_conformal = self.pred_down_predict - np.quantile(self.f_conformity_score_down(self.pred_down_cal,self.y_cal),1- alphas[0])
      self.pred_up_predict_asymetric_conformal = self.pred_up_predict + np.quantile(self.f_conformity_score_up(self.pred_up_cal,self.y_cal),alphas[1])
      if show_median:
        self.pred_median_predict_asymetric_conformal = self.pred_median_predict - np.quantile(self.f_conformity_score_median(self.pred_median_cal,self.y_cal),0.5)
    except Exception as e:
      print(e)
      self.pred_down_predict_asymetric_conformal,self.pred_up_predict_asymetric_conformal = self.pred_down_predict,self.pred_up_predict
      if show_median:
        self.pred_median_predict = self.pred_median_predict_asymetric_conformal

    if plot:
      self.plot(show_median = show_median)
    self.df_output[f"{np.round(self.alphas[0],decimals = 3)}_quantile"] = self.pred_down_predict_asymetric_conformal
    self.df_output[f"{np.round(self.alphas[1],decimals = 3)}_quantile"] = self.pred_up_predict_asymetric_conformal
    return(self.pred_down_predict,self.pred_up_predict)
    
  def plot(self,show_median) : #plot a visualisation of result
     
    if self.mode == "normal":
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
      if show_median:
        fig.add_trace(go.Scatter( y=self.pred_median_predict_asymetric_conformal,
                      mode='lines',
                      name=f'y_median',
                      line=dict(
                          color='rgb(256,0, 0)',
                          width=1),
                      showlegend = True))
      
      fig.update_layout(title = f"{int(np.round(100*self.confidence))}% Asymetric conformal prediction interval")
      fig.show()

    if self.mode == "test":
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=self.xX_test, y=self.y_test,
                    mode='lines',
                    name=f'y_true',
                    line=dict(
                        color='rgb(0,0, 256)',
                        width=1),
                    showlegend = True))
              
      fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_up_predict_asymetric_conformal,
                      mode='lines',
                      name=f'q_{alphas[1]}',
                      line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                      showlegend = False))

      fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_down_predict_asymetric_conformal,
                      mode='lines',
                      name=f'{int(np.round(100*(self.alphas[1]-self.alphas[0])))}% Confidence Interval',
                      line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                      fill='tonexty',
                      fillcolor='rgba(0,176,246,0.2)',
                      line_color='rgba(255,255,255,0)'))
      if show_median:
        fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_median_predict_asymetric_conformal,
                      mode='lines',
                      name=f'y_median',
                      line=dict(
                          color='rgb(256,0, 0)',
                          width=1),
                      showlegend = True))
      
      error = (np.sum(self.y_test<self.pred_down_predict_asymetric_conformal) + np.sum(self.y_test>self.pred_up_predict_asymetric_conformal))/len(self.y_test)
      fig.update_traces(mode='lines')
      fig.update_layout(title = f"Test : {100*(1-error)}% asymetric conformal prediction test")                      
      fig.show()


  def f_conformity_score(self,pred_down_cal,pred_up_cal,y_cal):
    """
    Compute the symetric conformity score
    """
    return np.max([pred_down_cal-y_cal,y_cal-pred_up_cal],axis = 0)
  
  def f_conformity_score_down(self,pred_down_cal,y_cal):
    """
    Compute the asymetric conformity score for down bound
    """
    return [pred_down_cal-y_cal]

  def f_conformity_score_up(self,pred_up_cal,y_cal):
    """
    Compute the asymetric conformity score for upper bound
    """
    return [y_cal-pred_up_cal]
  
  def f_conformity_score_median(self,pred_median_cal,y_cal):
    """
    Compute the asymetric conformity score for upper bound
    """
    return [pred_median_cal-y_cal]
  
  def f_miscoverage_rate(self,pred_down_cal,pred_up_cal,pred_down_val,pred_up_val,y_cal,y_val,alpha):
    """
    Compute the miscoverage rate
    """
    csq = np.quantile(self.f_conformity_score(pred_down_cal,pred_up_cal,y_cal),1-alpha)
    return(np.sum(np.max([pred_down_val-y_val,y_val-pred_up_val],axis = 0)>csq)/len(y_val))

  def f_miscoverage_rate_down(self,pred_down_cal,pred_down_val,y_cal,y_val,alpha):
    """
    Compute the asymetric miscoverage rate for down bound
    """
    csq_low = np.quantile(self.f_conformity_score_down(pred_down_cal,y_cal),1-alpha)
    return(np.sum((pred_down_val-y_val)>csq_low)/len(y_val))
  
  def f_miscoverage_rate_median(self,pred_median_cal,pred_median_val,y_cal,y_val,alpha):
    """
    Compute the asymetric miscoverage rate for median bound
    """
    csq_median = np.quantile(self.f_conformity_score_median(pred_median_cal,y_cal),1-alpha)
    return(np.sum((pred_median_val-y_val)>csq_median)/len(y_val))

  def f_miscoverage_rate_up(self,pred_up_cal,pred_up_val,y_cal,y_val,alpha):
    """
    Compute the asymetric miscoverage rate for upper bound
    """
    csq_up = np.quantile(self.f_conformity_score_up(pred_up_cal,y_cal),1 - alpha)
    return(np.sum((y_val-pred_up_val)>csq_up)/len(y_val))

# COMMAND ----------

class IC_model: #prediction models
  def __init__(self,alphas,groupe,dico_hyperparametres = None,gridsearch = None, param_grid_gb=None, param_grid_lgbm=None,n_fold_gridsearch = 3):
    self.alphas = alphas
    self.confidence = self.alphas[1]-self.alphas[0]
    self.model_down_set = []
    self.model_up_set = []
    self.model_median_set = []
    self.groupe = groupe
    self.dico_hyperparametres = dico_hyperparametres
    self.gridsearch = gridsearch
    self.param_grid_gb=param_grid_gb
    self.param_grid_lgbm=param_grid_lgbm
    self.n_folds_gridsearch = n_fold_gridsearch  



  def fit(self,df_train,y=None):
    """
    Fit the model
    """
    try :
      self.model_down_set.append(GradientBoostingRegressor(loss="quantile", alpha=alphas[0],**self.dico_hyperparametres[groupe]["GB"]["down"]))
      self.model_up_set.append(GradientBoostingRegressor(loss="quantile", alpha=alphas[1],**self.dico_hyperparametres[groupe]["GB"]["up"]))
      self.model_median_set.append(GradientBoostingRegressor(loss="quantile", alpha=0.5,**self.dico_hyperparametres[groupe]["GB"]["median"]))
      self.model_down_set.append(LGBMRegressor(alpha=self.alphas[0],  objective='quantile',**self.dico_hyperparametres[groupe]["LGBM"]["down"]))
      self.model_up_set.append(LGBMRegressor(alpha=self.alphas[1],  objective='quantile',**self.dico_hyperparametres[groupe]["LGBM"]["up"]))
      self.model_median_set.append(LGBMRegressor(alpha=0.5,  objective='quantile',**self.dico_hyperparametres[groupe]["LGBM"]["median"]))
      self.qrf =  RandomForestQuantileRegressor( min_samples_leaf = int(math.log(len(df_train))), n_estimators = 128,max_depth = 5)

    except :
      self.model_down_set.append(GradientBoostingRegressor(loss="quantile", alpha=self.alphas[0]))
      self.model_up_set.append(GradientBoostingRegressor(loss="quantile", alpha=self.alphas[1]))
      self.model_median_set.append(GradientBoostingRegressor(loss="quantile", alpha=0.5))
      self.model_down_set.append(LGBMRegressor(alpha=self.alphas[0],  objective='quantile'))
      self.model_up_set.append(LGBMRegressor(alpha=self.alphas[1],  objective='quantile'))
      self.model_median_set.append(LGBMRegressor(alpha=0.5,  objective='quantile'))
      self.qrf = RandomForestQuantileRegressor( min_samples_leaf = int(math.log(len(df_train))), n_estimators = 128,max_depth = 5)
      self.dico_hyperparametres = {}
      self.dico_hyperparametres[groupe] = {}
      self.dico_hyperparametres[groupe]["LGBM"] = {}
      self.dico_hyperparametres[groupe]["GB"] = {}
      self.dico_hyperparametres[groupe]["QRF"] = {}


    if y is not None :
      self.y_train = y
      self.X_train = df_train
    else:
      self.y_train = df_train["Valeur"]
      self.index_train = df_train["DT_VALR"]
      self.X_train = df_train.drop(["Valeur","DT_VALR"],axis = 1)

    if self.gridsearch:
      self.gridsearchCV(self.X_train,self.y_train)

    for i in range(len(self.model_down_set)):
      self.model_down_set[i] = self.model_down_set[i].fit(self.X_train,self.y_train)
      self.model_up_set[i] = self.model_up_set[i].fit(self.X_train,self.y_train)
      self.model_median_set[i] = self.model_median_set[i].fit(self.X_train,self.y_train)
      self.qrf.fit(self.X_train,self.y_train)
    self.fitted = True
    print("IC_fitted")

  def predict(self,df_test,plot = False):
    "predict on predict_set"
    X = df_test
    self.pred_down_set = []
    self.pred_up_set = []
    self.pred_median_set = []

    for _ in range(len(self.model_down_set)):
      self.pred_down_set.append(self.model_down_set[_].predict(X))
      self.pred_up_set.append(self.model_up_set[_].predict(X))
      self.pred_median_set.append(self.model_median_set[_].predict(X))
      self.pred_down_set.append(self.qrf.predict(X,quantiles = self.alphas[0]))
      self.pred_up_set.append(self.qrf.predict(X,quantiles = self.alphas[1]))
      self.pred_median_set.append(self.qrf.predict(X,quantiles = 0.5))
                                

    self.pred_down = np.mean(self.pred_down_set,axis = 0)
    self.pred_up = np.mean(self.pred_up_set,axis = 0)
    self.pred_median = np.mean(self.pred_median_set,axis = 0)
    
    if plot :
      self.plot()
    return(self.pred_down,self.pred_up,self.pred_median)
  
  def plot(self):
    fig = go.Figure()
    fig.add_trace(go.Scatter( y=self.pred_up,
                    mode='lines',
                    name=f'q_{alphas[1]}',
                    line=dict(
                        color='rgb(0, 256, 0)',
                        width=0),
                    showlegend = False))

    fig.add_trace(go.Scatter( y=self.pred_down,
                    mode='lines',
                    name=f'{int(np.round(100*(self.alphas[1]-self.alphas[0])))}% Confidence Interval',
                    line=dict(
                        color='rgb(0, 256, 0)',
                        width=0),
                    fill='tonexty',
                    fillcolor='rgba(0,176,246,0.2)',
                    line_color='rgba(255,255,255,0)'))
    
    fig.add_trace(go.Scatter( y=self.pred_median,
                  mode='lines',
                  name=f'y_median',
                  line=dict(
                      color='rgb(256,0, 0)',
                      width=1),
                  showlegend = True))
    fig.update_layout(title = f"{int(np.round(100*self.confidence))} Prediction interval")
    fig.show()
  
  
  def gridsearchCV(self,X = None,y = None): #GridSearchCV
    self.dico_hyperparametres[groupe] = {}
    if self.param_grid_gb == None:
      self.param_grid_gb = {'n_estimators': [100,200,300,1000],
                                      'learning_rate' : [0.01,0.05,0.1,0.5],
                                      'max_depth' : [3,5,10,50]
                          }
    if self.param_grid_lgbm == None :
      self.param_grid_lgbm = {'n_estimators': [100,200,300,1000],
                                      'learning_rate' : [0.01,0.05,0.1,0.5]
                          }
    self.model_down_set = []
    self.model_up_set = []
    self.model_median_set = []

    #GradientBoosting gridsearch
    mqloss_scorer_up = make_scorer(mqloss, alpha=0.90)
    mqloss_scorer_down = make_scorer(mqloss, alpha=0.05)
    mqloss_scorer_median = make_scorer(mqloss, alpha=0.5)

    
    print("GridSearchCV Gradient_boosting down")
    print("param_grid = ", self.param_grid_gb)
    self.dico_hyperparametres[groupe]["GB"] = {}
    GB_down = GridSearchCV(
                          estimator=GradientBoostingRegressor(loss="quantile", alpha=self.alphas[0]),
                          param_grid=self.param_grid_gb,
                          scoring=mqloss_scorer_down,
                          cv=TimeSeriesSplit(self.n_folds_gridsearch),
                          n_jobs=-1,
                          verbose=1
    )
    self.model_down_set.append(GB_down.fit(X, y).best_estimator_)
    print("GB_down.best_params = ", GB_down.best_params_)
    self.dico_hyperparametres[groupe]["GB"]["down"] = GB_down.best_params_

    print("GridSearchCV Gradient_boosting up")
    GB_up = GridSearchCV(
                          estimator=GradientBoostingRegressor(loss="quantile", alpha=self.alphas[1]),
                          param_grid=self.param_grid_gb,
                          cv=TimeSeriesSplit(self.n_folds_gridsearch),
                          scoring=mqloss_scorer_up,
                          n_jobs=-1,
                          verbose=1
    )
    self.model_up_set.append(GB_up.fit(X, y).best_estimator_)
    print("GB_up.best_params = ", GB_up.best_params_)
    self.dico_hyperparametres[groupe]["GB"]["up"] = GB_up.best_params_

    print("GridSearchCV Gradient_boosting median")
    GB_median = GridSearchCV(
                          estimator=GradientBoostingRegressor(loss="quantile", alpha=0.5),
                          param_grid=self.param_grid_gb,
                          cv=TimeSeriesSplit(self.n_folds_gridsearch),
                          scoring=mqloss_scorer_median,
                          n_jobs=-1,
                          verbose=1
    )
    self.model_median_set.append(GB_median.fit(X, y).best_estimator_)
    print("GB_median.best_params = ", GB_median.best_params_)
    self.dico_hyperparametres[groupe]["GB"]["median"] = GB_median.best_params_
    
    
    #LGBM gridsearch
    self.dico_hyperparametres[groupe]["LGBM"] = {}
    print("GridSearchCV LGBM down")
    print("param_grid = ", self.param_grid_lgbm)
    LGBM_down = GridSearchCV(
                          estimator=LGBMRegressor(alpha=self.alphas[0],  objective='quantile'),
                          param_grid= self.param_grid_lgbm,
                          cv=TimeSeriesSplit(self.n_folds_gridsearch),
                          scoring=mqloss_scorer_down,
                          n_jobs=-1,
                          verbose=1
    )
    self.model_down_set.append(LGBM_down.fit(X, y).best_estimator_)
    print("LGBM_down.best_params = ", LGBM_down.best_params_)
    self.dico_hyperparametres[groupe]["LGBM"]["down"] =  LGBM_down.best_params_

    print("GridSearchCV LGBM up")
    LGBM_up = GridSearchCV(
                          estimator=LGBMRegressor(alpha=self.alphas[1],  objective='quantile'),
                          param_grid= self.param_grid_lgbm,
                          cv=TimeSeriesSplit(self.n_folds_gridsearch),
                          scoring=mqloss_scorer_up,
                          n_jobs=-1,
                          verbose=1
    )
    self.model_up_set.append(LGBM_up.fit(X, y).best_estimator_)
    print("LGBM_up.best_params = ", LGBM_up.best_params_)
    self.dico_hyperparametres[groupe]["LGBM"]["up"] =  LGBM_up.best_params_

    print("GridSearchCV LGBM median")
    LGBM_median = GridSearchCV(
                          estimator=LGBMRegressor(alpha=0.5,  objective='quantile'),
                          param_grid= self.param_grid_lgbm,
                          cv=TimeSeriesSplit(self.n_folds_gridsearch),
                          scoring=mqloss_scorer_median,
                          n_jobs=-1,
                          verbose=1
    )
    self.model_median_set.append(LGBM_median.fit(X, y).best_estimator_)
    print("LGBM_median.best_params = ", LGBM_median.best_params_)
    self.dico_hyperparametres[groupe]["LGBM"]["median"] =  LGBM_median.best_params_


    print("GridSearch Terminated")

# COMMAND ----------

class aggregation_IC(): 
  def __init__(self,pred_set,y_true):
    from sklearn.preprocessing import MinMaxScaler
    self.scaler = MinMaxScaler()
    self.scaler.fit(pred_set)
    self.normalised = False
    self.coef_factor = np.max(y_true)
    self.pred_set = self.scaler.transform(pred_set)
    self.y = list(y_true)
    self.J = len(pred_set)
    self.len_y = len(y_true)
    #self.normalisation()

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

  def pinball_loss(self,pred,true,alpha):
    return (alpha*(true-pred)*(true-pred >= 0) + (alpha-1)*(true-pred)*(true-pred < 0))
  
  def mse(self,pred,true,alpha = None):
    return(math.sqrt(np.sum([(t-p)**2 for t,p in zip(true,pred)])))

  def boa(self,eta,alpha): #Bernstein online aggregation
    self.eta_boa = eta
    self.pi_t_j = [np.ones(self.J)/self.J]
    self.l_t_j = []
    self.epi = None
    for t in range(self.len_y):
      self.epi = np.dot(self.pi_t_j[-1],[self.loss(self.pred_set[i][t],self.y[t],alpha = alpha) for i in range(self.J)])
      self.l_t_j.append([self.loss(self.pred_set[i][t],self.y[t],alpha = alpha) - self.epi for i in range(self.J)])
      self.regularisation = np.sum([np.exp(-self.eta_boa*self.l_t_j[-1][j]*(1 + self.eta_boa*self.l_t_j[-1][j]))*self.pi_t_j[-1][j] for j in range(self.J)])
      self.pi_t_j.append([np.exp(-self.eta_boa*self.l_t_j[-1][j]*(1 + self.eta_boa*self.l_t_j[-1][j]))*self.pi_t_j[-1][j] / self.regularisation for j in range(self.J)])
      return self.pi_t_j[-1]

  def faboa(self,alpha): #Fully adaptative Bernstein online aggregation with pinball loss
    np.seterr(divide='ignore', invalid='ignore')
    self.pi_t_j = [np.ones(self.J)/self.J]
    self.L_t_j = [np.zeros(self.J)]
    self.n_t_j = [np.zeros(self.J)]
    self.E_t_j = 2*np.ones(self.J)
    self.l_t_j = []
    self.epi = None
    for t in range(self.len_y):
      self.epi = np.dot(self.pi_t_j[-1],[self.pinball_loss(self.pred_set[i][t],self.y[t],alpha = alpha) for i in range(self.J)])
      self.l_t_j.append([self.pinball_loss(self.pred_set[i][t],self.y[t],alpha = alpha) - self.epi for i in range(self.J)])
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
    self.pi_t_j = self.pi_t_j[1:]
    return self.pi_t_j

  def faboamse(self): #Fully adaptative Bernstein online aggregation with mse loss
    np.seterr(divide='ignore', invalid='ignore')
    self.pi_t_j = [np.ones(self.J)/self.J]
    self.L_t_j = [np.zeros(self.J)]
    self.n_t_j = [np.zeros(self.J)]
    self.E_t_j = 2*np.ones(self.J)
    self.l_t_j = []
    self.epi = None
    for t in range(self.len_y):
      self.epi = np.dot(self.pi_t_j[-1],[self.mse(self.pred_set[i][t],self.y[t]) for i in range(self.J)])
      self.l_t_j.append([self.mse(self.pred_set[i][t],self.y[t]) - self.epi for i in range(self.J)])
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
    return self.pi_t_j

def AgACI(self, gammas : list, window = 20): #Aggregative ACI
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # we need to detach h0 and c0 here since we are doing back propagation through time (BPTT)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # we need to detach h0 here since we are doing back propagation through time (BPTT)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out
      
def train(net, x_train, y_train, x_test, y_test, criterion, optimizer):
  print(net)
  net.to(device)
  start_time = time.time()
  hist = []
  for t in range(num_epochs):
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    y_train_pred = net(x_train)
    loss = criterion(y_train_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    hist.append(loss.item())
    if t !=0 and t % 100 == 0 :
      print(' Epoch: {:.6f} \t Training loss: {:.6f} ' .format(t, loss.item()))
      test_loss = criterion(net(x_test.to(device)), y_test.to(device)).item()
      print(' Epoch: {:.6f} \t Test loss: {:.6f} ' .format(t, test_loss))
  training_time = time.time()-start_time
  print("Training time: {}".format(training_time))
  return np.array(hist)

# COMMAND ----------

class Adaptative_Conformal_Inference(Conformal_Inference):
  def __init__(self,alphas,adaptative_window_length,gamma,cal_size,val_size,mode = "normal"):
    Conformal_Inference.__init__(self,alphas, cal_size, test_size = None,mode = mode)
    self.alphas = alphas
    self.confidence = self.alphas[1] - self.alphas[0]
    self.cal_size = cal_size
    self.val_size = val_size
    self.mode = mode
    self.adaptative_window_length = adaptative_window_length
    self.gamma = gamma
    self.adaptative_window_length = adaptative_window_length
    self.adaptative = True

  def predict(self,X_predict = None,plot = False):
    if self.fitted == False:
      print("please fit the model")

    self.pred_down_predict,self.pred_up_predict,self.pred_median_predict = self.predict_test(X_predict)
    #initialize adaptative alphas
    alpha_star_down_updated = self.alphas[0]
    alpha_star_up_updated = self.alphas[1]
    alpha_star_median_updated = 0.5
    pred_down_adaptative = self.pred_down_cal[-len(X_predict):].copy()
    pred_up_adaptative = self.pred_up_cal[-len(X_predict):].copy()
    pred_median_adaptative = self.pred_median_cal[-len(X_predict):].copy()
    y_cal_increased = self.y_cal[-len(X_predict):].copy()
    y_val = self.y_val.copy()
    #pred_up_val = self.pred_up_val.copy()
    #pred_down_val = self.pred_down_val.copy()
    pred_median_cal = self.pred_median_cal.copy()
    pred_median_val = self.pred_median_val.copy()
    
    self.alpha_star_down_updated_list = []
    self.alpha_star_up_updated_list = []
    self.alpha_star_median_updated_list = []
  
    #init alpha star
    try:
      alpha_star_down_updated = np.max([b for b in np.arange(0,1,0.0001) if (self.f_miscoverage_rate_down(self.pred_down_cal,self.pred_down_val,self.y_cal,self.y_val,alpha = b) < self.alphas[0])])
      alpha_star_up_updated = np.max([b for b in np.arange(0,1,0.0001) if (self.f_miscoverage_rate_up(self.pred_up_cal,self.pred_up_cal,self.y_cal,self.y_val,alpha = b) < 1 - self.alphas[1])])
      alpha_star_median_updated = np.max([b for b in np.arange(0,1,0.0001) if (self.f_miscoverage_rate_median(self.pred_median_cal,self.pred_median_val,self.y_cal,self.y_val,alpha = b) < 0.5)])
    except ValueError:
      print("no quantile found to cover the shift")
      print("setting default value")
      alpha_star_down_updated = 0.05
      alpha_star_up_updated = 0.95
      alpha_star_median_updated = 0.5
    
    for it in range(len(self.X_val)):
      #compute alpha star updated
      alpha_star_up_updated = alpha_star_up_updated + self.gamma*(1 - self.alphas[1] - np.sum([self.err_t_up(y_cal_increased,pred_up_adaptative,w)*math.exp(-w) for w in range(self.adaptative_window_length)])/np.sum([math.exp(k) for k in range(self.adaptative_window_length)]))
      alpha_star_down_updated = alpha_star_down_updated + self.gamma*(self.alphas[0] - np.sum([self.err_t_down(y_cal_increased,pred_down_adaptative,w)*math.exp(-w) for w in range(self.adaptative_window_length)])/np.sum([math.exp(k) for k in range(self.adaptative_window_length)]))
      alpha_star_median_updated = alpha_star_median_updated + self.gamma*(0.5 - np.sum([self.err_t_median(y_cal_increased,pred_median_adaptative,w)*math.exp(-w) for w in range(self.adaptative_window_length)])/np.sum([math.exp(k) for k in range(self.adaptative_window_length)]))
      self.alpha_star_down_updated_list.append(alpha_star_down_updated)
      self.alpha_star_up_updated_list.append(alpha_star_up_updated)
      self.alpha_star_median_updated_list.append(alpha_star_median_updated)

      if 0<alpha_star_up_updated <1:
        pred_up_adaptative = np.append(pred_up_adaptative,self.pred_up_val[it] + np.quantile(self.f_conformity_score_up(self.y_cal,self.pred_up_cal),1-alpha_star_up_updated))
      elif alpha_star_up_updated <= 0:
        pred_up_adaptative = np.append(pred_up_adaptative,self.pred_up_val[it] + np.min(self.f_conformity_score_up(self.y_cal,self.pred_up_cal)))
        alpha_star_up_updated = 0.1
        print(f" i = {it} alpha_star_up_updated < 0")
      elif alpha_star_up_updated >=1:
        pred_up_adaptative = np.append(pred_up_adaptative,self.pred_up_val[it] + np.max(self.f_conformity_score_up(self.y_cal,self.pred_up_cal)))
        alpha_star_up_updated = 0.9
        print(f" i = {it} alpha_star_up_updated >1")
      #lower bound 
      if 0<alpha_star_down_updated <1:
        pred_down_adaptative = np.append(pred_down_adaptative,self.pred_down_val[it] - np.quantile(self.f_conformity_score_down(self.y_cal,self.pred_up_cal),1-alpha_star_down_updated))
      elif alpha_star_down_updated <= 0:
        pred_down_adaptative = np.append(pred_down_adaptative,self.pred_down_val[it] - np.max(self.f_conformity_score_down(self.y_cal,self.pred_up_cal)))
        alpha_star_down_updated = 0.001
        print(f" i = {it} alpha_star_down_updated < 0")
      elif alpha_star_down_updated >=1:
        pred_down_adaptative = np.append(pred_down_adaptative,self.pred_down_val[it] - np.min(self.f_conformity_score_down(self.y_cal,self.pred_up_cal)))
        alpha_star_down_updated = 0.999
        print(f" i = {it} alpha_star_down_updated >1")
      #Median 
      if 0<alpha_star_median_updated <1:
        pred_median_adaptative = np.append(pred_median_adaptative,self.pred_median_val[it] - np.quantile(self.f_conformity_score_median(self.y_cal,self.pred_up_cal),1-alpha_star_median_updated))
      elif alpha_star_median_updated <= 0:
        pred_median_adaptative = np.append(pred_median_adaptative,self.pred_median_val[it] - np.max(self.f_conformity_score_median(self.y_cal,self.pred_up_cal)))
        alpha_star_median_updated = 0.1
        print(f" i = {it} alpha_star_down_updated < 0")
      elif alpha_star_median_updated >=1:
        pred_median_adaptative = np.append(pred_median_adaptative,self.pred_median_val[it] - np.min(self.f_conformity_score_median(self.y_cal,self.pred_up_cal)))
        alpha_star_median_updated = 0.9
        print(f" i = {it} alpha_star_down_updated >1")

      y_cal_increased = np.append(y_cal_increased,self.y_val[it])
      y_cal_increased = np.delete(y_cal_increased,0)
      pred_up_adaptative = np.delete(pred_up_adaptative,0)
      pred_down_adaptative = np.delete(pred_down_adaptative,0)
      pred_median_adaptative = np.delete(pred_median_adaptative,0)
      print(it , "alpha_star_down_updated : ",alpha_star_down_updated)
      print(it , "alpha_star_up_updated : ",alpha_star_up_updated)
      print(it , "alpha_star_median_updated : ",alpha_star_median_updated)

    if self.mode =="normal":
      pred_down_adaptative = self.pred_down_predict - np.quantile(self.f_conformity_score_down(self.y_cal,self.pred_up_cal), 1-alpha_star_down_updated)
      pred_up_adaptative = self.pred_up_predict - np.quantile(self.f_conformity_score_up(self.y_cal,self.pred_up_cal),1-alpha_star_up_updated)
      pred_median_adaptative = self.pred_median_predict - np.quantile(self.f_conformity_score_median(self.y_cal,self.pred_up_cal),1-alpha_star_median_updated)

    if plot :
      if self.mode == "test":
        fig = go.Figure()
        fig.add_trace(go.Scatter( y=pred_up_adaptative,
                        mode='lines',
                        name=f'q_{alphas[1]}',
                        line=dict(
                            color='rgb(0, 256, 0)',
                            width=0),
                        showlegend = False))
        
        fig.add_trace(go.Scatter( y=pred_down_adaptative,
                        mode='lines',
                        name=f'{int(np.round(100*(self.alphas[1]-self.alphas[0])))}% Confidence Interval',
                        line=dict(
                            color='rgb(0, 256, 0)',
                            width=0),
                        fill='tonexty',
                        fillcolor='rgba(0,176,246,0.2)',
                        line_color='rgba(255,255,255,0)'))
        
        fig.add_trace(go.Scatter( y=pred_median_adaptative,
                      mode='lines',
                      name=f'y_median',
                      line=dict(
                          color='rgb(256,0, 0)',
                          width=1),
                      showlegend = True))
        
        fig.add_trace(go.Scatter( y=self.y_val,
                      mode='lines',
                      name=f'y_true',
                      line=dict(
                          color='rgb(0,0, 256)',
                          width=1),
                      showlegend = True))
        error = (np.sum(self.y_val<pred_down_adaptative) + np.sum(self.y_val>pred_up_adaptative))/len(self.y_val)
        fig.update_layout(title = f"Test : {(1-error)*100}% Confidence Interval Prediction")

      if self.mode == "normal":
        fig = go.Figure()
        fig.add_trace(go.Scatter( y=pred_up_adaptative,
                        mode='lines',
                        name=f'q_{alphas[1]}',
                        line=dict(
                            color='rgb(0, 256, 0)',
                            width=0),
                        showlegend = False))
        
        fig.add_trace(go.Scatter( y=pred_down_adaptative,
                        mode='lines',
                        name=f'{int(np.round(100*(self.alphas[1]-self.alphas[0])))}% Confidence Interval',
                        line=dict(
                            color='rgb(0, 256, 0)',
                            width=0),
                        fill='tonexty',
                        fillcolor='rgba(0,176,246,0.2)',
                        line_color='rgba(255,255,255,0)'))
        
        fig.add_trace(go.Scatter( y=pred_median_adaptative,
                      mode='lines',
                      name=f'y_median',
                      line=dict(
                          color='rgb(256,0, 0)',
                          width=1),
                      showlegend = True))
        fig.update_layout(title = f"Predict  adaptative Interval Prediction")
             
      fig.update_traces(mode='lines')                 
      fig.show()  
    print(len(X_predict))
    return pred_down_adaptative,pred_up_adaptative

  def err_t(self,y_true,pred_up,pred_down,t):
    """
    Compute the adaptative error of adaptative conformal inference at time t
    """
    return (list(y_true)[-t]>list(pred_up)[-t] or list(y_true)[-t]<list(pred_down)[-t])

  def err_t_down(self,y_true,pred_down,t):
    """
    Compute the adaptative error of asymetric adaptative conformal inference for down bound at time t
    """
    return list(y_true)[-t]<=list(pred_down)[-t]

  def err_t_up(self,y_true,pred_up,t):
    """
    Compute the adaptative error of asymetric adaptative conformal inference for upper down bound at time t
    """
    return list(y_true)[-t]>=list(pred_up)[-t]
  
  def err_t_median(self,y_true,pred_median,t):
    """
    Compute the adaptative error of asymetric adaptative conformal inference for down bound at time t
    """
    return list(y_true)[-t]<=list(pred_median)[-t]

# COMMAND ----------

class faci(Adaptative_Conformal_Inference): #Fully adaptative conformal inference
  def __init__(self,alphas,adaptative_window_length,gamma,cal_size,val_size,mode = "normal"):
    Conformal_Inference.__init__(self,alphas, cal_size , val_size ,test_size = None,mode = mode)
    self.gammas = np.arange(0.001,1,0.001)
    self.alphas_i = [np.arange(0.001,1,0.001)]
    
  def FACI_up(self,sigma, eta):
    self.p_values = [np.max([b for b in np.arange(0,1,0.01) if (self.pred_up_val[t] + np.quantile(self.f_conformity_score_high(self.pred_up_cal,self.y_cal),1-b))>list(self.y_val)[t] ]) for t in range(len(self.pred_up_val))]
    k = len(self.pvalues)
    omegas = np.ones(len(gammas_k))
    output = []
    for t in range(k):
      p = omegas / np.sum(omegas)
      alpha_barre = np.dot(p,self.alphas_i)
      output.append(alpha_barre)
      omegas_barre = []
      for i in range(len(alphas_i)):
        omegas_barre.append(omegas[i]*np.exp(-eta*(self.alphas[1]*(self.pvalues[t]-self.alphas_i[i]) - min(0,self.pvalues[t]-self.alphas_i[i]))))
      omegas = (np.multiply((1-sigma),omegas_barre) + np.sum(omegas_barre)*sigma/k)
      err_t_i = [(self.pred_up_cal[t] + np.quantile(self.f_conformity_score_high(self.pred_up_cal,y_tcal),1-alpha_i))>list(y_cal)[t] for alpha_i in self.alphas_i]
      for i in range(len(self.alphas_i)):
        self.alphas_i[i] = max(0,min(1,self.alphas_i[i] + self.gammas[i]*(self.alphas[1] - err_t_i[i])))
    return output

# COMMAND ----------

class AgACI(Conformal_Inference):
  def __init__(self,alphas,adaptative_window_length,cal_size,val_size,gammas : list,mode = "normal"):
    Conformal_Inference.__init__(self,alphas, cal_size , val_size ,test_size = None,mode = mode)
    self.gammas = gammas
  def predict(self,X_test):
    #Initialisation 
    cal_test_iterative = self.X_cal
    for x_test_i in X_test:
      low_t = []
      high_t = []
      omega_t_low = []
      omega_t_high = []
      alpha_star_low_updated = np.max([b for b in np.arange(0,1,0.0001) if (self.f_miscoverage_rate_down(self.pred_down_cal,self.pred_down_val,self.y_cal,self.y_val,alpha = b) < self.alphas[0])])
      alpha_star_high_updated = np.max([b for b in np.arange(0,1,0.0001) if (self.f_miscoverage_rate_up(self.pred_up_cal,self.pred_up_val,self.y_cal,self.y_val,alpha = b) < 1 - self.alphas[1])])
      alpha_star_median = np.max([b for b in np.arange(0,1,0.0001) if (self.f_miscoverage_rate_median(self.pred_median_cal,self.pred_median_val,self.y_cal,self.y_val,alpha = b) < 0.5)])
      pred_down_test,pred_up_test,pred_median_test = super().predict_test(X_test)
      low = pred_down_test - np.quantile(self.f_conformity_score_down(self.pred_down_cal,self.y_cal),1-self.alphas[0])
      high = pred_up_test + np.quantile(self.f_conformity_score_up(self.pred_up_cal,self.y_cal),1-self.alphas[1])
      med = pred_median_test + np.quantile(self.f_conformity_score_median(self.pred_median_cal,self.y_cal),0.5)

      for gamma in self.gammas:
        alpha_star_low_upd = alpha_star_low_updated + gamma*(self.alphas[0] - np.mean([self.err_t_low(y_test_1,pred_down_test_1,i) for i in range(adaptative_window_length)]))
        if 0<alpha_star_low_upd <1:
          low = pred_down_test - np.quantile(self.f_conformity_score_low(self.pred_down_val,self.y_val,1-alpha_star_low_upd))
        else :
          print("alpha_star_low_updated out of 0-1")
        alpha_star_high_upd = alpha_star_high_updated + gamma*(1-self.alphas[1] - np.mean([err_t_high(y_test_1,pred_up_test_1,i) for i in range(adaptative_window_length)]))
        if 0<alpha_star_high_upd <1:
          high = pred_up_test_2 + np.quantile(f_conformity_score_high(pred_up_test_1,y_test_1),1-alpha_star_high_upd)
        else :
          print("alpha_star_high_updated out of 0-1")
        low_t.append(low)
        high_t.append(high)
      agup = aggregation_IC(high_t)
      agdown = aggregation_IC(low_t)
      agup.faboa()
      agdown.faboa()
      C_low = np.mean(low_t,axis = 0)
      C_high = np.mean(high_t,axis = 0)
    return C_low,C_high

# COMMAND ----------

class EnbPi:
  def __init__(self,alpha,n_bootstrap : int, batch_size : int = 84):
    self.n_bootstrap = n_bootstrap
    self.batch_size = batch_size
    self.alpha = alpha
    self.model = XGBRegressor()

  def fit(self,X_train,y_train):
    self.X_train=X_train
    self.y_train = y_train
    self.T = len(self.X_train)
    self.len_train = len(self.X_train)
    self.S_b = []
    self.f_b = []
    
    for b in range(self.n_bootstrap):
      print("1/3 Train 1/2 : ",int(100*b/self.n_bootstrap),"%")
      self.S_b.append(np.random.choice(list(self.X_train.index),self.T,replace = True))
      self.f_b.append(self.model.fit(self.X_train.loc[self.S_b[-1]],np.take(self.y_train, self.S_b[-1])))
    self.eps =[]
    self.f_phi_i = []
    for i in range(self.len_train):
      print("2/3 train 2/2 : ",int(100*i/self.len_train),"%")
      self.f_phi_temp = []
      for bi,b in enumerate(self.S_b):
        if i not in b:
          self.f_phi_temp.append(self.f_b[bi].predict(self.X_train.iloc[i:i+1]))
        self.f_phi_i.append(np.mean(self.f_phi_temp))
      self.eps_phi_i = np.abs(self.y_train[i]-self.f_phi_i[i])
      if math.isnan(self.eps_phi_i) == False:
        self.eps.append(self.eps_phi_i)

  def predict(self,X_test):
    self.X_test = X_test
    self.len_test = len(self.X_test)
    self.X_test = X_test
    self.C_t_low = []
    self.C_t_high = []
    self.f_phi_t = [] 
    for t in range(len(self.X_test)):
      print(t,"/",self.len_test)
      print(int(100*t/self.len_test),"%")
      self.f_phi_temp = []
      for bi,b in enumerate(self.S_b):
        if t not in b:
          self.f_phi_temp.append(self.f_b[bi].predict(self.X_test.iloc[t:t+1]))
      print("eps : ",self.eps)
      self.f_phi_t.append(np.quantile(self.f_phi_temp,1-self.alpha))
      self.w_phi_t = np.quantile(self.eps,1-self.alpha)
      print("f_phi_t : ",self.f_phi_t)
      print("w_phi_t : ",self.w_phi_t)
      self.C_t_low.append(self.f_phi_t - self.w_phi_t)
      self.C_t_high.append(self.f_phi_t + self.w_phi_t)
      """
      if t % self.batch_size == 0:
        for j in range(self.batch_size):
          self.e_phi_j = float(np.abs(self.y_test[j]-self.f_phi_t[j]))
          print("e_phi_j",self.e_phi_j)
          print("type e_phi_j",type(self.e_phi_j))
          print("eps_preappend :", len(self.eps))
          print("type eps preappend :", type(self.eps))
          self.eps.append(self.e_phi_j)
          print("eps_post_append " , len(self.eps))
          del self.eps[0]
          print("eps_postdel :", len(self.eps))
      """
    return self.C_t_low, self.C_t_high

# COMMAND ----------

class Conformal_Inference_cuted():
  def __init__(self,alphas, cal_size,test_size = None,mode = "normal"):
    self.alphas = alphas
    self.confidence = self.alphas[1] - self.alphas[0]
    self.cal_size = cal_size
    self.test_size = test_size
    self.fitted = False
    self.mode = mode
    self.adaptative = False
    self.dico = {}

  def fit(self,df_train):
    self.target_train = df_train["Valeur"]
    self.df_train = df_train
    #init models
    self.labels = set(df_train["label"])
    self.dico = {}
    if self.mode == "test":
      df_train, self.X_test, self.target_train, self.y_test = train_test_split(df_train, self.target_train, test_size = 0.1, shuffle = False)
      self.xX_test = self.X_test["DT_VALR"]

    for label in self.labels:
      self.dico[label] = {}
      """
      Fit the model
      """
      if self.mode == "normal" or self.adaptative: #Créer disctionnaire
        self.dico[label]["xX_train"], self.dico[label]["xX_cal"], self.dico[label]["X_train"], self.dico[label]["X_cal"], self.dico[label]["y_train"], self.dico[label]["y_cal"] = self.train_cal_val_split(self.df_train[self.df_train["label"] == label])
      if self.mode == "test" and not self.adaptative:
        self.dico[label]["xX_train"], self.dico[label]["xX_cal"], self.dico[label]["xX_test"], self.dico[label]["X_train"], self.dico[label]["X_cal"], self.dico[label]["X_test"], self.dico[label]["y_train"], self.dico[label]["y_cal"], self.dico[label]["y_test"] = self.train_cal_test_split(self.df_train[self.df_train["label"] == label])

      
      self.dico[label]["model"] = IC_model(self.alphas)
      self.dico[label]["model"].fit(self.dico[label]["X_train"],self.dico[label]["y_train"])
    self.predict_cals()
    self.fitted = True

  def train_cal_split(self,df):
    "split data into train, calibration and validation set, used in normal mode"
    target = df["Valeur"]
    X_train, X_cal , y_train, y_cal = train_test_split(df, target, test_size = self.cal_size, shuffle = False)
    xX_train = X_train["DT_VALR"]
    X_train = X_train.drop(["DT_VALR","Valeur"],axis = 1)
    xX_cal = X_cal["DT_VALR"]
    X_cal = X_cal.drop(["DT_VALR","Valeur"], axis = 1)
    y_train = list(y_train)
    y_cal = list(y_cal)
    return xX_train, xX_cal, X_train, X_cal,  y_train, y_cal,

  def train_cal_test_split(self,df):
    """
    split data into train, calibration validation and test set, used in test mode
    return self.xX_train, self.X_train, self.y_train, self.xX_cal, self.X_cal, self.y_cal, self.xX_val,self.X_val, self.y_val, self.xX_test, self.X_test, self.y_test
    """
    if self.test_size == None :
      self.test_size = self.cal_size
      print("test_size = ", self.test_size)
    target = df["Valeur"]
    X_train, X_test , y_train, y_test = train_test_split(df, target, test_size = self.test_size, shuffle = False)
    X_train, X_cal , y_train, y_cal = train_test_split(X_train, y_train, test_size = self.cal_size, shuffle = False)
    xX_train = X_train["DT_VALR"]
    X_train = X_train.drop(["DT_VALR","Valeur"],axis = 1)
    xX_cal = X_cal["DT_VALR"]
    X_cal = X_cal.drop(["DT_VALR","Valeur"], axis = 1)
    xX_test = X_test["DT_VALR"]
    X_test = X_test.drop(["DT_VALR","Valeur"], axis = 1)
    y_train = list(y_train)
    y_cal = list(y_cal)
    return xX_train, xX_cal, xX_test, X_train, X_cal, X_test, y_train, y_cal, y_test

  def predict_cals(self):
    "predict on calibration and validation set"
    #self.model_down_set = []
    #self.model_up_set = []
    for label in self.labels:
      pred_down_cal_set = []
      pred_up_cal_set = []
      pred_median_cal_set = []
                    
      for i in range(len(self.dico[label]["model_down_set"])):
        pred_down_cal_set.append(self.dico[label]["model_down_set"][i].predict(self.dico[label]["X_cal"]))
        pred_up_cal_set.append(self.dico[label]["model_up_set"][i].predict(self.dico[label]["X_cal"]))
        pred_median_cal_set.append(self.dico[label]["model_median_set"][i].predict(self.dico[label]["X_cal"]))

      #Aggregation
      self.dico[label]["pred_down_cal"] = np.mean(pred_down_cal_set)
      self.dico[label]["pred_up_cal"] = np.mean(pred_up_cal_set)
      self.dico[label]["pred_median_cal"] = np.mean(pred_median_cal_set)

  def predict_test(self,X):
    "predict on predict_set"
    pred_down = []
    pred_up = []
    pred_median = []
    try:
      X = X.drop(["DT_VALR","Valeur"],axis = 1)
    except:
      pass
    for i,label in enumerate(X["label"]):
      pred = self.dico[label]["model"].predict(X.iloc[i:i+1])
      pred_down_set = []
      pred_up_set = []
      pred_median_set = []
      for _ in range(len(self.dico[label]["model_down_set"])):
        pred_down_set.append(self.dico[label]["model_down_set"][_].predict(X.iloc[i:i+1]))
        pred_up_set.append(self.dico[label]["model_up_set"][_].predict(X.iloc[i:i+1]))
        pred_median_set.append(self.dico[label]["model_median_set"][_].predict(X.iloc[i:i+1]))
      pred_down = pred[0]
      pred_up = pred[1]
      pred_median = pred[2]
    return(pred_down,pred_up,pred_median)

  def predict(self,X_predict = None,plot = False):
    """
    Compute Asymetric Conformal Inference for predict set,
    X_predict = None,plot = False
    return self.pred_down_predict_asymetric_conformal,self.pred_up_predict_asymetric_conformal
    """
    self.X_predict = X_predict
    if self.mode == "test":
      self.X_predict = self.X_test
    self.pred_down_predict,self.pred_up_predict,self.pred_median_predict = self.predict_test(self.X_predict)

    self.pred_down_predict_asymetric_conformal = []
    self.pred_up_predict_asymetric_conformal = []
    self.pred_median_predict_asymetric_conformal = []

    for i,label in enumerate(self.X_predict["label"]):

      try:
        self.pred_down_predict_asymetric_conformal.append((self.pred_down_predict[i] - np.quantile(self.f_conformity_score_down(self.dico[label]["pred_down_cal"],self.dico[label]["y_cal"]),1-self.alphas[0]))[0])
      except Exception as e:
        print( "pred_down conformal exception : ", e)
        self.pred_down_predict_asymetric_conformal.append(self.pred_down_predict[i][0])

      try:
        self.pred_up_predict_asymetric_conformal.append((self.pred_up_predict[i] + np.quantile(self.f_conformity_score_up(self.dico[label]["pred_up_cal"],self.dico[label]["y_cal"]),self.alpha[1]))[0])
      except Exception as e:
        print( "pred_up conformal exception : ", e)
        self.pred_up_predict_asymetric_conformal.append(self.pred_up_predict[i][0])

      try:
        self.pred_median_predict_asymetric_conformal.append((self.pred_median_predict[i] - np.quantile(self.f_conformity_score_median(self.dico[label]["pred_median_cal"],self.dico[label]["y_cal"]),0.5))[0])
      except Exception as e:
        print( "pred_median conformal exception : ", e)
        self.pred_median_predict_asymetric_conformal.append(self.pred_median_predict[i][0])

    
    if plot :
      #Show plot pred test 
      if self.mode == "normal":
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
        fig.add_trace(go.Scatter( y=self.pred_median_predict_asymetric_conformal,
                      mode='lines',
                      name=f'y_median',
                      line=dict(
                          color='rgb(256,0, 0)',
                          width=1),
                      showlegend = True))
        fig.update_layout(title = f"{int(np.round(100*self.confidence))}% Asymetric conformal prediction interval")
        fig.show()

      if self.mode == "test":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.xX_test, y=self.y_test,
                      mode='lines',
                      name=f'y_true',
                      line=dict(
                          color='rgb(0,0, 256)',
                          width=1),
                      showlegend = True))
                
        fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_up_predict_asymetric_conformal,
                        mode='lines',
                        name=f'q_{alphas[1]}',
                        line=dict(
                            color='rgb(0, 256, 0)',
                            width=0),
                        showlegend = False))

        fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_down_predict_asymetric_conformal,
                        mode='lines',
                        name=f'{int(np.round(100*(self.alphas[1]-self.alphas[0])))}% Confidence Interval',
                        line=dict(
                            color='rgb(0, 256, 0)',
                            width=0),
                        fill='tonexty',
                        fillcolor='rgba(0,176,246,0.2)',
                        line_color='rgba(255,255,255,0)'))
        
        fig.add_trace(go.Scatter(x=self.xX_test, y=self.pred_median_predict_asymetric_conformal,
                      mode='lines',
                      name=f'y_median',
                      line=dict(
                          color='rgb(256,0, 0)',
                          width=1),
                      showlegend = True))
        
        error = (np.sum(np.array(self.y_test)<np.array(self.pred_down_predict_asymetric_conformal)) + np.sum(np.array(self.y_test)>np.array(self.pred_up_predict_asymetric_conformal)))/len(self.y_test)
        fig.update_traces(mode='lines')
        fig.update_layout(title = f"Test : {1-error}% asymetric conformal prediction test")                      
        fig.show()

    return self.pred_down_predict_asymetric_conformal,self.pred_up_predict_asymetric_conformal, self.pred_median_predict_asymetric_conformal

  def f_conformity_score(self,pred_down_cal,pred_up_cal,y_cal):
    """
    Compute the symetric conformity score
    """
    return np.max([pred_down_cal-y_cal,y_cal-pred_up_cal],axis = 0)
  
  def f_conformity_score_down(self,pred_down_cal,y_cal):
    """
    Compute the asymetric conformity score for down bound
    """
    return [pred_down_cal-y_cal]

  def f_conformity_score_up(self,pred_up_cal,y_cal):
    """
    Compute the asymetric conformity score for upper bound
    """
    return [y_cal-pred_up_cal]
  
  def f_conformity_score_median(self,pred_median_cal,y_cal):
    """
    Compute the asymetric conformity score for upper bound
    """
    return [pred_median_cal-y_cal]
  
  def f_miscoverage_rate(self,pred_down_cal,pred_up_cal,pred_down_val,pred_up_val,y_cal,y_val,alpha):
    """
    Compute the miscoverage rate
    """
    csq = np.quantile(self.f_conformity_score(pred_down_cal,pred_up_cal,y_cal),1-alpha)
    return(np.sum(np.max([pred_down_val-y_val,y_val-pred_up_val],axis = 0)>csq)/len(y_val))

  def f_miscoverage_rate_down(self,pred_down_cal,pred_down_val,y_cal,y_val,alpha):
    """
    Compute the asymetric miscoverage rate for down bound
    """
    csq_low = np.quantile(self.f_conformity_score_down(pred_down_cal,y_cal),1-alpha)
    return(np.sum((pred_down_val-y_val)>csq_low)/len(y_val))
  
  def f_miscoverage_rate_median(self,pred_median_cal,pred_median_val,y_cal,y_val,alpha):
    """
    Compute the asymetric miscoverage rate for median bound
    """
    csq_median = np.quantile(self.f_conformity_score_median(pred_median_cal,y_cal),1-alpha)
    return(np.sum((pred_median_val-y_val)>csq_median)/len(y_val))

  def f_miscoverage_rate_up(self,pred_up_cal,pred_up_val,y_cal,y_val,alpha):
    """
    Compute the asymetric miscoverage rate for upper bound
    """
    csq_up = np.quantile(self.f_conformity_score_up(pred_up_cal,y_cal),1 - alpha)
    return(np.sum((y_val-pred_up_val)>csq_up)/len(y_val))

# COMMAND ----------

def integral_score(up,down): #Calcule en millions la moyenne de la taille d'un intervalle
  return(np.sum(np.abs(np.array(up)-np.array(down))/(1000000*len(np.array(down)))))

# COMMAND ----------

class EnbPi_quantile: #Asymetric quantile bootstrap method (homemade)
  def __init__(self,alphas,n_bootstrap : int, batch_size : int = 84):
    self.n_bootstrap = n_bootstrap
    self.batch_size = batch_size
    self.alphas = alphas

  def fit(self,X_train,y_train):
    self.X_train=X_train
    self.y_train = y_train
    self.T = len(self.X_train)
    self.len_train = len(self.X_train)
    self.S_b = [] #liste des indices des echantillons bootstrap
    self.fb = []
    
    for b in range(self.n_bootstrap):
      print("1/3 Train 1/2 : ",int(100*b/self.n_bootstrap),"%")
      self.S_b.append(np.random.choice(list(self.X_train.index),self.T,replace = True)) #création de la liste des echantillons bootstrap
      self.fb.append(IC_model(self.alphas)) #initialisation des modèles
      self.fb[-1].fit(self.X_train.loc[self.S_b[-1]],np.take(self.y_train, self.S_b[-1])) #entrainement des modèles
      #self.f_b_up.append(self.model_up.fit(self.X_train.loc[self.S_b[-1]],np.take(self.y_train, self.S_b[-1])))
      #self.f_b_down.append(self.model_down.fit(self.X_train.loc[self.S_b[-1]],np.take(self.y_train, self.S_b[-1])))
      #self.f_b_median.append(self.model_median.fit(self.X_train.loc[self.S_b[-1]],np.take(self.y_train, self.S_b[-1])))
    self.eps_up =[] #liste des erreurs pour la borne superieure
    self.eps_down =[] #liste des erreurs pour la borne inferieure
    self.eps_median =[] #liste des erreurs pour la borne médiane
    self.f_phi_i_up = [] #initialisation de la liste des aggregat de predictions pour la borne supérieure
    self.f_phi_i_down = [] #initialisation de la liste des aggregat de predictions pour la borne inférieure
    self.f_phi_i_median = [] #initialisation de la liste des aggregat de predictions pour la médiane
    for i in range(self.len_train):
      print("2/3 train 2/2 : ",int(100*i/self.len_train),"%")
      self.f_phi_temp_up = [] #initialisation des predictions temporaires ( non agrégées )
      self.f_phi_temp_down = []  #initialisation des predictions temporaires ( non agrégées )
      self.f_phi_temp_median = []  #initialisation des predictions temporaires ( non agrégées )
      for bi,b in enumerate(self.S_b): 
        if i not in b: #selection des modèles n'ayant pas été entrainé sur ces valeurs
          #self.f_phi_temp_up.append(self.f_b_up[bi].predict(self.X_train.iloc[i:i+1]))
          #self.f_phi_temp_down.append(self.f_b_down[bi].predict(self.X_train.iloc[i:i+1]))
          #self.f_phi_temp_median.append(self.f_b_median[bi].predict(self.X_train.iloc[i:i+1]))
          predict_b = self.fb[bi].predict(self.X_train.iloc[i:i+1])  #predictions par les modèles n'ayant pas été entrainé sur ces valeurs
          self.f_phi_temp_up.append(predict_b[1]) 
          self.f_phi_temp_down.append(predict_b[0])
          self.f_phi_temp_median.append(predict_b[2])
        self.f_phi_i_up.append(np.mean(self.f_phi_temp_up)) # aggregation des prédictions
        self.f_phi_i_down.append(np.mean(self.f_phi_temp_down))  # aggregation des prédictions
        self.f_phi_i_median.append(np.mean(self.f_phi_temp_median))  # aggregation des prédictions
      self.eps_phi_i_down = self.f_phi_i_down[i] - self.y_train[i] #Calcul des erreurs
      self.eps_phi_i_up = self.y_train[i] - self.f_phi_i_up[i]  #Calcul des erreurs
      self.eps_phi_i_median = self.y_train[i]-self.f_phi_i_median[i]  #Calcul des erreurs
      if not np.isnan(self.eps_phi_i_up):
        self.eps_up.append(self.eps_phi_i_up)
      if not np.isnan(self.eps_phi_i_down):
        self.eps_down.append(self.eps_phi_i_down)
      if not np.isnan(self.eps_phi_i_median):
        self.eps_median.append(self.eps_phi_i_median)

  def predict(self,X_test,plot = False):
    self.X_test = X_test
    self.len_test = len(self.X_test)
    self.X_test = X_test
    self.C_t_up = []
    self.C_t_down = []
    self.C_t_median = []
    self.f_phi_t_up = [] 
    self.f_phi_t_down = [] 
    self.f_phi_t_median = [] 
    for t in range(len(self.X_test)):
      print(t,"/",self.len_test)
      print(int(100*t/self.len_test),"%")
      self.f_phi_temp_up = []
      self.f_phi_temp_down = []
      self.f_phi_temp_median = []
      self.f_phi = []
      for bi,b in enumerate(self.S_b):
        if t not in b:
          predict = self.fb[bi].predict(self.X_test.iloc[t:t+1])
          #self.f_phi_temp_up.append(self.f_b_up[bi].predict(self.X_test.iloc[t:t+1]))
          #self.f_phi_temp_down.append(self.f_b_down[bi].predict(self.X_test.iloc[t:t+1]))
          #self.f_phi_temp_median.append(self.f_b_median[bi].predict(self.X_test.iloc[t:t+1]))
          self.f_phi_temp_down.append(predict[0])
          self.f_phi_temp_up.append(predict[1])
          self.f_phi_temp_median.append(predict[2])
      self.C_t_down.append(np.mean(self.f_phi_temp_down) - np.quantile(self.eps_down,1-self.alphas[0]))
      self.C_t_up.append(np.mean(self.f_phi_temp_up) + np.quantile(self.eps_up,self.alphas[1]))
      self.C_t_median.append(np.mean(self.f_phi_temp_median) -  np.quantile(self.eps_up,0.5))
    return self.C_t_down, self.C_t_up, self.C_t_median

# COMMAND ----------

def mqloss(y_true, y_pred, alpha):  
  if (alpha > 0) and (alpha < 1):
    residual = y_true - y_pred 
    return np.mean(residual * (alpha - (residual<0)))
  else:
    return np.nan