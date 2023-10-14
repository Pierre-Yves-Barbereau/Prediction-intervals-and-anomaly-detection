# Databricks notebook source
class Dataloader():
  """ 
    load data from path_notebook_preprocessing path
  """
  def __init__(self,path_notebook_preproc_preprocessing):
    self.path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing #Set path notebook
    self.data_jour= spark.sql("select * from jours_target_csv")  #download target days
    """
                      .withColumn("Date", to_timestamp("Date", "yyyy-MM-dd")) \
                      .drop("_c5", 'Jour_target_UK', 'Jour_target_US').toPandas()
    self.data_jour["Is Weekend"] = (self.data_jour["Date"].dt.day_name() == "Sunday").values + (self.data_jour["Date"].dt.day_name() == "Saturday").values # ad is weeknd in data_jour
    """
  def load_train_predict(self,groupe : str) :
    """ 
    load dataset of variables used for prediciton
    """
    df_predict = self.load_predict(groupe) #load predict dataset
    df_train = self.load_train(groupe) #load train dataset  
    
    return(df_train, df_predict) 

  def load_train(self,groupe : str):
    """ 
    load dataset used for training
    return df_train of groupe
    """
    path_df_train = "/dbfs/tmp/pre_proc_ml/train/" + groupe + ".sav" #set path for train dataset
    df_group = lib_instance.get_only_groupe_from_jointure(groupe) 
    df_group = df_group[df_group['DT_VALR'] >= "2017-1-1"] 
    df_group.reset_index(inplace = True)   #reset index
    path_train = dbutils.notebook.run(self.path_notebook_preproc_preprocessing,
                            600,
                            {"path_dataframe": path_df_train,
                             'groupe': groupe,
                             'setter_train': 1,
                             'debutDate': "2017-01-01"
                             })
    pickle_in_train = open(path_train, 'rb')
    df_train = pickle.load(pickle_in_train) #download df_trian
    df_train['DT_VALR'] = df_group['DT_VALR']
    nnans = np.sum(np.sum(df_train.isna()))
    if nnans != 0:
      print(f"{nnans} Nans Detected in df_train_{groupe}")
    df_train = df_train.fillna(df_train.mean()) #fill nans in train dataset
    if groupe == "EncUP":
      df_train = df_train.iloc[1350:,:]
    if groupe == "DecTaxPet":
      df_train == df_train.iloc[205:,:]
    if groupe == "EncRist":
      df_train = df_train.iloc[1482:,:]
    if groupe == "DecTaxPet":
      df_train == df_train.iloc[205:,:]
    return df_train
  
  def load_predict(self,groupe : str):
    """
    load dataset of variables used for predictions
    return df_predict of groupe
    """
    path_df_predict = "/dbfs/tmp/pre_proc_ml/predict/" + groupe + ".sav" #set path df_predict
    df_group = lib_instance.get_only_groupe_from_jointure(groupe)
    df_group = df_group[df_group['DT_VALR'] >= "2017-1-1"]
    df_group.reset_index(inplace = True)
    path_predict = dbutils.notebook.run(self.path_notebook_preproc_preprocessing,
                            600,
                            {"path_dataframe": path_df_predict,
                             'groupe': groupe,
                             'setter_train': 1,
                             'debutDate': "2017-01-01"
                             })
    path_predict = path_predict
    pickle_in_predict = open(path_predict, 'rb')
    df_predict = pickle.load(pickle_in_predict) 
    df_predict = df_predict.drop(["Valeur"],axis = 1)
    nnans = np.sum(np.sum(df_predict.isna()))
    if nnans != 0:
      print(f"{nnans} Nans Detected in df_predict_{groupe}")
    df_predict = df_predict.fillna(df_predict.mean())
    return df_predict
  
  def load_train_predict_set(self,groupes : list):
    """
    load sets of datasets used for prediction for each groupe in groupes
    return df_train_set, df_predict_set 
    """
    df_train_set = self.load_train_set(groupes)
    df_predict_set = self.load_predict_set(groupes)
    return (df_train_set, df_predict_set)
  
  def load_train_set(self,groupes):
    """
    load sets of datasets used for training for each groupe in groupes
    return df_train_set
    """
    df_train_set = []
    for i,groupe in enumerate(groupes):
      print(groupe,'  : ',i+1,"/",len(groupes))
      path_df_train = "/dbfs/tmp/pre_proc_ml/train/" + groupe + ".sav"
      df_group = lib_instance.get_only_groupe_from_jointure(groupe)
      df_group = df_group[df_group['DT_VALR'] >= "2017-1-1"]
      df_group.reset_index(inplace = True)
      path_train = dbutils.notebook.run(path_notebook_preproc_preprocessing,
                              600,
                              {"path_dataframe": path_df_train,
                              'groupe': groupe,
                              'setter_train': 1,
                              'debutDate': "2017-01-01"
                              })
      pickle_in_train = open(path_train, 'rb')
      df_train = pickle.load(pickle_in_train)
      df_train['DT_VALR'] = df_group['DT_VALR']
      df_train_set.append(df_train)
      nnans_train = np.sum(np.sum(df_train.isna()))
      if nnans_train != 0:
        print(f"{nnans_train} Nans Detected in df_train_{groupe}")
        df_train_set[i] = df_train.fillna(method = 'ffill')
      print("")
    return df_train_set

  def load_predict_set(self,groupes):
    """
    load sets of datasets used for prediction for each groupe in groupes
    return df_predict_set 
    """
    df_predict_set = []
    for i,groupe in enumerate(groupes):
      print(groupe,'  : ',i+1,"/",len(groupes))
      path_df_predict = "/dbfs/tmp/pre_proc_ml/predict/" + groupe + ".sav"
      df_group = lib_instance.get_only_groupe_from_jointure(groupe)
      df_group = df_group[df_group['DT_VALR'] >= "2017-1-1"]
      df_group.reset_index(inplace = True)
      path_predict = dbutils.notebook.run(path_notebook_preproc_preprocessing,
                              600,
                              {"path_dataframe": path_df_predict,
                              'groupe': groupe,
                              'setter_train': 1,
                              'debutDate': "2017-01-01"
                              })
      
      pickle_in_predict = open(path_predict, 'rb')
      df_predict = pickle.load(pickle_in_predict)
      df_predict = df_predict.drop(["Valeur"],axis = 1)

      nnans = np.sum(np.sum(df_predict.isna()))
      if nnans != 0:
        print(f"{nnans} Nans Detected in df_predict_{groupe}")
        df_predict = df_predict.fillna(method = 'ffill')
      df_predict_set.append(df_predict)
      print("")
    return df_predict_set

# COMMAND ----------

class Preprocessing2: #intervalles de confiance ( PCA à supprimer ou a choisir le nombre de dimensions à garder en fonction du groupe)
  """preprocessing of dataframes by adding a label colums, label encoding, normalize and PCA"""
  def __init__(self,groupe):
    self.groupe = groupe
    self.data_jour= spark.sql("select * from jours_target_csv") \
                      .withColumn("Date", to_timestamp("Date", "yyyy-MM-dd")) \
                      .drop("_c5", 'Jour_target_UK', 'Jour_target_US').toPandas()
    self.data_jour["Is_Weekend"] = (self.data_jour["Date"].dt.day_name() == "Sunday").values + (self.data_jour["Date"].dt.day_name() == "Saturday").values



  def preproc(self,df_train,df_predict= None,PCA = False):
    """preprocessing of dataframes by adding a label colums, label encoding, scale and PCA """

    #fill nans by mean
    nnans = np.sum(np.sum(df_train.isna()))
    
    if nnans != 0:
      print(f"{nnans} Nans Detected in df_predict_{groupe}")
    df_train = df_train.fillna(df_train.mean())              
    self.target_df_train= df_train["Valeur"]
    self.DT_VALR_train = df_train["DT_VALR"]
    #add explicative features
    df_train  = self.label_by_groupe(df_train) #Add label column for regles metier
    self.labels_str = df_train["label"]
    df_train = self.label_date(df_train) #add date columns
    self.day_of_week = df_train["Day_Of_Week"]
    #label encoding qualitative features
    le_day_of_week = preprocessing.LabelEncoder()
    
    le_label = preprocessing.LabelEncoder()
    df_train["label"] = le_label.fit_transform(df_train["label"])
    self.labels = df_train["label"]
    df_train["Day_Of_Week"] = le_day_of_week.fit_transform(df_train["Day_Of_Week"])
    
    #Scaling data
     #remove DT_VALR
    df_train = df_train.drop(["DT_VALR"],axis = 1)
    scaler_train = MinMaxScaler()
    scaler_train.fit(df_train)
    columns = df_train.columns
    df_train = scaler_train.transform(df_train)
    if PCA:
      self.principal=PCA(n_components=3) # valeur à déterminer en fonction du groupe
      self.principal.fit(df_train)
      df_train=self.principal.transform(df_train)

    df_train = pd.DataFrame(df_train,columns = columns)
    df_train["DT_VALR"] = self.DT_VALR_train #Reput DT_VALR
    df_train["Valeur"] = self.target_df_train

    if df_predict is not None:
      #fill nans by mean
      nnans = np.sum(np.sum(df_predict.isna()))
      if nnans != 0:
        print(f"{nnans} Nans Detected in df_predict_{groupe}")
        df_predict = df_predict.fillna(df_predict.mean()) 

      #compute date of df_predict
      df_predict = self.set_DT_VALR(df_predict,df_train) #compute DT_VALR to set label
      df_predict["DT_VALR"] = pd.to_datetime(df_predict["DT_VALR"])

      #add explicative columns
      df_predict  = self.label_by_groupe(df_predict)
      df_predict = self.label_date(df_predict)

      #Label encoding qualitatives variables
      df_predict["label"] = le_label.transform(df_predict["label"])
      df_predict["Day_Of_Week"] = le_day_of_week.fit_transform(df_predict["Day_Of_Week"])
      df_predict = df_predict.drop(["DT_VALR"],axis = 1) #remove DT_VALR added

      #Scaling data
      scaler_predict = MinMaxScaler()
      scaler_predict.fit(df_predict)
      columns = df_predict.columns
      df_predict = scaler_predict.transform(df_predict)
      if PCA:
        self.principal=PCA(n_components=3)
        self.principal.fit(df_train)
        df_train=self.principal.transform(df_train)
      df_predict = pd.DataFrame(df_predict,columns = columns)
      return df_train,df_predict
      
    else : 
      return df_train



  def closed_days_index(self,df):
    """
    return index of closed days ( target + weekend)
    """
    return set(np.where( (df["Jour_ferie"] + df["Jour_target_EUR"] + df["Is_Weekend"]) != 0)[0])
  
  def open_days_index(self,df):
    """
    return index of open days
    """
    temp = df
    return set(np.where( (temp["Jour_ferie"] + temp["Jour_target_EUR"] + temp["Is_Weekend"]) == 0)[0])
  
  def pic_end_of_month_index(self,df):
    "return indexs of the first open day after last day of month"
    out = df
    out["end_of_month_pic"] = False
    for i,date in enumerate(out["DT_VALR"]):
      if (date + timedelta(days = 1)).day == 1:
        k=0
        while (out["Jour_ferie"][i+k] + out["Jour_target_EUR"][i+k] + (out["DT_VALR"][i+k].day_name() == "Sunday") + (out["DT_VALR"][i+k].day_name() == "Saturday") !=0) and (i + k <= len(out)):
          k+=1
        if i+k < len(out):
          out["end_of_month_pic"][i+k] = True
    return set(np.where(out["end_of_month_pic"] == True)[0])

  def label_pic_next_ferie(self,df):
    """
    add the label "pic_next_ferié" to the label column of a dataframe
    """
    out = df.assign(jour_ferie=False)
    jours_feries_names = ['JOUR_DE_L_AN','PAQUES','LUNDI_DE_PAQUES',"FETE_DU_TRAVAIL","VICTOIRE_1945","ASCENSION","PENTECOTE","LUNDI_DE_PENTECOTE","FETE_NATIONALE","ASSOMPTION","TOUSSAINT","ARMISTICE_1918","NOEL"]
    for i,date in enumerate(out["DT_VALR"]):
      jf = JoursFeries__(date.year)
      for jour_ferie in jours_feries_names:
        if date == getattr(jf,jour_ferie):
          k=0
          while (out["Jour_ferie"][i+k] + out["Jour_target_EUR"][i+k] + (out["DT_VALR"][i+k].day_name() == "Sunday") + (out["DT_VALR"][i+k].day_name() == "Saturday") !=0) and (i + k <= len(out)):
            k+=1
          if i+k < len(out):
            df["label"][i+k] = "pic_next_ferié"
    return df
  
  def next_open_day(self,index,df):
    """
    return indexs of the first open day after the days in argument
    """
    out = set()
    for i in index:
      if df["Jour_ferie"][i] + df["Jour_target_EUR"][i] + df["Is_Weekend"][i] != 0:
        k=0
        while (df["Jour_ferie"][i+k] + df["Jour_target_EUR"][i+k] + (df["DT_VALR"][i+k].day_name() == "Sunday") + (df["DT_VALR"][i+k].day_name() == "Saturday") !=0) and (i + k <= len(df)):
          k+=1
        if i+k < len(df):
          out.add(i+k)
      else:
        out.add(i)
    return list(out)
  
  def is_open(self,i,df):
    """
    bolean return true if the i-th day of df is open, else return false
    """
    print(df["Jour_ferie"][i])
    print(df["Jour_target_EUR"][i])
    print(df["DT_VALR"][i].day_name())
    if (df["Jour_ferie"][i] + df["Jour_target_EUR"][i] + (df["DT_VALR"][i].day_name() == "Sunday") + (df["DT_VALR"][i].day_name() == "Saturday")) == 0 :
      return(True)
    else:
      return(False)
  
  def day_index(self,df,day): #weekdayindex
    """
    return list of index of the day writed in string
    """
    return set(np.where(df["DT_VALR"].dt.day_name() == day)[0])
  
  def label_date(self,df):
    df["Year"] = df["DT_VALR"].dt.year
    df["Month"] = df["DT_VALR"].dt.month
    df["Day"] = df["DT_VALR"].dt.day
    df["Day_Of_Week"] = df["DT_VALR"].dt.day_name()
    return(df)

  def cut_index(self,df): #sans doute obsolete
    self.pic_end_of_month_i = self.pic_end_of_month_index(df)
    self.open_days_i = self.open_days_index(df)
    index_end_of_month = list(df.loc[self.pic_end_of_month_i].index)
    index_monday = list(df.loc[self.day_index(df,"Monday").intersection(self.open_days_i).difference(self.pic_end_of_month_i)].index)
    index_tuesday = list(df.loc[self.day_index(df,"Tuesday").intersection(self.open_days_i).difference(self.pic_end_of_month_i)].index)
    index_wednesday = list(df.loc[self.day_index(df,"Wednesday").intersection(self.open_days_i).difference(self.pic_end_of_month_i)].index)
    index_thursday = list(df.loc[self.day_index(df,"Thursday").intersection(self.open_days_i).difference(self.pic_end_of_month_i)].index)
    index_friday = list(df.loc[self.day_index(df,"Friday").intersection(self.open_days_i).difference(self.pic_end_of_month_i)].index)
    index_closed = self.closed_days_index(df)
    return index_end_of_month,index_monday,index_tuesday,index_wednesday,index_thursday,index_friday,index_closed

  def label(self,df):
    """ 
    add a label colums with labels day of the week, pic end of month, pic_next_ferié, and closed
    """
    columns = list(df.columns)
    columns.append("label")
    out = self.df.copy(deep = True).merge(self.data_jour, left_on ="DT_VALR",right_on="Date")
    out["label"]= 'unknow'
    out["label"] = df["DT_VALR"].dt.day_name()
    out["label"][self.closed_days_index(out)] = "Closed" #we et jours feriés
    return out[columns]
  
  def set_DT_VALR(self,df_predict,df_train):
    "Set DT_VALR column on predict dataset"
    df_predict["DT_VALR"]=False
    for i in range(len(df_predict["DT_VALR"])):
      df_predict["DT_VALR"][i] =  df_train["DT_VALR"][0] + timedelta(days = 1 + i)
    return df_predict
  
  def plus_n_jours_ouvrés(self,index,df,n_day = 3):
    """
    return list of index of days correponing to the index argument + 3 open days
    """
    out = set()
    for i in index:
      nb_jours_ouvrés = 0
      while nb_jours_ouvrés <n_day and (i + nb_jours_ouvrés <= len(out)):
        i = i+1
        if self.is_open(i,df):
          nb_jours_ouvrés +=1
      out.add(i)
    return(list(out))

  def label_by_groupe(self,df): #ajoute la colonne label par groupe
    """
    label dataset by group with specific rules for each group
    """
    columns_without_DT_VALR = list(df.columns)
    columns_without_DT_VALR.remove("DT_VALR") 
    columns = list(df.columns)
    columns.append("label")
    out = df.merge(self.data_jour, left_on ="DT_VALR",right_on="Date")
    out["label"]= 'Other'
    index_end_of_month,index_monday,index_tuesday,index_wednesday,index_thursday,index_friday,index_closed = self.cut_index(out)
    out = self.label_date(out)

    if self.groupe == "DecPDV":
      out["label"][index_monday] = "Monday"
      out["label"][index_tuesday] = "Tuesday"
      out["label"][index_wednesday] = "Wednesday"
      out["label"][index_thursday] = "Thursday"
      out["label"][index_friday] = "Friday"
      out["label"][self.next_open_day(index_end_of_month,out)] = "Pic_end_of_month"
      out["label"][index_closed] = "Closed"
      out = self.label_pic_next_ferie(out)
      return out[columns]
    
    if self.groupe == "EncPDV":
      out["label"][self.next_open_day(index_thursday,out)] = "next open next Thursday"
      return out[columns]
    
    elif self.groupe == "EncUP":
      for column in columns_without_DT_VALR:
        out[column] = pd.to_numeric(out[column])
      
      #le = preprocessing.LabelEncoder()
      #out["jourFerie"] =  le_jour_ferie.fit_transform(out["jourFerie"])
      out["label"][self.next_open_day(index_thursday,out)] = "Thursday"
      out["label"][self.next_open_day(np.where(out["DT_VALR"].dt.day == 10)[0],out)] = "10 du mois"
      out["label"][self.next_open_day(np.where(out["DT_VALR"].dt.day == 20)[0],out)] = "20 du mois"
      out["label"][self.next_open_day(index_end_of_month,out)] = "Pic_end_of_month"
      out["label"][index_closed] = "closed"
      out = self.label_pic_next_ferie(out)
      return out[columns]
    
    elif self.groupe == "DecUP":
      out["label"][self.next_open_day(index_monday,out)] = "Monday"
      out["label"][self.next_open_day(index_end_of_month,out)] = "Pic_end_of_month"
      out["label"][index_closed] = "closed"
      return out[columns]

    elif self.groupe == "EncPet":
      out["label"][self.next_open_day(index_monday,out)] = "Monday"
      out["label"][self.next_open_day(index_thursday,out)] = "Thursday"
      out["label"][index_closed] = "closed"
      return out[columns]
    
    elif self.groupe == "EncRprox":
      out["label"]= 'Other'
      out["label"][self.next_open_day(np.where(out["DT_VALR"].dt.day == 10)[0],out)] = "10 du mois"
      out["label"][self.next_open_day(np.where(out["DT_VALR"].dt.day == 20)[0],out)] = "20 du mois"
      out["label"][self.next_open_day(index_end_of_month,out)] = "Pic_end_of_month"
      out["label"][self.closed_days_index(out)] = "closed"
      return out[columns]
    
    elif self.groupe == "DecTaxPet":
      out["label"]= 'Other'
      out["label"][self.plus_n_jours_ouvrés(np.where(out["DT_VALR"].dt.day == 6)[0],out,3)] = "6 + 3 jours ouvrés"
      out["label"][self.plus_n_jours_ouvrés(np.where(out["DT_VALR"].dt.day == 16)[0],out,3)] = "16 + 3 jours ouvrés"
      out["label"][self.plus_n_jours_ouvrés(np.where(out["DT_VALR"].dt.day == 26)[0],out,3)] = "26 + 3 jours ouvrés"
      out["label"][self.closed_days_index(out)] = "closed"
      return out[columns]
    
    elif self.groupe == "DecTaxAlc":
      out["label"]= 'Other'
      out["label"][self.next_open_day(np.where(out["Date"].dt.day == (12 or 13 or 14 or 15))[0],out)] = "Entre 12 et 15"
      out["label"][self.closed_days_index(out)] = "closed"
      return out[columns]
    
    else :
      for column in columns_without_DT_VALR:
        out[column] = pd.to_numeric(out[column])
      out["label"][index_monday] = "Monday"
      out["label"][index_tuesday] = "Tuesday"
      out["label"][index_wednesday] = "Wednesday"
      out["label"][index_thursday] = "Thursday"
      out["label"][index_friday] = "Friday"
      out["label"][index_end_of_month] = "Pic_end_of_month"
      out["label"][index_closed] = "Closed"
      out = self.label_pic_next_ferie(out)
      return out[columns]
    


# COMMAND ----------

class Preprocessing:
  def __init__(self,groupe):
    self.groupe = groupe
    self.data_jour= spark.sql("select * from jours_target_csv") \
                      .withColumn("Date", to_timestamp("Date", "yyyy-MM-dd")) \
                      .drop("_c5", 'Jour_target_UK', 'Jour_target_US').toPandas()
    self.data_jour["Is_Weekend"] = (self.data_jour["Date"].dt.day_name() == "Sunday").values + (self.data_jour["Date"].dt.day_name() == "Saturday").values

  def preproc(self,df_train,df_predict= None):
    """preprocessing of dataframes by adding a label colums and label encoding"""
    df_train  = self.label_by_groupe(df_train)
    df_train = self.label_date(df_train)
    le_day_of_week = preprocessing.LabelEncoder()
    self.labels_str = list(set(df_train["label"]))
    le_label = preprocessing.LabelEncoder()
    self.labels = le_label.fit_transform(self.labels_str)
    df_train["label"] = le_label.fit_transform(df_train["label"])
    df_train["Day_Of_Week"] = le_day_of_week.fit_transform(df_train["Day_Of_Week"])
    if df_predict is not None:
      df_predict = self.set_DT_VALR(df_predict,df_train)
      df_predict["DT_VALR"] = pd.to_datetime(df_predict["DT_VALR"])
      df_predict  = self.label_by_groupe(df_predict)
      df_predict = self.label_date(df_predict)
      df_predict["label"] = le_label.transform(df_predict["label"])
      df_predict["Day_Of_Week"] = le_day_of_week.fit_transform(df_predict["Day_Of_Week"])
      df_predict = df_predict.drop(["DT_VALR"],axis = 1)
      return df_train,df_predict
    else : 
      return df_train


  def closed_days_index(self,df):
    """
    return index of closed days ( target + weekend)
    """
    return set(np.where( (df["Jour_ferie"] + df["Jour_target_EUR"] + df["Is_Weekend"]) != 0)[0])
  
  def open_days_index(self,df):
    """
    return index of open days
    """
    temp = df
    return set(np.where( (temp["Jour_ferie"] + temp["Jour_target_EUR"] + temp["Is_Weekend"]) == 0)[0])
  
  def pic_end_of_month_index(self,df):
    "return indexs of the first open day after last day of month"
    out = df
    out["end_of_month_pic"] = False
    for i,date in enumerate(out["DT_VALR"]):
      if (date + timedelta(days = 1)).day == 1:
        k=0
        while (out["Jour_ferie"][i+k] + out["Jour_target_EUR"][i+k] + (out["DT_VALR"][i+k].day_name() == "Sunday") + (out["DT_VALR"][i+k].day_name() == "Saturday") !=0) and (i + k <= len(out)):
          k+=1
        if i+k < len(out):
          out["end_of_month_pic"][i+k] = True
    return set(np.where(out["end_of_month_pic"] == True)[0])

  def label_pic_next_ferie(self,df):
    """
    add the label "pic_next_ferié" to the label column of a dataframe
    """
    out = df.assign(jour_ferie=False)
    jours_feries_names = ['JOUR_DE_L_AN','PAQUES','LUNDI_DE_PAQUES',"FETE_DU_TRAVAIL","VICTOIRE_1945","ASCENSION","PENTECOTE","LUNDI_DE_PENTECOTE","FETE_NATIONALE","ASSOMPTION","TOUSSAINT","ARMISTICE_1918","NOEL"]
    for i,date in enumerate(out["DT_VALR"]):
      jf = JoursFeries__(date.year)
      for jour_ferie in jours_feries_names:
        if date == getattr(jf,jour_ferie):
          k=0
          while (out["Jour_ferie"][i+k] + out["Jour_target_EUR"][i+k] + (out["DT_VALR"][i+k].day_name() == "Sunday") + (out["DT_VALR"][i+k].day_name() == "Saturday") !=0) and (i + k <= len(out)):
            k+=1
          if i+k < len(out):
            df["label"][i+k] = "pic_next_ferié"
    return df
  
  def next_open_day(self,index,df):
    """
    return indexs of the first open day after the days in argument
    """
    out = set()
    for i in index:
      if df["Jour_ferie"][i] + df["Jour_target_EUR"][i] + df["Is_Weekend"][i] != 0:
        k=0
        while (df["Jour_ferie"][i+k] + df["Jour_target_EUR"][i+k] + (df["DT_VALR"][i+k].day_name() == "Sunday") + (df["DT_VALR"][i+k].day_name() == "Saturday") !=0) and (i + k <= len(df)):
          k+=1
        if i+k < len(df):
          out.add(i+k)
      else:
        out.add(i)
    return list(out)
  
  def is_open(self,i,df):
    """
    bolean return true if the i-th day of df is open, else return false
    """
    print(df["Jour_ferie"][i])
    print(df["Jour_target_EUR"][i])
    print(df["DT_VALR"][i].day_name())
    if (df["Jour_ferie"][i] + df["Jour_target_EUR"][i] + (df["DT_VALR"][i].day_name() == "Sunday") + (df["DT_VALR"][i].day_name() == "Saturday")) == 0 :
      return(True)
    else:
      return(False)
  
  def day_index(self,df,day):
    """
    return list of index of the day writed in string
    """
    return set(np.where(df["DT_VALR"].dt.day_name() == day)[0])
  
  def label_date(self,df):
    df["Year"] = df["DT_VALR"].dt.year
    df["Month"] = df["DT_VALR"].dt.month
    df["Day"] = df["DT_VALR"].dt.day
    df["Day_Of_Week"] = df["DT_VALR"].dt.day_name()
    return(df)

  def cut_index(self,df):
    self.pic_end_of_month_i = self.pic_end_of_month_index(df)
    self.open_days_i = self.open_days_index(df)
    index_end_of_month = list(df.loc[self.pic_end_of_month_i].index)
    index_monday = list(df.loc[self.day_index(df,"Monday").intersection(self.open_days_i).difference(self.pic_end_of_month_i)].index)
    index_tuesday = list(df.loc[self.day_index(df,"Tuesday").intersection(self.open_days_i).difference(self.pic_end_of_month_i)].index)
    index_wednesday = list(df.loc[self.day_index(df,"Wednesday").intersection(self.open_days_i).difference(self.pic_end_of_month_i)].index)
    index_thursday = list(df.loc[self.day_index(df,"Thursday").intersection(self.open_days_i).difference(self.pic_end_of_month_i)].index)
    index_friday = list(df.loc[self.day_index(df,"Friday").intersection(self.open_days_i).difference(self.pic_end_of_month_i)].index)
    index_closed = self.closed_days_index(df)
    return index_end_of_month,index_monday,index_tuesday,index_wednesday,index_thursday,index_friday,index_closed

  def label(self,df):
    """ 
    add a label colums with labels day of the week, pic end of month, pic_next_ferié, and closed
    """
    columns = list(df.columns)
    columns.append("label")
    out = self.df.copy(deep = True).merge(self.data_jour, left_on ="DT_VALR",right_on="Date")
    out["label"]= 'unknow'
    index_end_of_month,index_monday,index_tuesday,index_wednesday,index_thursday,index_friday,index_closed = self.cut_index()
    out["label"][index_monday] = "Monday"
    out["label"][index_tuesday] = "Tuesday"
    out["label"][index_wednesday] = "Wednesday"
    out["label"][index_thursday] = "Thursday"
    out["label"][index_friday] = "Friday"
    out["label"][index_end_of_month] = "Pic_end_of_month"
    out["label"][index_closed] = "Closed"
    out = self.label_pic_next_ferie(out)
    return out[columns]
  
  def set_DT_VALR(self,df_predict,df_train):
    "Set DT_VALR column on predict dataset"
    df_predict["DT_VALR"]=False
    for i in range(len(df_predict["DT_VALR"])):
      df_predict["DT_VALR"][i] =  df_train["DT_VALR"][0] + timedelta(days = 1 + i)
    return df_predict
  
  def plus_n_jours_ouvrés(self,index,df,n_day = 3):
    """
    return list of index of days correponing to the index argument + 3 open days
    """
    out = set()
    for i in index:
      nb_jours_ouvrés = 0
      while nb_jours_ouvrés <n_day and (i + nb_jours_ouvrés <= len(out)):
        i = i+1
        if self.is_open(i,df):
          nb_jours_ouvrés +=1
      out.add(i)
    return(list(out))

  def label_by_groupe(self,df):
    """
    label dataset by group with specific rules for each group
    """
    columns_without_DT_VALR = list(df.columns)
    columns_without_DT_VALR.remove("DT_VALR") 
    columns = list(df.columns)
    columns.append("label")
    out = df.merge(self.data_jour, left_on ="DT_VALR",right_on="Date")
    out["label"]= 'Other'
    index_end_of_month,index_monday,index_tuesday,index_wednesday,index_thursday,index_friday,index_closed = self.cut_index(out)
    out = self.label_date(out)
    if self.groupe == "DecPDV":
      out["label"][index_monday] = "Monday"
      out["label"][index_tuesday] = "Tuesday"
      out["label"][index_wednesday] = "Wednesday"
      out["label"][index_thursday] = "Thursday"
      out["label"][index_friday] = "Friday"
      out["label"][self.next_open_day(index_end_of_month,out)] = "Pic_end_of_month"
      out["label"][index_closed] = "Closed"
      out = self.label_pic_next_ferie(out)
      return out[columns]
    
    if self.groupe == "EncPDV":
      out["label"][self.next_open_day(index_thursday,out)] = "next open next Thursday"
      return out[columns]
    
    elif self.groupe == "EncUP":
      for column in columns_without_DT_VALR:
        out[column] = pd.to_numeric(out[column])
      
      #le = preprocessing.LabelEncoder()
      #out["jourFerie"] =  le_jour_ferie.fit_transform(out["jourFerie"])
      out["label"][self.next_open_day(index_thursday,out)] = "Thursday"
      out["label"][self.next_open_day(np.where(out["DT_VALR"].dt.day == 10)[0],out)] = "10 du mois"
      out["label"][self.next_open_day(np.where(out["DT_VALR"].dt.day == 20)[0],out)] = "20 du mois"
      out["label"][self.next_open_day(index_end_of_month,out)] = "Pic_end_of_month"
      out["label"][index_closed] = "closed"
      out = self.label_pic_next_ferie(out)
      return out[columns]
    
    elif self.groupe == "DecUP":
      out["label"][self.next_open_day(index_monday,out)] = "Monday"
      out["label"][self.next_open_day(index_end_of_month,out)] = "Pic_end_of_month"
      out["label"][index_closed] = "closed"
      return out[columns]

    elif self.groupe == "EncPet":
      out["label"][self.next_open_day(index_monday,out)] = "Monday"
      out["label"][self.next_open_day(index_thursday,out)] = "Thursday"
      out["label"][index_closed] = "closed"
      return out[columns]
    
    elif self.groupe == "EncRprox":
      out["label"]= 'Other'
      out["label"][self.next_open_day(np.where(out["DT_VALR"].dt.day == 10)[0],out)] = "10 du mois"
      out["label"][self.next_open_day(np.where(out["DT_VALR"].dt.day == 20)[0],out)] = "20 du mois"
      out["label"][self.next_open_day(index_end_of_month,out)] = "Pic_end_of_month"
      out["label"][self.closed_days_index(out)] = "closed"
      return out[columns]
    
    elif self.groupe == "DecTaxPet":
      out["label"]= 'Other'
      out["label"][self.plus_n_jours_ouvrés(np.where(out["DT_VALR"].dt.day == 6)[0],out,3)] = "6 + 3 jours ouvrés"
      out["label"][self.plus_n_jours_ouvrés(np.where(out["DT_VALR"].dt.day == 16)[0],out,3)] = "16 + 3 jours ouvrés"
      out["label"][self.plus_n_jours_ouvrés(np.where(out["DT_VALR"].dt.day == 26)[0],out,3)] = "26 + 3 jours ouvrés"
      out["label"][self.closed_days_index(out)] = "closed"
      return out[columns]
    
    elif self.groupe == "DecTaxAlc":
      out["label"]= 'Other'
      out["label"][self.next_open_day(np.where(out["Date"].dt.day == (12 or 13 or 14 or 15))[0],out)] = "Entre 12 et 15"
      out["label"][self.closed_days_index(out)] = "closed"
      return out[columns]
    
    else :
      for column in columns_without_DT_VALR:
        out[column] = pd.to_numeric(out[column])
      out["label"][index_monday] = "Monday"
      out["label"][index_tuesday] = "Tuesday"
      out["label"][index_wednesday] = "Wednesday"
      out["label"][index_thursday] = "Thursday"
      out["label"][index_friday] = "Friday"
      out["label"][index_end_of_month] = "Pic_end_of_month"
      out["label"][index_closed] = "Closed"
      out = self.label_pic_next_ferie(out)
      return out[columns]
    


# COMMAND ----------

class to_do:
  def is_weekend(self,df):
    self.data_jour["Is Weekend"] = (self.data_jour["Date"].dt.day_name() == "Sunday").values + (self.data_jour["Date"].dt.day_name() == "Saturday").values
    df2 = df.merge(self.data_jour, left_on ="DT_VALR",right_on="Date")
    return df
  
  def label_last_day_of_month(self,df):
    out = df.copy(deep=True).merge(self.data_jour, left_on ="DT_VALR",right_on="Date")
    out["last_day_of_month"] = ((out["DT_VALR"] + timedelta(days = 1)) == 1)
    return out
  
  def is_last_day_of_month(self,date):
    if (date + timedelta(days = 1)).day == 1:
      return True
    else :
      return False

  def df_day(self,df,day):
    out = df.copy(deep=True)
    out = out[out["DT_VALR"].dt.day_name()==day]

  def label_closed_days(self,df):
    out = df.copy(deep=True)
    out = out[out["jourFerie"] + out["jourtargetEUR"] + (out["DT_VALR"].dt.day_name() == "Sunday") + (out["DT_VALR"].dt.day_name() == "Saturday") == 0]
    return out

  def df_pic_end_of_month(self):
    out = self.df.copy(deep=True).assign(end_of_month_pic=False)
    out = out.merge(self.data_jour, left_on ="DT_VALR",right_on="Date")
    for i,date in enumerate(df["DT_VALR"]):
      if self.is_last_day_of_month(date):
        k=0
        while (out["jourFerie"][i+k] + out["jourtargetEUR"][i+k] + (out["DT_VALR"][i+k].day_name() == "Sunday") + (out["DT_VALR"][i+k].day_name() == "Saturday") !=0) and (i + k <= len(df)):
          k+=1
        if i+k < len(df):
          out["end_of_month_pic"][i+k] = True
    return out
  
  def label_jours_feries(self,df):
    out = df.copy(deep=True)
    jours_feries_names = ['JOUR_DE_L_AN','PAQUES','LUNDI_DE_PAQUES',"FETE_DU_TRAVAIL","VICTOIRE_1945","ASCENSION","PENTECOTE","LUNDI_DE_PENTECOTE","FETE_NATIONALE","ASSOMPTION","TOUSSAINT","ARMISTICE_1918","NOEL"]
    data = np.zeros((len(df),len(jours_feries_names)))
    feries_cols = pd.DataFrame(data=data,columns=jours_feries_names)
    out = pd.concat([df,feries_cols], axis=1)
    for i,date in enumerate(out["DT_VALR"]):
      jf = JoursFeries__(date.year)
      if date == jf.JOUR_DE_L_AN:
        out["JOUR_DE_L_AN"][i] = True
      if date == jf.PAQUES:
        out["PAQUES"][i] = True
      if date == jf.LUNDI_DE_PAQUES:
        out["LUNDI_DE_PAQUES"][i] = True
      if date == jf.FETE_DU_TRAVAIL:
        out["FETE_DU_TRAVAIL"][i] = True
      if date == jf.VICTOIRE_1945:
        out["VICTOIRE_1945"][i] = True
      if date == jf.ASCENSION:
        out["ASCENSION"][i] = True
      if date == jf.PENTECOTE:
        out["PENTECOTE"][i] = True
      if date == jf.LUNDI_DE_PENTECOTE:
        out["LUNDI_DE_PENTECOTE"][i] = True
      if date == jf.FETE_NATIONALE:
        out["FETE_NATIONALE"][i] = True
      if date == jf.ASSOMPTION:
        out["ASSOMPTION"][i] = True
      if date == jf.TOUSSAINT:
        out["TOUSSAINT"][i] = True
      if date == jf.ARMISTICE_1918:
        out["ARMISTICE_1918"][i] = True
      if date == jf.VICTOIRE_1945:
        out["VICTOIRE_1945"][i] = True
      if date == jf.NOEL:
        out["NOEL"][i] = True
    return out

  def label_pic_next_ferie(self,df):
    out = df.copy(deep=True)
    jours_feries_names = ['JOUR_DE_L_AN','PAQUES','LUNDI_DE_PAQUES',"FETE_DU_TRAVAIL","VICTOIRE_1945","ASCENSION","PENTECOTE","LUNDI_DE_PENTECOTE","FETE_NATIONALE","ASSOMPTION","TOUSSAINT","ARMISTICE_1918","NOEL"]
    data = np.zeros((len(df),len(jours_feries_names)))
    feries_cols = pd.DataFrame(data=data,columns=jours_feries_names)
    out = pd.concat([df,feries_cols], axis=1)
    for i,date in enumerate(out["DT_VALR"]):
      jf = JoursFeries__(date.year)
      if date == jf.JOUR_DE_L_AN:
        out["JOUR_DE_L_AN"][i] = True
      if date == jf.PAQUES:
        out["PAQUES"][i] = True
      if date == jf.LUNDI_DE_PAQUES:
        out["LUNDI_DE_PAQUES"][i] = True
      if date == jf.FETE_DU_TRAVAIL:
        out["FETE_DU_TRAVAIL"][i] = True
      if date == jf.VICTOIRE_1945:
        out["VICTOIRE_1945"][i] = True
      if date == jf.ASCENSION:
        out["ASCENSION"][i] = True
      if date == jf.PENTECOTE:
        out["PENTECOTE"][i] = True
      if date == jf.LUNDI_DE_PENTECOTE:
        out["LUNDI_DE_PENTECOTE"][i] = True
      if date == jf.FETE_NATIONALE:
        out["FETE_NATIONALE"][i] = True
      if date == jf.ASSOMPTION:
        out["ASSOMPTION"][i] = True
      if date == jf.TOUSSAINT:
        out["TOUSSAINT"][i] = True
      if date == jf.ARMISTICE_1918:
        out["ARMISTICE_1918"][i] = True
      if date == jf.VICTOIRE_1945:
        out["VICTOIRE_1945"][i] = True
      if date == jf.NOEL:
        out["NOEL"][i] = True
    return out
  
  def next_open_day(self,date):
    while np.sum(np.sum(self.data_jour[self.data_jour["Date"]==date])) != 0:
      date = date + timedelta(days = 1)
    return date