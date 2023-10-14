# Databricks notebook source
# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/Prod/preprocessing

# COMMAND ----------

fluxs = ["EncPDV", "DecPDV", "EncUP", "DecUP", "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"]

NAME_ADB_BASE_CFM_IC = "IntervalleConfiance"
#Nom des vues à créer
NAME_ADB_VIEW_FLUX_HISTORIQUE = "cfm_flux_historique"
NAME_ADB_VIEW_FLUX_FIN_TRAN = "cfm_fin_tran"

#PAth notebooks
#path_notebook_preproc_preprocessing = '/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing'
#path_dataframe_train = '/dbfs/tmp/pre_proc_ml/train/DecPDV.sav'

path_notebook_preproc_preprocessing = '/Notebooks/Shared/Industrialisation/PrepocFolder/Preprocessing'

alphas = [0.05,0.95]
test_size = 7
val_size = 7
cal_size = 7



# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

# COMMAND ----------

# MAGIC %run /Users/pierre-yves.barbereau@mousquetaires.com/IC_functions

# COMMAND ----------

preproc = preprocessing(path_notebook_preproc_preprocessing = path_notebook_preproc_preprocessing)

# COMMAND ----------

dataframe_dectrain_set = preproc.load_train_set(fluxs)

# COMMAND ----------

from plotly.subplots import make_subplots
error_list = []
fig = make_subplots(rows=len(fluxs),
                    cols=1,
                    #subplot_titles=("EncPDV", "DecPDV"))
                    subplot_titles=("EncPDV", "DecPDV", "EncUP", "DecUP", "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"))

for i,flux in enumerate(fluxs) :

  print("")
  print(fluxs[i],"  ",i+1,"/",len(fluxs))
  if np.sum(np.sum(dataframe_dectrain_set[i].isna())) != 0:
    print(f"{np.sum(np.sum(dataframe_dectrain_set[i].isna()))} Nans Detected in dataframe_decpredict_{fluxs[i]}")
    dataframe_dectrain_set[i] = dataframe_dectrain_set[i].fillna(method = 'ffill')
  set_size = int(dataframe_dectrain_set[i].shape[0]/7)
  print("set_size =",set_size)
  ci_test = Conformal_Inference(alphas = alphas,test_size = set_size,val_size = set_size,cal_size = set_size)
  ci_test.fit(dataframe_dectrain_set[i],mode = "test")
  asymetric_conformal_ic_test = ci_test.asymetric_conformal_IC(ci_test.X_test)
  error_list.append((np.sum(ci_test.y_test<asymetric_conformal_ic_test[0]) + np.sum(ci_test.y_test>asymetric_conformal_ic_test[1]))/len(ci_test.y_test))

  fig.append_trace(go.Scatter(x = ci_test.xX_test, y=asymetric_conformal_ic_test[0],
                          mode='lines',
                          name=f'q_{alphas[0]}',
                          line=dict(
                          color='rgb(256, 0, 0)',
                          width=0),
                          legendgroup=fluxs[i],
                          showlegend = False),
                      row=i+1, col=1)

  fig.append_trace(go.Scatter(x = ci_test.xX_test, y=asymetric_conformal_ic_test[1],
                        mode='lines',
                        name=f'{int(np.round(100*(alphas[1]-alphas[0])))}% Confidence Interval',
                        line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                          legendgroup=fluxs[i],
                        fill='tonexty',
                        fillcolor='rgba(0,176,246,0.2)'),
                    row=i+1, col=1)
  
  fig.append_trace(go.Scatter(x = ci_test.xX_test, y=asymetric_conformal_ic_test[2],
                        mode='lines',
                        name=f'y_median'),
                    row=i+1, col=1)
  
  fig.append_trace(go.Scatter(x = ci_test.xX_test, y=ci_test.y_test,
                        mode='lines',
                        name='y_true',
                        line=dict(
                          color='rgb(0, 0, 256)',
                          width=1),
                          legendgroup=fluxs[i]),
                    row=i+1, col=1)

dict_error = { k: v for k, v in zip(fluxs,error_list) }
fig.for_each_annotation(lambda a: a.update(text = a.text + ' : ' +  f"Confidence = {1 - dict_error[a.text]}"))
fig.update_layout(height=500*len(fluxs), title_text="Subplots by flux") 
fig.update_layout(legend_tracegroupgap=480)    
fig.show()

# COMMAND ----------

from plotly.subplots import make_subplots
error_list = []
fig = make_subplots(rows=len(fluxs),
                    cols=1,
                    #subplot_titles=("EncPDV", "DecPDV"))
                    subplot_titles=("EncPDV", "DecPDV", "EncUP", "DecUP", "DecPet", "DecTaxAlc", "DecTaxPet", "EncRist", "EncRprox"))

for i,flux in enumerate(fluxs) :

  print("")
  print(fluxs[i],"  ",i+1,"/",len(fluxs))
  if np.sum(np.sum(dataframe_dectrain_set[i].isna())) != 0:
    print(f"{np.sum(np.sum(dataframe_dectrain_set[i].isna()))} Nans Detected in dataframe_decpredict_{fluxs[i]}")
    dataframe_dectrain_set[i] = dataframe_dectrain_set[i].fillna(method = 'ffill')
  set_size = int(dataframe_dectrain_set[i].shape[0]/7)
  print("set_size =",set_size)
  ci_test = Conformal_Inference(alphas = alphas,test_size = set_size,val_size = set_size,cal_size = set_size)
  ci_test.fit(dataframe_dectrain_set[i],mode = "test")
  asymetric_conformal_ic_test = ci_test.asymetric_conformal_IC(ci_test.X_test)
  error_list.append((np.sum(ci_test.y_test<asymetric_conformal_ic_test[0]) + np.sum(ci_test.y_test>asymetric_conformal_ic_test[1]))/len(ci_test.y_test))

  fig.append_trace(go.Scatter(x = ci_test.xX_test, y=asymetric_conformal_ic_test[0],
                          mode='lines',
                          name=f'q_{alphas[0]}',
                          line=dict(
                          color='rgb(256, 0, 0)',
                          width=0),
                          legendgroup=fluxs[i],
                          showlegend = False),
                      row=i+1, col=1)

  fig.append_trace(go.Scatter(x = ci_test.xX_test, y=asymetric_conformal_ic_test[1],
                        mode='lines',
                        name=f'{int(np.round(100*(alphas[1]-alphas[0])))}% Confidence Interval',
                        line=dict(
                          color='rgb(0, 256, 0)',
                          width=0),
                          legendgroup=fluxs[i],
                        fill='tonexty',
                        fillcolor='rgba(0,176,246,0.2)'),
                    row=i+1, col=1)
  
  fig.append_trace(go.Scatter(x = ci_test.xX_test, y=asymetric_conformal_ic_test[2],
                        mode='lines',
                        name=f'y_median'line=dict(
                          color='rgb(0, 0, 256)',
                          width=1),
                          legendgroup=fluxs[i]),
                    row=i+1, col=1)
  
  fig.append_trace(go.Scatter(x = ci_test.xX_test, y=ci_test.y_test,
                        mode='lines',
                        name='y_true',
                        line=dict(
                          color='rgb(0, 0, 256)',
                          width=1),
                          legendgroup=fluxs[i]),
                    row=i+1, col=1)

dict_error = { k: v for k, v in zip(fluxs,error_list) }
fig.for_each_annotation(lambda a: a.update(text = a.text + ' : ' +  f"Confidence = {1 - dict_error[a.text]}"))
fig.update_layout(height=500*len(fluxs), title_text="Subplots by flux") 
fig.update_layout(legend_tracegroupgap=480)    
fig.show()

# COMMAND ----------



# COMMAND ----------

pickle_in = open('/dbfs/tmp/models/*.sav', 'rb')

modele_ml = pickle.load(pickle_in)

pickle_in = open('/dbfs/tmp/models/RandomForestRegressorEncPDV.sav', 'rb') modele_ml = pickle.load(pickle_in)

# COMMAND ----------

ls

# COMMAND ----------

