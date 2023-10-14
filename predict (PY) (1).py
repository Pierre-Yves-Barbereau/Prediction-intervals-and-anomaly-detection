# Databricks notebook source
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Import des packages

# COMMAND ----------

import time

import numpy as np
import pandas as pd
import pickle
#import time
import copy

# fonction spark pour gestion du dataframe
from pyspark.sql.functions import *
from pyspark.sql.types import *

from datetime import datetime, timedelta, date

import pathlib

# parallelisation
import concurrent.futures

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import json

import forestci as fci

# COMMAND ----------

# MAGIC %run /Tools/library/CFM

# COMMAND ----------

lib_cfm = Cfm()

# COMMAND ----------

# MAGIC %md
# MAGIC # Variables

# COMMAND ----------

# var retry parallel
global submit_retry
submit_retry = 1

# var retry parallel
global max_workers_in_parallel
threading_active_count =threading.active_count()
max_worker_param=2
if threading_active_count > max_worker_param :
  max_workers_in_parallel =  max_worker_param
else :
  max_workers_in_parallel = threading_active_count
print(f"threading_active_count is {threading_active_count}")
print(f"Max_Workers_In_Parallel is {max_workers_in_parallel}")

# COMMAND ----------

print(f" :anneeHist {lib_cfm.define_widget('anneeHist')}")
print(f" :taux {lib_cfm.define_widget('taux')}")
print(f" :montantFidelite {lib_cfm.define_widget('montantFidelite')}")
print(f" :debutDate {lib_cfm.define_widget('debutDate')}")
print(f" :montantRemiseConditionnelle {lib_cfm.define_widget('montantRemiseConditionnelle')}")
print(f" :ristourneCompensee {lib_cfm.define_widget('ristourneCompensee')}")
print(f" :montantAjustement {lib_cfm.define_widget('montantAjustement')}")
print(f" :evenements {lib_cfm.define_widget('evenements')}")
print(f" :prixBaril {lib_cfm.define_widget('prixBaril')}")
print(f" :prixDollar {lib_cfm.define_widget('prixDollar')}")
print(f" :ristournePompe {lib_cfm.define_widget('ristournePompe')}")
print(f" :scenario_name {lib_cfm.define_widget('scenario_name')}")
print(f" :creation_date {lib_cfm.define_widget('creation_date')}")


# COMMAND ----------

param_predict = Parametres(
  annee_hist=lib_cfm.define_widget('anneeHist'),
  taux=lib_cfm.define_widget('taux'),
  montant_fidelite=lib_cfm.define_widget('montantFidelite'),
  montant_remise_conditionnelle=lib_cfm.define_widget('montantRemiseConditionnelle'),
  ristournes_compensees=lib_cfm.define_widget('ristourneCompensee'),
  montants_ajustement=lib_cfm.define_widget('montantAjustement'),
  date_prev=datetime.strptime(lib_cfm.define_widget('debutDate'), '%Y-%m-%d'),
  evenements=lib_cfm.define_widget('evenements'),
  prixBaril=lib_cfm.define_widget('prixBaril'),
  prixDollar=lib_cfm.define_widget('prixDollar'),
  ristournePompe=lib_cfm.define_widget('ristournePompe'),
  scenario_name=lib_cfm.define_widget('scenario_name'),
  creation_date=lib_cfm.define_widget('creation_date')
)

# COMMAND ----------

groupe = TypeGroup(lib_cfm.define_widget('groupe'), param_predict)  

# COMMAND ----------

# MAGIC %md
# MAGIC # Partie ML

# COMMAND ----------

def predict_machine_learning(groupe_: TypeGroup):
  res = groupe_.predict_ml()
  return res

# COMMAND ----------

# MAGIC %md
# MAGIC # Prédictions paralleles

# COMMAND ----------

def parallel_ml_baseline(max_workers_in_parallel_: int, submit_retry_: int, groupe_: TypeGroup):
    """
    Main fonction de la prediction. Parallelise les algorithmes baseline et machine learning, via le ThreadPoolExecutor
    
    Si le TypeGroup n'a pas d'algorithme ML, on realise une deepcopy du json recupere
    Si une erreur a ete relevee pendant l'execution d'un des deux algos, on retry le nombre de submit_retry correspondant.
    
    params: max_workers_in_parallel_ - nombre de workers lie a la parallelisation
            submit_retry_ - nombre d'essais si les algo plantent
            groupe_ - le type de groupe a predire
    """
    # parallelisation entre ML et baseline
    groupe_ml = groupe_.__copy__()
    futures = [executor.submit(groupe_.predict_baseline),
                executor.submit(groupe_ml.predict_ml)]
    return_value = [f.result() for f in futures]
        
    return return_value

# COMMAND ----------

liste_exit = parallel_ml_baseline(max_workers_in_parallel_=max_workers_in_parallel, submit_retry_=submit_retry, groupe_=groupe)

# COMMAND ----------

liste_exit

# COMMAND ----------

dbutils.notebook.exit(json.dumps(liste_exit))