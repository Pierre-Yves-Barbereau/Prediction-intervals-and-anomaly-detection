# Databricks notebook source
import os

# COMMAND ----------

# ADSL 
DATALAKE_CONTAINER = f"/mnt/datalake_cfm_{os.getenv('ENVIRONMENT_PREFIXE')}d91381fs01/"
FOLDER_INIT = "brutes/exploratoire/Stage_Intervalle_Confiance/prediction/Prod/"
FOLDER_LEVEL = "raffinees/"
FOLDER_ADB= "BDDDatabricks/"

#Nom Bases
NAME_ADB_BASE_CFM= "default"
NAME_ADB_BASE_CFM_IC = "IntervalleConfiance"

#Nom table
NAME_ADB_BASE_CFM_FLUX_HISTORIQUE= "jointuretable"

#Nom des vues à créer
NAME_ADB_VIEW_FLUX_HISTORIQUE = "cfm_flux_historique"

FILE_TYPE = "delta"

# COMMAND ----------

# DBTITLE 1,CREATE BDD
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {NAME_ADB_BASE_CFM_IC}  LOCATION \'{DATALAKE_CONTAINER+FOLDER_LEVEL+FOLDER_ADB+NAME_ADB_BASE_CFM_IC}\';")

# COMMAND ----------

# DBTITLE 1,CREATE VIEW
# Supprimer la vue si besoin
#spark.sql(f"DROP VIEW IF EXISTS {NAME_ADB_BASE_CFM_IC}.{NAME_ADB_VIEW_FLUX_HISTORIQUE}")
 
# Créer la vue
spark.sql(f"""
          CREATE VIEW IF NOT EXISTS {NAME_ADB_BASE_CFM_IC}.{NAME_ADB_VIEW_FLUX_HISTORIQUE} AS
          SELECT *
          FROM {NAME_ADB_BASE_CFM}.{NAME_ADB_BASE_CFM_FLUX_HISTORIQUE}
""")

# COMMAND ----------

spark.sql(f"""
          SELECT *
          FROM {NAME_ADB_BASE_CFM_IC}.{NAME_ADB_VIEW_FLUX_HISTORIQUE}
""")

# COMMAND ----------

