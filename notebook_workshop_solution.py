# Databricks notebook source
# MAGIC %md
# MAGIC # Import the packages

# COMMAND ----------

pip install taipy

# COMMAND ----------

from taipy.gui import Gui, Markdown, notify
from taipy import Config, Scope
import taipy as tp

import datetime as dt

from pmdarima import auto_arima

from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC # Taipy Gui Basics
# MAGIC ## Markdown Syntax

# COMMAND ----------

# MAGIC %md
# MAGIC Taipy uses the Markdown syntax to display elements. `#` creates a title, `*` puts your text in italics and `**` puts it in bold.

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/gui_basic_eng.png)

# COMMAND ----------

page_md = """
# Taipy

Test **here** to put some *markdown*

Click to access the [doc](https://docs.taipy.io/en/latest/)
"""

# COMMAND ----------

Gui(page_md).run(dark_mode=False, run_browser=False, port=6001)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visual elements
# MAGIC Create different visual elements. The syntax is always the same for each visual element.  `<|{value}|name_of_visual_element|property_1=value_of_property_1|...|>`
# MAGIC - Create a [slider](https://docs.taipy.io/en/latest/manuals/gui/viselements/slider/) `<|{value}|slider|>`
# MAGIC
# MAGIC - Create a [date](https://docs.taipy.io/en/latest/manuals/gui/viselements/date/) `<|{value}|date|>`
# MAGIC
# MAGIC - Create a [selector](https://docs.taipy.io/en/latest/manuals/gui/viselements/selector/) `<|{value}|selector|lov={list_of_values}|>`
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/control.png)

# COMMAND ----------

slider_value = 0
date_value = None
selected_value = None
selector = ['Test 1', 'Test 2', 'Test 3']
vark = 42
control_md = """
## Controls

<|{slider_value}|slider|max=50> <|{slider_value}|>
<|{vark}|slider|> <|{vark}|>
<|{date_value}|date|> <|{date_value}|>

<|{selected_value}|selector|lov={selector}|> <|{selected_value}|>
"""

# COMMAND ----------

Gui(control_md).run(dark_mode=False, run_browser=False, port=6099)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Viz
# MAGIC
# MAGIC A dataset gathering information on the number of deaths, confirmed cases and recovered for different regions is going to be used to create an interactive Dashboard.

# COMMAND ----------

path_to_data = "data/covid-19-all.csv"
data = pd.read_csv(path_to_data, low_memory=False)
data[-5:]

# COMMAND ----------

def initialize_case_evolution(data, selected_country='India') -> pd.DataFrame:
    # Aggregation of the dataframe per Country/Region
    country_date_df = data.groupby(["Country/Region",'Date']).sum().reset_index()
    
    # a country is selected, here India by default
    country_date_df = country_date_df.loc[country_date_df['Country/Region']==selected_country]
    return country_date_df

# COMMAND ----------

country_date_df = initialize_case_evolution(data)
country_date_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Create a [chart](https://docs.taipy.io/en/latest/manuals/gui/viselements/chart/) showing the evolution of Deaths in France (_Deaths_ for _y_ and _Date_ for _x_). The visual element (chart) has the same syntax as the other ones with specific properties (_x_, _y_, _type_ for example). Here are some [examples of charts](https://docs.taipy.io/en/release-1.1/manuals/gui/viselements/charts/bar/). The _x_ and _y_ porperties only needs the name of the dataframe columns to display.

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/simple_graph.png)

# COMMAND ----------

country_md = "<|{country_date_df}|chart|x=Date|y=Deaths|type=bar|>"

# COMMAND ----------

Gui(country_md).run(dark_mode=False, run_browser=False, port=6003)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add new traces
# MAGIC
# MAGIC - Add on the graph the number of Confirmed and Recovered cases (_Confirmed_ and _Recovered_) with the number of Deaths
# MAGIC - _y_ (and _x_) can be indexed this way to add more traces (`y[1]=`, `y[2]=`, `y[3]=`).

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/multi_traces.png)

# COMMAND ----------

country_md = "<|{country_date_df}|chart|type=bar|x=Date|y[1]=Deaths|y[2]=Recovered|y[3]=Confirmed|>"

# COMMAND ----------

Gui(country_md).run(dark_mode=False, run_browser=False, port=6004)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Style the graph with personalized properties
# MAGIC The _layout_ dictionnary specifies how bars should be displayed. They would be 'stacked'.
# MAGIC
# MAGIC The _options_ dictionary will change the opacity of the unselected markers.
# MAGIC
# MAGIC These are Plotly properties.

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/stack_chart.png)

# COMMAND ----------

layout = {'barmode':'stack'}
options = {"unselected":{"marker":{"opacity":0.5}}}
country_md = "<|{country_date_df}|chart|type=bar|x=Date|y[1]=Deaths|y[2]=Recovered|y[3]=Confirmed|layout={layout}|options={options}|>"

# COMMAND ----------

Gui(country_md).run(dark_mode=False, run_browser=False, port=6005)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add texts that sums up the data
# MAGIC
# MAGIC Use the [text](https://docs.taipy.io/en/latest/manuals/gui/viselements/text/) visual element.
# MAGIC
# MAGIC - Add the total number of Deaths (last line of _country_date_df_)
# MAGIC - Add the total number of Recovered (last line of _country_date_df_)
# MAGIC - Add the total number of Confirmed (last line of _country_date_df_)
# MAGIC

# COMMAND ----------

country_date_df

# COMMAND ----------

# MAGIC %md
# MAGIC This is how we can get the total number of Deaths from the dataset for India.

# COMMAND ----------

country_date_df.iloc[-1, 6] # gives the number of deaths for India (5 is for recovered and 4 is confirmed)

# COMMAND ----------

# MAGIC %md
# MAGIC Use the [text](https://docs.taipy.io/en/release-1.1/manuals/gui/viselements/text/) visual element. Note that between `{}`, any Python variable can be put but also any Python code.

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/control_text.png)

# COMMAND ----------

country_md = """
## Deaths <|{country_date_df.iloc[-1, 6]}|text|>

## Recovered <|{country_date_df.iloc[-1, 5]}|text|>

## Confirmed <|{country_date_df.iloc[-1, 4]}|text|>

<|{country_date_df}|chart|type=bar|x=Date|y[1]=Deaths|y[2]=Recovered|y[3]=Confirmed|layout={layout}|options={options}|>
"""

# COMMAND ----------

Gui(country_md).run(dark_mode=False, run_browser=False, port=6006)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local _on_change_
# MAGIC
# MAGIC - Add a [selector](https://docs.taipy.io/en/latest/manuals/gui/viselements/selector/) with `dropdown=True` containing the name of all the _Country/region_
# MAGIC - Give to the _on_change_ selector property the name of the _on_change_country_ function. This function will be called when the selector will be used.
# MAGIC - This function has a 'state' parameter and has to be completed. When the selector is used, this function is called with the _state_ argument. It contains all the Gui variables; 'state.country_date_df' is then the dataframe used in the Gui.

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/on_change_local.png)

# COMMAND ----------

country_lov = sorted(data["Country/Region"].dropna().unique().tolist())
selected_country = "India"

country_md = """
<|{selected_country}|selector|lov={country_lov}|on_change=on_change_country|dropdown|label=Country|>

## Deaths <|{country_date_df.iloc[-1, 6]}|>

## Recovered <|{country_date_df.iloc[-1, 5]}|>

## Confirmed <|{country_date_df.iloc[-1, 4]}|>

<|{country_date_df}|chart|type=bar|x=Date|y[1]=Deaths|y[2]=Recovered|y[3]=Confirmed|layout={layout}|options={options}|>
"""

# COMMAND ----------

def on_change_country(state):
    # state contains all the Gui variables and this is through this state variable that we can update the Gui
    # state.selected_country, state.country_date_df, ...
    # update country_date_df with the right country (use initialize_case_evolution)
    print("Chosen country: ", state.selected_country)
    state.country_date_df = initialize_case_evolution(data, state.selected_country)

# COMMAND ----------

Gui(country_md).run(dark_mode=False, run_browser=False, port=6007)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Layout
# MAGIC
# MAGIC Use the [layout](https://docs.taipy.io/en/latest/manuals/gui/viselements/layout/) block to change the page structure. This block creates invisible columns to put text/visual elements in.
# MAGIC
# MAGIC Syntax :
# MAGIC ```
# MAGIC <|layout|columns=1 1 1 ...|
# MAGIC (first column)
# MAGIC
# MAGIC (in second column)
# MAGIC
# MAGIC (third column)
# MAGIC (again, third column)
# MAGIC
# MAGIC (...)
# MAGIC |>
# MAGIC ```

# COMMAND ----------

final_country_md = """
<|layout|columns=1 1 1 1|
<|{selected_country}|selector|lov={country_lov}|on_change=on_change_country|dropdown|label=Country|>

## Deaths <|{country_date_df.iloc[-1, 6]}|>

## Recovered <|{country_date_df.iloc[-1, 5]}|>

## Confirmed <|{country_date_df.iloc[-1, 4]}|>
|>

<|{country_date_df}|chart|type=bar|x=Date|y[1]=Deaths|y[2]=Recovered|y[3]=Confirmed|layout={layout}|options={options}|>
"""

# COMMAND ----------

Gui(final_country_md).run(dark_mode=False, run_browser=False, port=6008)

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/layout.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Map

# COMMAND ----------

def initialize_map(data):
    data['Province/State'] = data['Province/State'].fillna(data["Country/Region"])
    data_province = data.groupby(["Country/Region",
                                  'Province/State',
                                  'Longitude',
                                  'Latitude'])\
                         .max()

    data_province_displayed = data_province[data_province['Deaths']>10].reset_index()

    data_province_displayed['Size'] = np.sqrt(data_province_displayed.loc[:,'Deaths']/data_province_displayed.loc[:,'Deaths'].max())*80 + 3
    data_province_displayed['Text'] = data_province_displayed.loc[:,'Deaths'].astype(str) + ' deaths</br>' + data_province_displayed.loc[:,'Province/State']
    return data_province_displayed

# COMMAND ----------

data_province_displayed = initialize_map(data)
data_province_displayed.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Properties to style the map
# MAGIC - marker color corresponds to the number of Deaths (column _Deaths_)
# MAGIC - marker sizes corresponds to the size in _Size_ column which is created from the number of Deaths
# MAGIC
# MAGIC layout_map permet defined the initial zoom and position of the map
# MAGIC

# COMMAND ----------

marker_map = {"color":"Deaths", "size": "Size", "showscale":True, "colorscale":"Viridis"}
layout_map = {
            "dragmode": "zoom",
            "mapbox": { "style": "open-street-map", "center": { "lat": 38, "lon": -90 }, "zoom": 3}
            }

# COMMAND ----------

# MAGIC %md
# MAGIC We give to Plotly:
# MAGIC - a map type
# MAGIC - the name of the latitude column
# MAGIC - the name of the longitude column
# MAGIC - properties: on the size and color of the markers
# MAGIC - the name of the column for the text of the points

# COMMAND ----------

selected_points = []
map_md = """
<|{data_province_displayed}|chart|type=scattermapbox|selected={selected_points}|lat=Latitude|lon=Longitude|marker={marker_map}|layout={layout_map}|text=Text|mode=markers|height=800px|options={options}|>
"""

# COMMAND ----------

Gui(map_md).run(dark_mode=False, run_browser=False, port=6009)

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/carte.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part and the _render_ property
# MAGIC - Create a [toggle](https://docs.taipy.io/en/latest/manuals/gui/viselements/toggle/) (works the same as a selector) with a lov of 'Map' an 'Country'
# MAGIC - Create two part blocks that renders or not depending on the value of the toggle
# MAGIC     - To do this, use the fact that in the _render_ property of the part block, Python code can be inserted in `{}`

# COMMAND ----------

representation_toggle = ['Map', 'Country']
selected_representation = representation_toggle[0]

# COMMAND ----------

main_page = """
<|{selected_representation}|toggle|lov={representation_toggle}|>

<|part|render={selected_representation == "Country"}|
"""+final_country_md+"""
|>

<|part|render={selected_representation == "Map"}|
"""+map_md+"""
|>
""" 

# COMMAND ----------

Gui(main_page).run(dark_mode=False, run_browser=False, port=6010)

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/part_render.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Taipy Core
# MAGIC Here are the functions that we are going to use to predict the number of Deaths for a country.
# MAGIC We will:
# MAGIC - preprocess the data (_preprocess_),
# MAGIC - create a training and testing database (_make_train_test_data_),
# MAGIC - train a model (_train_model_),
# MAGIC - generate predictions (_forecast_),
# MAGIC - generate a dataframe with the historical data and the predictions (_result_)
# MAGIC
# MAGIC ![](img/all_architecture.svg)

# COMMAND ----------

# initialise variables
selected_scenario = None
scenario_selector = None

first_date = dt.datetime(2020,11,1)

scenario_name = None

result = None

# COMMAND ----------

#Config.configure_job_executions(mode="standalone", nb_of_workers=2)

# COMMAND ----------


def add_features(data):
    dates = pd.to_datetime(data["Date"])
    data["Months"] = dates.dt.month
    data["Days"] = dates.dt.isocalendar().day
    data["Week"] = dates.dt.isocalendar().week
    data["Day of week"] = dates.dt.dayofweek
    return data

def create_train_data(final_data, date):
    bool_index = pd.to_datetime(final_data['Date']) <= date
    train_data = final_data[bool_index]
    return train_data

def preprocess(initial_data, country, date):
    data = initial_data.groupby(["Country/Region",'Date'])\
                       .sum()\
                       .dropna()\
                       .reset_index()

    final_data = data.loc[data['Country/Region']==country].reset_index(drop=True)
    final_data = final_data[['Date','Deaths']]
    final_data = add_features(final_data)
    
    train_data = create_train_data(final_data, date)
    return final_data, train_data


def train_arima(train_data):
    model = auto_arima(train_data['Deaths'],
                       start_p=1, start_q=1,
                       max_p=5, max_q=5,
                       start_P=0, seasonal=False,
                       d=1, D=1, trace=True,
                       error_action='ignore',  
                       suppress_warnings=True)
    model.fit(train_data['Deaths'])
    return model


def forecast(model):
    predictions = model.predict(n_periods=60)
    return np.array(predictions)


def result(final_data, predictions, date):
    dates = pd.to_datetime([date + dt.timedelta(days=i)
                            for i in range(len(predictions))])
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    predictions = pd.concat([pd.Series(dates, name="Date"),
                             pd.Series(predictions, name="Predictions")], axis=1)
    return final_data.merge(predictions, on="Date", how="outer")


def train_linear_regression(train_data):    
    y = train_data['Deaths']
    X = train_data.drop(['Deaths','Date'], axis=1)
    
    model = LinearRegression()
    model.fit(X,y)
    return model

def forecast_linear_regression(model, date):
    dates = pd.to_datetime([date + dt.timedelta(days=i)
                            for i in range(60)])
    X = add_features(pd.DataFrame({"Date":dates}))
    X.drop('Date', axis=1, inplace=True)
    predictions = model.predict(X)
    return predictions

# COMMAND ----------

# MAGIC %md
# MAGIC First we must define the Data Nodes then the tasks (associated to the Python function). Furthermore, we gather these tasks into different pipelines and these pipelines into a scenario.
# MAGIC
# MAGIC A Data Node needs a **unique id**. If needed, the storage type can be changed for CSV and SQL. Other parameters are then needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Nodes and Task for preprocess

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="img/preprocess.svg" alt="drawing" width="500"/>

# COMMAND ----------

initial_data_cfg = Config.configure_data_node(id="initial_data",
                                              storage_type="csv",
                                              path=path_to_data,
                                              cacheable=True,
                                              validity_period=dt.timedelta(days=5),
                                              scope=Scope.GLOBAL)

country_cfg = Config.configure_data_node(id="country", default_data="India",
                                         cacheable=True, validity_period=dt.timedelta(days=5))


date_cfg = Config.configure_data_node(id="date", default_data=dt.datetime(2020,10,10),
                                         cacheable=True, validity_period=dt.timedelta(days=5))

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="img/preprocess.svg" alt="drawing" width="500"/>

# COMMAND ----------

final_data_cfg =  Config.configure_data_node(id="final_data",
                                            cacheable=True, validity_period=dt.timedelta(days=5))


train_data_cfg =  Config.configure_data_node(id="train_data", cacheable=True, validity_period=dt.timedelta(days=5))


# COMMAND ----------

# MAGIC %md
# MAGIC <img src="img/preprocess.svg" alt="drawing" width="500"/>

# COMMAND ----------

task_preprocess_cfg = Config.configure_task(id="task_preprocess_data",
                                           function=preprocess,
                                           input=[initial_data_cfg, country_cfg, date_cfg],
                                           output=[final_data_cfg,train_data_cfg])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Nodes and Task for train_model

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="img/train_model.svg" alt="drawing" width="500"/>

# COMMAND ----------

model_cfg = Config.configure_data_node(id="model", cacheable=True, validity_period=dt.timedelta(days=5), scope=Scope.PIPELINE)

task_train_cfg = Config.configure_task(id="task_train",
                                      function=train_arima,
                                      input=train_data_cfg,
                                      output=model_cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Nodes and Task for forecast

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="img/forecast_arima.svg" alt="drawing" width="500"/>

# COMMAND ----------

predictions_cfg = Config.configure_data_node(id="predictions", scope=Scope.PIPELINE)

task_forecast_cfg = Config.configure_task(id="task_forecast",
                                      function=forecast,
                                      input=model_cfg,
                                      output=predictions_cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Nodes and Task for result

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="img/result.svg" alt="drawing" width="500"/>

# COMMAND ----------

result_cfg = Config.configure_data_node(id="result", scope=Scope.PIPELINE)

task_result_cfg = Config.configure_task(id="task_result",
                                      function=result,
                                      input=[final_data_cfg, predictions_cfg, date_cfg],
                                      output=result_cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Configuration of pipelines](https://docs.taipy.io/en/release-1.1/manuals/reference/taipy.Config/#taipy.core.config.config.Config.configure_default_pipeline)

# COMMAND ----------

pipeline_preprocessing_cfg = Config.configure_pipeline(id="pipeline_preprocessing",
                                                       task_configs=[task_preprocess_cfg])

pipeline_arima_cfg = Config.configure_pipeline(id="ARIMA",
                                               task_configs=[task_train_cfg,
                                                             task_forecast_cfg,
                                                             task_result_cfg])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add more models
# MAGIC
# MAGIC <img src="img/pipeline_linear_regression.svg" alt="drawing" width="500"/>

# COMMAND ----------

def train_linear_regression(train_data):    
    y = train_data['Deaths']
    X = train_data.drop(['Deaths','Date'], axis=1)
    
    model = LinearRegression()
    model.fit(X,y)
    return model

def forecast_linear_regression(model, date):
    dates = pd.to_datetime([date + dt.timedelta(days=i)
                            for i in range(60)])
    X = add_features(pd.DataFrame({"Date":dates}))
    X.drop('Date', axis=1, inplace=True)
    predictions = model.predict(X)
    return pd.Series(predictions)


task_train_linear_cfg = Config.configure_task(id="task_train_linear",
                                      function=train_linear_regression,
                                      input=train_data_cfg,
                                      output=model_cfg)

task_forecast_linear_cfg = Config.configure_task(id="task_forecast_linear",
                                      function=forecast_linear_regression,
                                      input=[model_cfg, date_cfg],
                                      output=predictions_cfg)

pipeline_linear_regression_cfg = Config.configure_pipeline(id="LinearRegression",
                                               task_configs=[task_train_linear_cfg,
                                                             task_forecast_linear_cfg,
                                                             task_result_cfg])

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Configuration of scénario](https://docs.taipy.io/en/release-1.1/manuals/reference/taipy.Config/#taipy.core.config.config.Config.configure_default_scenario)

# COMMAND ----------

scenario_cfg = Config.configure_scenario(id='scenario', pipeline_configs=[pipeline_preprocessing_cfg,
                                                                          pipeline_arima_cfg,
                                                                          pipeline_linear_regression_cfg])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creation and submit of scenario

# COMMAND ----------

tp.Core().run()
scenario = tp.create_scenario(scenario_cfg, name='First Scenario')
tp.submit(scenario)

# COMMAND ----------

scenario.initial_data.read()

# COMMAND ----------

scenario.train_data.read()

# COMMAND ----------

scenario.ARIMA.predictions.read()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Caching
# MAGIC Some job are skipped because no change has been done to the "input" Data Nodes.

# COMMAND ----------

tp.submit(scenario)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write in data nodes
# MAGIC
# MAGIC To write a data node:
# MAGIC
# MAGIC `<Data Node>.write(new_value)`

# COMMAND ----------

scenario.country.write('US')
tp.submit(scenario)
scenario.result.read()

# COMMAND ----------

scenario.ARIMA.predictions.read()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple framework

# COMMAND ----------

scenario = tp.create_scenario(scenario_cfg, name='Second Scenario')
tp.submit(scenario)

# COMMAND ----------

scenario.ARIMA.task_forecast.function

# COMMAND ----------

scenario.ARIMA.model.read()

# COMMAND ----------

scenario.pipelines['LinearRegression'].model.read()

# COMMAND ----------

[s.country.read() for s in tp.get_scenarios()]

# COMMAND ----------

[s.date.read() for s in tp.get_scenarios()]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Gui for the backend
# MAGIC _scenario_selector_ lets you choose a scenario and display its results.

# COMMAND ----------

scenario_selector = [(s.id, s.name) for s in tp.get_scenarios()]
selected_scenario = scenario.id
print(scenario_selector,'\n', selected_scenario)

# COMMAND ----------

result_arima = scenario.pipelines['ARIMA'].result.read()
result_rd = scenario.pipelines['LinearRegression'].result.read()
result = result_rd.merge(result_arima, on="Date", how="outer").sort_values(by='Date')
result

# COMMAND ----------

# MAGIC %md
# MAGIC **Tips** : the _value_by_id_ property if set to True for a selected will make _selected_scenario_ directly refer to the first element of the tupple (here the id)

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/predictions.png)

# COMMAND ----------

prediction_md = """
<|layout|columns=1 2 5 1 3|
<|{scenario_name}|input|label=Name|>

<br/>
<|Create|button|on_action=create_new_scenario|>

Prediction date
<|{first_date}|date|>

<|{selected_country}|selector|lov={country_lov}|dropdown|on_change=on_change_country|label=Country|>

<br/>
<|Submit|button|on_action=submit_scenario|>

<|{selected_scenario}|selector|lov={scenario_selector}|on_change=actualize_graph|dropdown|value_by_id|label=Scenario|>
|>

<|{result}|chart|x=Date|y[1]=Deaths_x|type[1]=bar|y[2]=Predictions_x|y[3]=Predictions_y|>
"""

# COMMAND ----------

def create_new_scenario(state):
    scenario = tp.create_scenario(scenario_cfg, name=state.scenario_name)
    state.scenario_selector += [(scenario.id, scenario.name)]

# COMMAND ----------

def submit_scenario(state):
    # 1) get the selected scenario
    # 2) write in country Data Node, the selected country
    # 3) submit the scenario
    # 4) actualize le graph avec actualize_graph
    scenario = tp.get(state.selected_scenario)
    scenario.country.write(state.selected_country)
    scenario.date.write(state.first_date.replace(tzinfo=None))
    tp.submit(scenario)
    actualize_graph(state)

# COMMAND ----------

def actualize_graph(state):
    # 1) update the result dataframe
    # 2) change selected_country with the predicted country of the scenario
    scenario = tp.get(state.selected_scenario)
    result_arima = scenario.pipelines['ARIMA'].result.read()
    result_rd = scenario.pipelines['LinearRegression'].result.read()
    if result_arima is not None and result_rd is not None:
        state.result = result_rd.merge(result_arima, on="Date", how="outer").sort_values(by='Date')
    state.selected_country = scenario.country.read()

# COMMAND ----------

Gui(prediction_md).run(dark_mode=False, port=5090)

# COMMAND ----------

# MAGIC %md
# MAGIC # Multi-pages and Taipy Rest

# COMMAND ----------

# MAGIC %md
# MAGIC To create a multi-pages app, we only need a dictionary with names as the keys and the Markdowns as the values.
# MAGIC
# MAGIC The _navbar_ control (<|navbar|>) has a default behaviour. It redirects to the different pages of the app automatically. Other solutions exists.

# COMMAND ----------

# MAGIC %md
# MAGIC ![](img/multi_pages.png)

# COMMAND ----------

navbar_md = "<center>\n<|navbar|>\n</center>"

pages = {
    "Map":navbar_md+map_md,
    "Country":navbar_md+final_country_md,
    "Predictions":navbar_md+prediction_md
}

rest = tp.Rest()

gui_multi_pages = Gui(pages=pages)
tp.run(gui_multi_pages, rest, dark_mode=False, port=6066)

# COMMAND ----------

