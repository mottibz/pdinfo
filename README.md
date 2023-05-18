<!-- ![](img/AnalyticsTitle2.jpg) -->

<p align="center">
  <img width="500" height="150" src="img/PdInfo_logo.png">
</p>

# PdInfo - Pandas Extended Data Analysis

Being frustrated by not having one package to analyze a Pandas dataframe as I want, and developing the code for that, I decided to publish *pdinfo*, to save others time. 
It is inspired by packages like Sidetable and others and includes additional functionality and flexibility.

The first release includes functions to summarize a dataframe with lots of details, outlier detection, graphing, etc.

Following a standard installation, the functions are accessed through a Pandas accessor (.inf).

Below examples use the dataset:

"Appliances energy prediction Data Set":  
Data includes monitored house temperature and humidity conditions. Each wireless node transmitted the temperature and humidity conditions around 3.3 min. Then, the wireless data was averaged for 10 minutes periods. Weather from the nearest airport weather station was downloaded from a public dataset and merged together with the experimental datasets using the date and time column.  
https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

"Sales Prediction for Big Mart Outlets":  
BigMart have collected 2013 sales data for products across stores, in different cities. Certain attributes of each product and store have been defined. The aim is to build a predictive model and predict the sales of each product at a particular outlet, and understand the properties of products and outlets which play a key role in increasing sales.  
https://www.kaggle.com/datasets/mlrecipesbybharat/big-mart-sales-prediction?select=train_big_mart_sales_prediction.csv

