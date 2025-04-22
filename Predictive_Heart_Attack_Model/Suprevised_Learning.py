# Basic Packages
import urllib
import statistics
from statistics import linear_regression

import numpy as nu
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# To use Heart Attack in Russia Data to create a predictive model

# X being the factors to base the prediction on
# X Factors = Heart Attack History, Cholesterol, Diet, Excercise Habits, Obesity and others

# Y being the factor to predict (prediction)



# Data Exploration
Heart_Data =pd.read_csv ( r"C:\Users\USER\Documents\VSC\Git_\AI\Predictive_Heart_Attack_Model\heart_attack_russia.csv" )
print(Heart_Data.info())
print(Heart_Data.head())
print(Heart_Data.shape)
# print(Heart_Data.index)

# X= Heart_Data["Obesity", "Heart-Attack-History","Cholesterol","Diet",]
x= Heart_Data.loc[:, Heart_Data.columns != 'Heart-Attack']
Y= Heart_Data["Heart-Attack"]


kn=KNeighborsRegressor()
kn.fit(x,Y)
pre_1=kn.predict(x)

lr=LinearRegression()
lr.fit(x,Y)
per_1= lr.predict(x)