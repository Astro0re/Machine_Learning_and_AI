import urllib

import pandas
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# To use Heart Attack in Russia Data to create a predictive model

# X being the factors to base the prediction on
# X Factors = Heart Attack History, Cholesterol, Diet, Excercise Habits, Obesity and others

# Y being the factor to predict (prediction)

# Basic Packages

Heart_Data =pandas.read_csv (heart_attack_russia.csv)

X=Heart_Data["Obesity", "Heart-Attack-History","Cholesterol","Diet","Excercise-Habits"] 
Y= Heart_Data["Heart-Attack"]