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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from Predictive_Heart_Attack_Model.Suprevised_Learning import organize_col

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

Heart_Data = Heart_Data.fillna('Null')

factor = Heart_Data.loc[:, Heart_Data.columns != 'Heart_Attack']

print(factor)

type(factor.column).value_counts()

def organize_col(df):
    factor_int=pd.DataFrame()
    factor_char=pd.DataFrame()
    factor_bool=pd.DataFrame()

    for column in df.columns:
        if type(int) == True:
            factor_int.append
        if type(chr) == True :
            factor_char.append
        if factor.column.astype(bool) == True :
            factor_bool.append



# X= Heart_Data["Obesity", "Heart-Attack-History","Cholesterol","Diet",]

target = Heart_Data["Heart_Attack"]
print(target)


kn=KNeighborsRegressor()
kn.fit(factor,target)
pre_1=kn.predict(factor)
print(pre_1)

lr=LinearRegression()
lr.fit(factor,target)
pre_2 = lr.predict(factor)
print(pre_2)